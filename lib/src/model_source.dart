import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import 'manifest.dart';

/// Download progress: which file, bytes received, total bytes when known.
typedef DownloadProgress = void Function(String file, int received, int? total);

/// Where to download an on-device model from. The model is a folder with a `decision_ai.json` manifest
/// and the files it lists; it is downloaded once, checked, cached and reused offline.
sealed class ModelSource {
  const ModelSource();

  /// A Hugging Face repository (`org/model` or its link). Files are pinned to the commit [revision] resolves to.
  /// Pass [manifest] to use a repo that has no `decision_ai.json`.
  const factory ModelSource.huggingFace(
    String repo, {
    String revision,
    String manifestPath,
    ModelManifest? manifest,
    String? token,
  }) = HuggingFaceSource;

  /// Any HTTP folder (a CDN, a bucket, your server): `<baseUrl>/decision_ai.json` and the files it lists.
  /// Pass [manifest] to use a folder that has no `decision_ai.json`.
  const factory ModelSource.url(
    String baseUrl, {
    String manifestPath,
    ModelManifest? manifest,
    Map<String, String> headers,
  }) = UrlSource;

  /// Stable key for the cache folder.
  String get cacheKey;
  String get manifestPath;

  /// A manifest given in code; when set, nothing named [manifestPath] is downloaded.
  ModelManifest? get manifest;
  Map<String, String> get headers;

  /// URL of a file inside the model folder, at [version] (a commit for Hugging Face).
  Uri fileUri(String path, String version);

  /// The version to download (a commit for Hugging Face; the manifest's hash for plain URLs).
  Future<String> resolveVersion(http.Client client);
}

class HuggingFaceSource extends ModelSource {
  const HuggingFaceSource(
    this.repo, {
    this.revision = 'main',
    this.manifestPath = 'decision_ai.json',
    this.manifest,
    this.token,
  });
  final String repo;
  final String revision;
  @override
  final String manifestPath;
  @override
  final ModelManifest? manifest;
  final String? token;

  String get repoId {
    final m = RegExp(r'huggingface\.co/([^/\s]+/[^/\s?#]+)').firstMatch(repo.trim());
    return m != null ? m.group(1)! : repo.trim();
  }

  @override
  String get cacheKey => 'hf__${repoId.replaceAll('/', '__')}__$revision';
  @override
  Map<String, String> get headers => {if (token != null) 'Authorization': 'Bearer $token'};

  @override
  Uri fileUri(String path, String version) => Uri.parse('https://huggingface.co/$repoId/resolve/$version/$path');

  @override
  Future<String> resolveVersion(http.Client client) async {
    // Any file pins the commit; the manifest when the repo has one, else the model file.
    final probe = manifest?.modelFile ?? manifestPath;
    final req = http.Request('HEAD', fileUri(probe, revision))
      ..followRedirects = false
      ..headers.addAll(headers);
    final res = await client.send(req).timeout(const Duration(seconds: 20));
    await res.stream.drain<void>();
    final commit = res.headers['x-repo-commit'];
    if (res.statusCode >= 400 || commit == null) {
      throw HttpException('cannot resolve $repoId@$revision/$probe (HTTP ${res.statusCode})');
    }
    return commit;
  }
}

class UrlSource extends ModelSource {
  const UrlSource(this.baseUrl, {this.manifestPath = 'decision_ai.json', this.manifest, this.headers = const {}});
  final String baseUrl;
  @override
  final String manifestPath;
  @override
  final ModelManifest? manifest;
  @override
  final Map<String, String> headers;

  String get _base => baseUrl.endsWith('/') ? baseUrl : '$baseUrl/';

  @override
  String get cacheKey => 'url__${sha1.convert(utf8.encode(_base)).toString().substring(0, 16)}';

  @override
  Uri fileUri(String path, String version) => Uri.parse(_base).resolve(path);

  @override
  Future<String> resolveVersion(http.Client client) async {
    final inline = manifest;
    if (inline != null) {
      // No remote manifest: the version is the manifest given in code (change it to force a new download).
      final probe = await client
          .head(fileUri(inline.modelFile, ''), headers: headers)
          .timeout(const Duration(seconds: 20));
      if (probe.statusCode >= 400) throw HttpException('HTTP ${probe.statusCode} for ${inline.modelFile}');
      return sha1.convert(utf8.encode(jsonEncode(inline.json))).toString().substring(0, 16);
    }
    final res = await client.get(fileUri(manifestPath, ''), headers: headers).timeout(const Duration(seconds: 20));
    if (res.statusCode != 200) throw HttpException('HTTP ${res.statusCode} for $manifestPath');
    return sha1.convert(res.bodyBytes).toString().substring(0, 16);
  }
}

/// Local copy of a downloaded model.
class FetchedModel {
  FetchedModel(this.manifest, this.directory);
  final ModelManifest manifest;
  final String directory;
  String path(String file) => '$directory/$file';
}

/// Downloads what is missing and returns the local folder. Offline, the last complete download is used.
Future<FetchedModel> fetchModel(ModelSource source, {Directory? cacheDir, DownloadProgress? onProgress}) async {
  final root = cacheDir ?? Directory('${(await getApplicationSupportDirectory()).path}/decision_ai');
  final base = Directory('${root.path}/${source.cacheKey}');
  final ref = File('${base.path}/ref');
  final client = http.Client();
  try {
    String version;
    try {
      version = await source.resolveVersion(client);
    } on Exception {
      if (!await ref.exists()) rethrow;
      version = (await ref.readAsString()).trim(); // offline
    }
    final dir = Directory('${base.path}/$version');
    ModelManifest manifest;
    final inline = source.manifest;
    if (inline != null) {
      manifest = inline;
    } else {
      final manifestFile = File('${dir.path}/${source.manifestPath}');
      if (!await File('${manifestFile.path}.ok').exists()) {
        await _download(client, source, version, source.manifestPath, manifestFile, null, onProgress);
      }
      manifest = ModelManifest.parse(await manifestFile.readAsString());
    }
    for (final f in manifest.files) {
      final dest = File('${dir.path}/$f');
      if (!await File('${dest.path}.ok').exists()) {
        await _download(client, source, version, f, dest, manifest.sha256[f], onProgress);
      }
    }
    await ref.parent.create(recursive: true);
    await ref.writeAsString(version);
    return FetchedModel(manifest, dir.path);
  } finally {
    client.close();
  }
}

Future<void> _download(
  http.Client client,
  ModelSource source,
  String version,
  String path,
  File dest,
  String? expectedSha,
  DownloadProgress? onProgress,
) async {
  final uri = source.fileUri(path, version);
  var expected = expectedSha;
  if (expected == null && source is HuggingFaceSource) {
    // The Hub answers with the LFS object's SHA-256 before redirecting to storage.
    final head = http.Request('HEAD', uri)
      ..followRedirects = false
      ..headers.addAll(source.headers);
    final meta = await client.send(head).timeout(const Duration(seconds: 20));
    await meta.stream.drain<void>();
    final etag = meta.headers['x-linked-etag']?.replaceAll('"', '');
    if (etag != null && etag.length == 64) expected = etag;
  }
  final res = await client.send(http.Request('GET', uri)..headers.addAll(source.headers));
  if (res.statusCode != 200) throw HttpException('HTTP ${res.statusCode} for $path');
  await dest.parent.create(recursive: true);
  final part = File('${dest.path}.part');
  final sink = part.openWrite();
  final digest = _DigestSink();
  final hasher = sha256.startChunkedConversion(digest);
  var received = 0;
  await for (final chunk in res.stream) {
    sink.add(chunk);
    hasher.add(chunk);
    received += chunk.length;
    onProgress?.call(path, received, res.contentLength);
  }
  await sink.close();
  hasher.close();
  final got = digest.value.toString();
  if (expected != null && got != expected) {
    await part.delete();
    throw StateError('SHA-256 mismatch for $path: expected $expected, got $got');
  }
  await part.rename(dest.path);
  await File('${dest.path}.ok').writeAsString(got);
}

class _DigestSink implements Sink<Digest> {
  late Digest value;
  @override
  void add(Digest data) => value = data;
  @override
  void close() {}
}
