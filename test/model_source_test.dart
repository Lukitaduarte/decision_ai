import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:decision_ai/decision_ai.dart';
import 'package:decision_ai/src/model_source.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

/// Serves [files] on 127.0.0.1 and counts the requests per path.
Future<(HttpServer, Map<String, int>)> serve(Map<String, List<int>> files) async {
  final hits = <String, int>{};
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  server.listen((req) {
    final path = req.uri.path.substring(1);
    hits[path] = (hits[path] ?? 0) + 1;
    final body = files[path];
    req.response.statusCode = body == null ? 404 : 200;
    req.response.contentLength = body?.length ?? 0;
    if (body != null && req.method != 'HEAD') req.response.add(body);
    req.response.close();
  });
  return (server, hits);
}

void main() {
  final model = utf8.encode('model bytes');
  final tokenizer = utf8.encode('{"tokenizer": true}');
  late Directory cache;

  setUp(() async => cache = await Directory.systemTemp.createTemp('decision_ai_source'));
  tearDown(() async => cache.delete(recursive: true));

  Map<String, dynamic> manifest({String? modelSha}) => {
    'format': 'decision-ai/model@1',
    'reader': 'option-reader',
    'files': {'model': 'onnx/model.onnx', 'tokenizer': 'tokenizer.json'},
    'sha256': {'onnx/model.onnx': ?modelSha},
    'max_length': 512,
  };

  test('downloads the manifest and its files, checks SHA-256, then reuses the cache offline', () async {
    final (server, hits) = await serve({
      'decision_ai.json': utf8.encode(jsonEncode(manifest(modelSha: sha256.convert(model).toString()))),
      'onnx/model.onnx': model,
      'tokenizer.json': tokenizer,
    });
    final source = ModelSource.url('http://127.0.0.1:${server.port}');
    final progress = <String>[];
    final first = await fetchModel(source, cacheDir: cache, onProgress: (file, received, total) => progress.add(file));
    expect(await File(first.path('onnx/model.onnx')).readAsBytes(), model);
    expect(await File(first.path('tokenizer.json')).readAsBytes(), tokenizer);
    expect(progress, containsAll(['onnx/model.onnx', 'tokenizer.json']));

    await fetchModel(source, cacheDir: cache);
    expect(hits['onnx/model.onnx'], 1, reason: 'a complete download is not fetched again');

    await server.close(force: true);
    final offline = await fetchModel(source, cacheDir: cache);
    expect(offline.directory, first.directory);
  });

  test('a file whose SHA-256 does not match is rejected', () async {
    final (server, _) = await serve({
      'decision_ai.json': utf8.encode(jsonEncode(manifest(modelSha: '0' * 64))),
      'onnx/model.onnx': model,
      'tokenizer.json': tokenizer,
    });
    await expectLater(
      fetchModel(ModelSource.url('http://127.0.0.1:${server.port}/'), cacheDir: cache),
      throwsA(anything),
    );
    await server.close(force: true);
  });

  test('a manifest given in code needs no decision_ai.json', () async {
    final (server, hits) = await serve({'onnx/model.onnx': model, 'tokenizer.json': tokenizer});
    final inline = ModelManifest(model: 'onnx/model.onnx', tokenizer: 'tokenizer.json', maxLength: 512);
    final fetched = await fetchModel(
      ModelSource.url('http://127.0.0.1:${server.port}', manifest: inline),
      cacheDir: cache,
    );
    expect(fetched.manifest.reader, 'option-reader');
    expect(hits.containsKey('decision_ai.json'), isFalse);
    await server.close(force: true);
  });

  test('without a cache, an unreachable source fails', () async {
    await expectLater(fetchModel(ModelSource.url('http://127.0.0.1:9'), cacheDir: cache), throwsA(anything));
  });

  test('Hugging Face sources accept a repo id or its link', () {
    final a = ModelSource.huggingFace('https://huggingface.co/org/model') as HuggingFaceSource;
    final b = ModelSource.huggingFace('org/model') as HuggingFaceSource;
    expect(a.repoId, 'org/model');
    expect(a.cacheKey, b.cacheKey);
    expect(
      b.fileUri('decision_ai.json', 'abc').toString(),
      'https://huggingface.co/org/model/resolve/abc/decision_ai.json',
    );
  });

  test('on the web, the manifest and tokenizer come over HTTP and the model is opened by URL', () async {
    final seen = <String>[];
    final client = MockClient((req) async {
      seen.add(req.url.toString());
      if (req.url.path.endsWith('decision_ai.json')) return http.Response(jsonEncode(manifest()), 200);
      if (req.url.path.endsWith('tokenizer.json')) return http.Response('{"t": 1}', 200);
      return http.Response('', 404);
    });
    final web = await fetchModelForWeb(ModelSource.huggingFace('org/model', revision: 'abc'), client: client);
    expect(web.tokenizerJson, '{"t": 1}');
    expect(web.modelUrl, 'https://huggingface.co/org/model/resolve/abc/onnx/model.onnx');
    expect(seen.first, 'https://huggingface.co/org/model/resolve/abc/decision_ai.json');
    final missing = MockClient((req) async => http.Response('', 404));
    await expectLater(
      fetchModelForWeb(ModelSource.url('https://cdn.example.com/m'), client: missing),
      throwsA(isA<http.ClientException>()),
    );
  });
}
