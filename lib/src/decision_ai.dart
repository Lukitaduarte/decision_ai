import 'dart:io';

import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

import 'api_engine.dart';
import 'calibrator.dart';
import 'decisions.dart';
import 'manifest.dart';
import 'model_engine.dart';
import 'model_source.dart';
import 'reader.dart';
import 'readers/label_logits_reader.dart';
import 'readers/option_reader.dart';
import 'runtime.dart';
import 'tokenizer.dart';

/// Opens the model file of a manifest: an asset key for [DecisionAI.local], a file path for [DecisionAI.remote].
typedef RuntimeOpener = Future<Runtime> Function(String location, ModelManifest manifest);

/// Entry point: the same [DecisionEngine] whether the model runs on the device or behind an API.
///
/// ```dart
/// final ai = await DecisionAI.local();                                       // bundled in the app
/// final ai = await DecisionAI.huggingFace('org/model');                     // downloaded once, cached
/// final ai = await DecisionAI.remote(ModelSource.url('https://cdn.example.com/model/'));
/// final ai = DecisionAI.api(endpoint: 'https://provider.example.com/v1/systemone', apiKey: key, model: 'm');
/// final ai = DecisionAI.openRouter(apiKey: key, model: 'provider/model');
/// final ai = DecisionAI.custom(runtime: myRuntime, reader: myReader);      // everything yours
/// final answers = await ai.decide(state: ..., questions: {...});
/// ```
///
/// On-device models are a [Runtime] (ONNX Runtime by default), a [Tokenizer] (byte-level BPE by default), a [Reader]
/// chosen by the manifest's `reader` name, and an optional [Calibrator]. Each piece can be replaced.
abstract final class DecisionAI {
  static final Map<String, ReaderFactory> _readers = {
    'option-reader': OptionReader.new,
    'label-logits': LabelLogitsReader.new,
  };

  /// Makes [factory] the reader for manifests whose `reader` is [name]. Built in: `option-reader` (encoders that
  /// score options at marker tokens) and `label-logits` (causal LMs scored on option keys).
  static void registerReader(String name, ReaderFactory factory) => _readers[name] = factory;

  /// Names of the readers available to manifests.
  static Iterable<String> get readers => _readers.keys;

  /// A model bundled with the app as Flutter assets: the manifest at [manifestPath] and the files it lists, relative
  /// to it. Pass [manifest] instead to bundle a model that has no `decision_ai.json` (files relative to [assetsDir]).
  static Future<DecisionEngine> local({
    String manifestPath = 'assets/model/decision_ai.json',
    ModelManifest? manifest,
    String assetsDir = 'assets/model/',
    int? threads,
    RuntimeOpener? openRuntime,
    Tokenizer? tokenizer,
    Calibrator? calibrator,
  }) async {
    final m = manifest ?? ModelManifest.parse(await rootBundle.loadString(manifestPath));
    final dir = manifest != null
        ? (assetsDir.isEmpty || assetsDir.endsWith('/') ? assetsDir : '$assetsDir/')
        : (manifestPath.contains('/') ? manifestPath.substring(0, manifestPath.lastIndexOf('/') + 1) : '');
    final tok = tokenizer ?? BpeTokenizer.fromJson(await rootBundle.loadString('$dir${m.tokenizerFile}'));
    final runtime = openRuntime != null
        ? await openRuntime('$dir${m.modelFile}', m)
        : await OnnxRuntimeBackend.asset('$dir${m.modelFile}', threads: threads);
    return _engine(m, runtime, tok, calibrator);
  }

  /// A model downloaded from [source] on first use, checked, cached, and reused offline.
  static Future<DecisionEngine> remote(
    ModelSource source, {
    int? threads,
    Directory? cacheDir,
    DownloadProgress? onProgress,
    RuntimeOpener? openRuntime,
    Tokenizer? tokenizer,
    Calibrator? calibrator,
  }) async {
    final fetched = await fetchModel(source, cacheDir: cacheDir, onProgress: onProgress);
    final m = fetched.manifest;
    final tok = tokenizer ?? BpeTokenizer.fromJson(await File(fetched.path(m.tokenizerFile)).readAsString());
    final path = fetched.path(m.modelFile);
    final runtime = openRuntime != null
        ? await openRuntime(path, m)
        : await OnnxRuntimeBackend.file(path, threads: threads);
    return _engine(m, runtime, tok, calibrator);
  }

  /// Shortcut for [remote] with a Hugging Face repository (`org/model` or its link).
  /// Pass [manifest] for a repo that has no `decision_ai.json`.
  static Future<DecisionEngine> huggingFace(
    String repo, {
    String revision = 'main',
    ModelManifest? manifest,
    String? token,
    int? threads,
    Directory? cacheDir,
    DownloadProgress? onProgress,
    RuntimeOpener? openRuntime,
    Tokenizer? tokenizer,
    Calibrator? calibrator,
  }) => remote(
    ModelSource.huggingFace(repo, revision: revision, manifest: manifest, token: token),
    threads: threads,
    cacheDir: cacheDir,
    onProgress: onProgress,
    openRuntime: openRuntime,
    tokenizer: tokenizer,
    calibrator: calibrator,
  );

  /// A model you assemble yourself: any [Runtime], any [Reader], an optional [Calibrator].
  static DecisionEngine custom({required Runtime runtime, required Reader reader, Calibrator? calibrator}) =>
      ModelEngine(runtime: runtime, reader: reader, calibrator: calibrator);

  static DecisionEngine _engine(ModelManifest m, Runtime runtime, Tokenizer tokenizer, Calibrator? calibrator) {
    final factory = _readers[m.reader];
    if (factory == null) {
      runtime.close();
      throw UnsupportedError(
        'no reader named "${m.reader}"; register one with DecisionAI.registerReader '
        '(available: ${_readers.keys.join(', ')})',
      );
    }
    final cal = calibrator ?? (m.calibration != null ? TemperatureCalibrator.fromJson(m.calibration!) : null);
    return ModelEngine(runtime: runtime, reader: factory(m, tokenizer), calibrator: cal);
  }

  /// Any provider of the Decision API wire format. [endpoint] is the full URL that receives the POST.
  static DecisionEngine api({
    required String endpoint,
    String? apiKey,
    String? model,
    Map<String, String> headers = const {},
    Map<String, dynamic> extra = const {},
    Duration timeout = const Duration(seconds: 60),
    http.Client? client,
  }) => ApiEngine(
    endpoint: Uri.parse(endpoint),
    apiKey: apiKey,
    model: model,
    headers: headers,
    extra: extra,
    timeout: timeout,
    client: client,
  );

  /// OpenRouter's Decisions API. [model] is an OpenRouter model id; [appName] and [appUrl] identify your app to
  /// OpenRouter (optional). Provider routing options (data policy, for example) go in [extra].
  static DecisionEngine openRouter({
    required String apiKey,
    required String model,
    String? appName,
    String? appUrl,
    Map<String, dynamic> extra = const {},
    Duration timeout = const Duration(seconds: 60),
    http.Client? client,
  }) => ApiEngine(
    endpoint: Uri.parse('https://openrouter.ai/api/alpha/decisions'),
    apiKey: apiKey,
    model: model,
    headers: {'X-Title': ?appName, 'HTTP-Referer': ?appUrl},
    extra: extra,
    timeout: timeout,
    client: client,
  );
}
