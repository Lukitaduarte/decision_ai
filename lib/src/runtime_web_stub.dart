import 'runtime.dart';

/// Only the web build has ONNX Runtime Web; native platforms use [OnnxRuntimeBackend].
Future<Runtime> openWebRuntime(String url) => throw UnsupportedError('ONNX Runtime Web is only available on the web');
