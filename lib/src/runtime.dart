import 'dart:typed_data';

import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// Element type of a [Tensor].
enum TensorType { int64, float32, bool }

/// A plain tensor passed to or returned by a [Runtime]: flat data in row-major order plus its shape.
class Tensor {
  const Tensor._(this.type, this.shape, this.data);

  factory Tensor.int64(List<int> values, List<int> shape) =>
      Tensor._(TensorType.int64, shape, Int64List.fromList(values));
  factory Tensor.float32(List<double> values, List<int> shape) =>
      Tensor._(TensorType.float32, shape, Float32List.fromList(values));
  factory Tensor.bool(List<bool> values, List<int> shape) => Tensor._(TensorType.bool, shape, List<bool>.of(values));

  final TensorType type;
  final List<int> shape;

  /// `Int64List`, `Float32List` or `List<bool>`, depending on [type].
  final Object data;

  List<int> get ints => (data as Int64List);
  List<double> get doubles =>
      data is Float32List ? (data as Float32List) : (data as List).cast<num>().map((x) => x.toDouble()).toList();
  List<bool> get bools => (data as List<bool>);
}

/// Executes a model: named tensors in, named tensors out. The library ships [OnnxRuntimeBackend]; implement this to
/// plug another engine (llama.cpp, TFLite, Core ML, a platform channel of your own).
abstract interface class Runtime {
  List<String> get inputNames;
  List<String> get outputNames;
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs);
  Future<void> close();
}

/// ONNX Runtime through the `flutter_onnxruntime` plugin (ORT 1.23 on Android and iOS).
class OnnxRuntimeBackend implements Runtime {
  OnnxRuntimeBackend._(this._session);

  final OrtSession _session;

  /// Opens a model file on disk.
  static Future<OnnxRuntimeBackend> file(String path, {int? threads}) async => OnnxRuntimeBackend._(
    await OnnxRuntime().createSession(path, options: OrtSessionOptions(intraOpNumThreads: threads)),
  );

  /// Opens a model bundled as a Flutter asset.
  static Future<OnnxRuntimeBackend> asset(String key, {int? threads}) async => OnnxRuntimeBackend._(
    await OnnxRuntime().createSessionFromAsset(key, options: OrtSessionOptions(intraOpNumThreads: threads)),
  );

  @override
  List<String> get inputNames => _session.inputNames;
  @override
  List<String> get outputNames => _session.outputNames;

  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async {
    final ort = <String, OrtValue>{};
    for (final e in inputs.entries) {
      ort[e.key] = await OrtValue.fromList(e.value.data, e.value.shape);
    }
    final outputs = await _session.run(ort);
    final result = <String, Tensor>{};
    for (final e in outputs.entries) {
      final flat = await e.value.asFlattenedList();
      result[e.key] = switch (e.value.dataType) {
        OrtDataType.int64 ||
        OrtDataType.int32 => Tensor.int64(flat.cast<num>().map((x) => x.toInt()).toList(), e.value.shape),
        OrtDataType.bool => Tensor.bool(flat.map((x) => x == true || x == 1).toList(), e.value.shape),
        _ => Tensor.float32(flat.cast<num>().map((x) => x.toDouble()).toList(), e.value.shape),
      };
    }
    for (final v in [...ort.values, ...outputs.values]) {
      await v.dispose();
    }
    return result;
  }

  @override
  Future<void> close() => _session.close();
}
