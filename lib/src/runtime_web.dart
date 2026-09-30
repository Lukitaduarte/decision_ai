import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'runtime.dart';

/// ONNX Runtime Web (`ort`, loaded by the page), called directly. The plugin's web implementation cannot build int64
/// tensors (it passes plain numbers to `BigInt64Array.from`), and token ids are int64 in every model we run.
class WebOnnxRuntime implements Runtime {
  WebOnnxRuntime._(this._session);

  final JSObject _session;

  static JSObject get _ort {
    final ort = globalContext.getProperty<JSObject?>('ort'.toJS);
    if (ort == null) {
      throw StateError(
        'ONNX Runtime Web is not loaded: add onnxruntime-web 1.23.0 (ort.wasm.min.js) to web/index.html, before '
        'flutter_bootstrap.js',
      );
    }
    return ort;
  }

  /// Opens the model at [url] (fetched, and cached, by the browser).
  static Future<WebOnnxRuntime> open(String url) async {
    // Run inference in ONNX Runtime Web's worker: on the page's main thread, compiling a model and every forward
    // pass freeze the page (browsers then offer to kill it as unresponsive).
    _ort.getProperty<JSObject>('env'.toJS).getProperty<JSObject>('wasm'.toJS).setProperty('proxy'.toJS, true.toJS);
    final inferenceSession = _ort.getProperty<JSObject>('InferenceSession'.toJS);
    final options = JSObject()..setProperty('executionProviders'.toJS, <JSString>['wasm'.toJS].toJS);
    final session = await inferenceSession.callMethod<JSPromise<JSObject>>('create'.toJS, url.toJS, options).toDart;
    return WebOnnxRuntime._(session);
  }

  List<String> _names(String property) =>
      _session.getProperty<JSArray<JSString>>(property.toJS).toDart.map((s) => s.toDart).toList();

  @override
  List<String> get inputNames => _names('inputNames');
  @override
  List<String> get outputNames => _names('outputNames');

  static JSObject _toOrt(Tensor t) {
    final shape = <JSNumber>[for (final d in t.shape) d.toJS].toJS;
    final (String type, JSAny data) = switch (t.type) {
      TensorType.int64 => (
        'int64',
        globalContext
            .getProperty<JSObject>('BigInt64Array'.toJS)
            .callMethod<JSAny>('from'.toJS, <JSNumber>[for (final v in t.ints) v.toJS].toJS, globalContext['BigInt']),
      ),
      TensorType.float32 => ('float32', Float32List.fromList(t.doubles).toJS),
      TensorType.bool => ('bool', Uint8List.fromList([for (final b in t.bools) b ? 1 : 0]).toJS),
    };
    return _ort.getProperty<JSFunction>('Tensor'.toJS).callAsConstructor<JSObject>(type.toJS, data, shape);
  }

  static Tensor _fromOrt(JSObject t) {
    final type = t.getProperty<JSString>('type'.toJS).toDart;
    final shape = t.getProperty<JSArray<JSNumber>>('dims'.toJS).toDart.map((d) => d.toDartInt).toList();
    final data = t.getProperty<JSObject>('data'.toJS);
    // Array.from(data, Number) turns BigInt64Array, Uint8Array and Float32Array alike into plain numbers.
    final values = globalContext
        .getProperty<JSObject>('Array'.toJS)
        .callMethod<JSArray<JSNumber>>('from'.toJS, data, globalContext['Number'])
        .toDart;
    return switch (type) {
      'int64' || 'int32' => Tensor.int64([for (final v in values) v.toDartInt], shape),
      'bool' => Tensor.bool([for (final v in values) v.toDartDouble != 0], shape),
      _ => Tensor.float32([for (final v in values) v.toDartDouble], shape),
    };
  }

  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs, {List<String>? outputs}) async {
    final feeds = JSObject();
    for (final e in inputs.entries) {
      feeds.setProperty(e.key.toJS, _toOrt(e.value));
    }
    final wanted = outputs ?? outputNames;
    // Asking for the outputs by name also keeps ONNX Runtime Web from copying the others out of its worker.
    final results = await _session
        .callMethod<JSPromise<JSObject>>('run'.toJS, feeds, <JSString>[for (final n in wanted) n.toJS].toJS)
        .toDart;
    return {for (final name in wanted) name: _fromOrt(results.getProperty<JSObject>(name.toJS))};
  }

  @override
  Future<void> close() async {
    await _session.callMethod<JSPromise<JSAny?>>('release'.toJS).toDart;
  }
}

/// Opens [url] with ONNX Runtime Web.
Future<Runtime> openWebRuntime(String url) => WebOnnxRuntime.open(url);
