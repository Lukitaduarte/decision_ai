/// Typed decisions (choice, noul, score) from a model on the device or any Decision API provider.
///
/// On-device models are built from four replaceable pieces: a [Runtime], a [Tokenizer], a [Reader] (how a model
/// family reads a question and scores options) and a [Calibrator].
library;

export 'src/api_engine.dart' show ApiEngine;
export 'src/calibrator.dart';
export 'src/decision_ai.dart' show DecisionAI, RuntimeOpener;
export 'src/decisions.dart';
export 'src/manifest.dart' show ModelManifest;
export 'src/model_engine.dart' show ModelEngine, answerFrom;
export 'src/model_source.dart' show ModelSource, DownloadProgress, WebModel, fetchModelForWeb;
export 'src/reader.dart';
export 'src/readers/label_logits_reader.dart' show LabelLogitsReader;
export 'src/readers/laya_reader.dart' show LayaReader;
export 'src/readers/option_reader.dart' show OptionReader;
export 'src/render.dart' show pythonJson, render;
export 'src/runtime.dart';
export 'src/tokenizer.dart';
