import 'dart:convert';

/// Describes an on-device decision model (`decision_ai.json`): its files, which [Reader] reads it, and that
/// reader's settings.
///
/// ```json
/// {"format": "decision-ai/model@1", "name": "…", "reader": "option-reader",
///  "files": {"model": "onnx/model.onnx", "tokenizer": "tokenizer.json"}, "sha256": {…},
///  "max_length": 8192, "calibration": {"temperature": 1.0}, "reader_config": {…}}
/// ```
///
/// The earlier format `decision-ai/option-reader@1` (settings at the top level) is still read.
class ModelManifest {
  ModelManifest._(this.json);

  static const format = 'decision-ai/model@1';
  static const legacyOptionReaderFormat = 'decision-ai/option-reader@1';

  final Map<String, dynamic> json;

  /// A manifest written in code, for a model that does not ship `decision_ai.json`.
  factory ModelManifest.custom({
    required String reader,
    required Map<String, String> files,
    required int maxLength,
    Map<String, dynamic> readerConfig = const {},
    Map<String, dynamic>? calibration,
    Map<String, String> sha256 = const {},
    String? name,
    String? license,
  }) => ModelManifest._({
    'format': format,
    'name': ?name,
    'license': ?license,
    'reader': reader,
    'files': files,
    'sha256': sha256,
    'max_length': maxLength,
    'calibration': ?calibration,
    'reader_config': readerConfig,
  });

  /// A manifest in code for an `option-reader` model (the Dinah protocol). Only the files and the context length are
  /// required; everything else defaults to the protocol.
  factory ModelManifest({
    required String model,
    required String tokenizer,
    required int maxLength,
    int padMultiple = 1,
    String? name,
    String? license,
    String clsToken = '[CLS]',
    String sepToken = '[SEP]',
    String padToken = '[PAD]',
    String optionToken = '[OPT]',
    Map<String, int> questionTypes = const {'noul': 0, 'choice': 1, 'score': 2},
    String noulFalseDefault = 'The statement above is false.',
    Map<String, String> inputs = const {},
    Map<String, String> outputs = const {},
    Map<String, String> sha256 = const {},
  }) => ModelManifest.custom(
    reader: 'option-reader',
    files: {'model': model, 'tokenizer': tokenizer},
    maxLength: maxLength,
    sha256: sha256,
    name: name,
    license: license,
    readerConfig: {
      'pad_multiple': padMultiple,
      'tokens': {'cls': clsToken, 'sep': sepToken, 'pad': padToken, 'option': optionToken},
      'question_types': questionTypes,
      'noul_false_default': noulFalseDefault,
      'inputs': inputs,
      'outputs': outputs,
    },
  );

  factory ModelManifest.parse(String source) {
    final json = jsonDecode(source) as Map<String, dynamic>;
    if (json['format'] == legacyOptionReaderFormat) {
      // Settings lived at the top level; files and max_length stay where they are.
      return ModelManifest._({...json, 'format': format, 'reader': 'option-reader', 'reader_config': json});
    }
    if (json['format'] != format) {
      throw FormatException('unsupported manifest format ${json['format']} (expected $format)');
    }
    return ModelManifest._(json);
  }

  String get name => json['name'] as String? ?? 'unnamed model';
  String? get license => json['license'] as String?;
  String get reader => json['reader'] as String;

  /// Files by role (`model`, `tokenizer`, and any the reader uses), relative to the manifest.
  Map<String, String> get fileMap => (json['files'] as Map<String, dynamic>).cast<String, String>();
  String get modelFile => fileMap['model']!;
  String get tokenizerFile => fileMap['tokenizer']!;
  List<String> get files => fileMap.values.toSet().toList();

  /// Optional SHA-256 per file, checked after download.
  Map<String, String> get sha256 => (json['sha256'] as Map<String, dynamic>? ?? const {}).cast<String, String>();

  int get maxLength => json['max_length'] as int;
  Map<String, dynamic>? get calibration => json['calibration'] as Map<String, dynamic>?;
  Map<String, dynamic> get readerConfig => json['reader_config'] as Map<String, dynamic>? ?? const {};
}
