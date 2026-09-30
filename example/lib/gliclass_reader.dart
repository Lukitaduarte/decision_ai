import 'package:decision_ai/decision_ai.dart';

/// An extension example: a [Reader] for GLiClass-style encoders, written against Verdict
/// (heman10x/rlcd-modernbert-151m, Apache-2.0) and its `core/formatting.py`. It is not part of the library; it shows
/// how a model family the library does not know is plugged in.
///
/// Input: `[CLS] <<LABEL>>label_1 ... <<LABEL>>insufficient evidence <<SEP>> text [SEP]`; output: one logit per label,
/// in label order. The last label is an abstention slot.
///
/// ```dart
/// DecisionAI.registerReader('gliclass', GliClassReader.new);
/// final ai = await DecisionAI.huggingFace('heman10x/rlcd-modernbert-151m',
///     manifest: ModelManifest.custom(reader: 'gliclass', files: {'model': 'model.onnx', 'tokenizer': 'tokenizer.json'},
///         maxLength: 512),
///     calibrator: verdictCalibrator(calibratorJson));
/// ```
class GliClassReader implements Reader {
  GliClassReader(this.manifest, this.tokenizer)
    : _cls = tokenizer.tokenId('[CLS]')!,
      _sep = tokenizer.tokenId('[SEP]')!;

  static const abstentionLabel = 'insufficient evidence';

  final ModelManifest manifest;
  final Tokenizer tokenizer;
  final int _cls, _sep;

  @override
  ReaderInput encode(Object? state, Question question) {
    final context = render(state);
    final ask = render(question.instructions);
    final List<String> labels;
    final String text;
    switch (question) {
      case Choice():
        labels = [for (final e in question.options.entries) 'It is ${render(e.value ?? e.key)}'];
        text = ask.isEmpty ? context : 'Question: $ask\n\nContext:\n$context';
      case Score():
        // Levels are valued by their index; Verdict prints values as Python floats.
        labels = [for (var i = 0; i < question.levels.length; i++) '${render(question.levels[i])} (Value: $i.0)'];
        text = ask.isEmpty ? context : 'Question: $ask\n\nContext:\n$context';
      case Noul():
        // Verdict lists true before false; decode puts them back in the library's order.
        labels = ['true: $ask', 'false: not $ask'];
        text = 'Context:\n$context\n\nEvaluate proposition: $ask';
    }
    final prompt = '${[...labels, abstentionLabel].map((l) => '<<LABEL>>$l').join()}<<SEP>>$text';
    final ids = [_cls, ...tokenizer.encode(prompt), _sep];
    if (ids.length > manifest.maxLength) throw RequestTooLong(ids.length, manifest.maxLength);
    final n = labels.length + 1;
    return ReaderInput(
      optionCount: n,
      abstentionIndex: n - 1,
      extra: question is Noul,
      tensors: {
        'input_ids': Tensor.int64(ids, [1, ids.length]),
        'attention_mask': Tensor.int64(List.filled(ids.length, 1), [1, ids.length]),
      },
    );
  }

  @override
  Readout decode(ReaderInput input, Map<String, Tensor> outputs) {
    final z = outputs['logits']!.doubles.take(input.optionCount).toList();
    return Readout(input.extra == true ? [z[1], z[0], z[2]] : z);
  }
}

/// Verdict's `calibrator.json`: a temperature per label count (`per_k`, abstention included), else a global one.
Calibrator verdictCalibrator(Map<String, dynamic> json) => TemperatureCalibrator(
  temperature: (json['temperature'] as num).toDouble(),
  perOptionCount: (json['per_k'] as Map).map((k, v) => MapEntry(int.parse(k as String), (v as num).toDouble())),
);
