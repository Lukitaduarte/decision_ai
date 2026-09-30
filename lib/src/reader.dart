import 'decisions.dart';
import 'manifest.dart';
import 'runtime.dart';
import 'tokenizer.dart';

/// What a [Reader] prepared for one question: the model inputs and how many option scores to expect.
class ReaderInput {
  ReaderInput({required this.tensors, required this.optionCount, this.abstentionIndex, this.extra});

  /// Named inputs for the [Runtime].
  final Map<String, Tensor> tensors;

  /// Scores to read back, in option order (the question's options, then the abstention slot when there is one).
  final int optionCount;

  /// Index of an "insufficient evidence" slot among the scores, when the model has one.
  final int? abstentionIndex;

  /// Anything the reader wants back in [Reader.decode].
  final Object? extra;
}

/// Raw scores for one question, before calibration: one logit per option (in [ReaderInput] order), and an optional
/// confidence logit from a dedicated head.
class Readout {
  Readout(this.logits, {this.confidenceLogit});
  final List<double> logits;
  final double? confidenceLogit;
}

/// How a model family reads a typed question and how its outputs become option scores. This is the extension point
/// for new model families: implement it and register it with [DecisionAI.registerReader].
abstract interface class Reader {
  /// The options of [question] in the order the model sees them (labels come from the question).
  ReaderInput encode(Object? state, Question question);

  /// Option logits (and an optional confidence logit) from the runtime's outputs.
  Readout decode(ReaderInput input, Map<String, Tensor> outputs);
}

/// Builds a [Reader] for a model from its manifest and tokenizer.
typedef ReaderFactory = Reader Function(ModelManifest manifest, Tokenizer tokenizer);

/// The texts a question offers, in order: choice descriptions (or labels), noul [false, true], score levels.
List<Object?> optionTexts(Question q, {required Object? noulFalseDefault}) => switch (q) {
  Choice() => q.options.entries.map((e) => e.value ?? e.key).toList(),
  Noul() => [q.whenFalse ?? noulFalseDefault, q.whenTrue ?? q.instructions],
  Score() => q.levels,
};
