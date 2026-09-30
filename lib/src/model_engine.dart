import 'dart:math' as math;

import 'calibrator.dart';
import 'decisions.dart';
import 'reader.dart';
import 'runtime.dart';

/// A model on the device: a [Runtime] runs it, a [Reader] turns questions into inputs and outputs into scores,
/// a [Calibrator] adjusts the scores. One forward pass per question.
class ModelEngine implements DecisionEngine {
  ModelEngine({required this.runtime, required this.reader, this.calibrator});

  final Runtime runtime;
  final Reader reader;
  final Calibrator? calibrator;

  @override
  Future<Map<String, Answer>> decide({Object? state, required Map<String, Question> questions}) async {
    // Encode everything first: a request that does not fit fails before any compute.
    final inputs = {for (final e in questions.entries) e.key: reader.encode(state, e.value)};
    final answers = <String, Answer>{};
    for (final e in inputs.entries) {
      final readout = reader.decode(e.value, await runtime.run(e.value.tensors, outputs: e.value.outputs));
      answers[e.key] = answerFrom(questions[e.key]!, e.value, readout, calibrator);
    }
    return answers;
  }

  @override
  Future<void> close() => runtime.close();
}

/// Softmax over the calibrated option logits, the abstention slot split off, and the answer for the question type.
/// Confidence: the reader's confidence head when there is one, else the chosen option's probability (with the
/// abstention mass still in the denominator).
Answer answerFrom(Question question, ReaderInput input, Readout readout, Calibrator? calibrator) {
  var z = readout.logits.take(input.optionCount).toList();
  if (calibrator != null) z = calibrator.apply(z, input.optionCount);
  final top = z.reduce(math.max);
  final e = z.map((x) => math.exp(x - top)).toList();
  final sum = e.reduce((a, b) => a + b);
  final p = e.map((x) => x / sum).toList();
  final ai = input.abstentionIndex;
  final abstention = ai == null ? null : p[ai];
  final raw = [
    for (var i = 0; i < p.length; i++)
      if (i != ai) p[i],
  ];
  final mass = raw.fold(0.0, (a, b) => a + b);
  final cond = mass > 0 ? raw.map((x) => x / mass).toList() : raw;
  var best = 0;
  for (var i = 1; i < cond.length; i++) {
    if (cond[i] > cond[best]) best = i;
  }
  final confLogit = readout.confidenceLogit;
  final confidence = confLogit != null ? 1 / (1 + math.exp(-confLogit)) : raw[best];
  switch (question) {
    case Noul():
      return Answer.noul(cond[1], abstention: abstention);
    case Choice():
      final labels = question.options.keys.toList();
      return Answer.choice(labels[best], confidence, {
        for (var i = 0; i < labels.length; i++) labels[i]: cond[i],
      }, abstention: abstention);
    case Score():
      var expected = 0.0;
      for (var i = 0; i < cond.length; i++) {
        expected += i * cond[i];
      }
      return Answer.score(expected, confidence, {
        for (var i = 0; i < cond.length; i++) '$i': cond[i],
      }, abstention: abstention);
  }
}
