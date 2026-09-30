/// Typed decisions in the Decision API wire format: a state, and questions of three kinds.
sealed class Question {
  const Question({this.instructions});

  /// The question itself, as text or JSON.
  final Object? instructions;

  String get type;

  /// Builds a question from its wire JSON (`{"type": "choice", "instructions": ..., "criteria": ...}`).
  factory Question.fromJson(Map<String, dynamic> json) {
    final instructions = json['instructions'];
    final criteria = json['criteria'];
    switch (json['type']) {
      case 'choice':
        return Choice((criteria as Map).map((k, v) => MapEntry(k.toString(), v)), instructions: instructions);
      case 'noul':
        final c = (criteria as Map?) ?? const {};
        return Noul(instructions, whenTrue: c['true'], whenFalse: c['false']);
      case 'score':
        return Score((criteria as List).cast<Object?>(), instructions: instructions);
      default:
        throw ArgumentError('unknown question type ${json['type']}');
    }
  }

  /// The wire JSON of this question.
  Map<String, dynamic> toJson();
}

/// Pick one label. Each option maps a label to its description (null = the label describes itself).
class Choice extends Question {
  const Choice(this.options, {super.instructions});
  final Map<String, Object?> options;
  @override
  String get type => 'choice';
  @override
  Map<String, dynamic> toJson() => {
    'type': type,
    if (instructions != null) 'instructions': instructions,
    'criteria': options,
  };
}

/// Probability that a statement is true. Describe both poles for best results.
class Noul extends Question {
  const Noul(Object? statement, {this.whenTrue, this.whenFalse}) : super(instructions: statement);
  final Object? whenTrue;
  final Object? whenFalse;
  @override
  String get type => 'noul';
  @override
  Map<String, dynamic> toJson() => {
    'type': type,
    if (instructions != null) 'instructions': instructions,
    if (whenTrue != null || whenFalse != null)
      'criteria': {if (whenTrue != null) 'true': whenTrue, if (whenFalse != null) 'false': whenFalse},
  };
}

/// Expected value on an ordered rubric; levels go from lowest (0) to highest.
class Score extends Question {
  const Score(this.levels, {super.instructions});
  final List<Object?> levels;
  @override
  String get type => 'score';
  @override
  Map<String, dynamic> toJson() => {
    'type': type,
    if (instructions != null) 'instructions': instructions,
    'criteria': levels,
  };
}

/// An answer in the Decision API wire format.
class Answer {
  const Answer._(
    this.type, {
    this.choice,
    this.noul,
    this.score,
    this.confidence,
    this.probabilities = const {},
    this.abstention,
    this.raw,
  });

  factory Answer.choice(String choice, double confidence, Map<String, double> probabilities, {double? abstention}) =>
      Answer._('choice', choice: choice, confidence: confidence, probabilities: probabilities, abstention: abstention);

  factory Answer.noul(double probabilityTrue, {double? abstention}) =>
      Answer._('noul', noul: probabilityTrue, abstention: abstention);

  factory Answer.score(double score, double confidence, Map<String, double> probabilities, {double? abstention}) =>
      Answer._('score', score: score, confidence: confidence, probabilities: probabilities, abstention: abstention);

  /// Parses an answer returned by a provider; unknown fields stay in [raw].
  factory Answer.fromJson(Map<String, dynamic> json) => Answer._(
    json['type'] as String? ??
        (json.containsKey('noul')
            ? 'noul'
            : json.containsKey('choice')
            ? 'choice'
            : 'score'),
    choice: json['choice'] as String?,
    noul: (json['noul'] as num?)?.toDouble(),
    score: (json['score'] as num?)?.toDouble(),
    confidence: (json['confidence'] as num?)?.toDouble(),
    probabilities: ((json['probabilities'] as Map?) ?? const {}).map(
      (k, v) => MapEntry(k.toString(), (v as num).toDouble()),
    ),
    abstention: (json['abstention'] as num?)?.toDouble(),
    raw: json,
  );

  final String type;
  final String? choice;
  final double? noul;
  final double? score;
  final double? confidence;
  final Map<String, double> probabilities;

  /// Probability that the context is not enough to answer, for models with an abstention slot. [probabilities] and
  /// the answer are then conditional on the context being sufficient.
  final double? abstention;

  /// The provider's full answer, when it came from an API.
  final Map<String, dynamic>? raw;

  Map<String, dynamic> toJson() => {
    'type': type,
    if (choice != null) 'choice': choice,
    if (noul != null) 'noul': noul,
    if (score != null) 'score': score,
    if (confidence != null) 'confidence': confidence,
    if (probabilities.isNotEmpty) 'probabilities': probabilities,
    'abstention': ?abstention,
  };
}

/// Anything that answers typed decisions: a model on the device or an API.
abstract interface class DecisionEngine {
  /// Answers every question about [state]; keys are kept.
  Future<Map<String, Answer>> decide({Object? state, required Map<String, Question> questions});

  Future<void> close();
}

/// A request does not fit in the model's context. Nothing is ever truncated.
class RequestTooLong implements Exception {
  RequestTooLong(this.tokens, this.maxLength);
  final int tokens;
  final int maxLength;
  @override
  String toString() => 'RequestTooLong: $tokens tokens, above max_length=$maxLength (no truncation)';
}

/// A provider answered with an error.
class DecisionApiException implements Exception {
  DecisionApiException(this.statusCode, this.body);
  final int statusCode;
  final String body;
  @override
  String toString() =>
      'DecisionApiException: HTTP $statusCode ${body.length > 300 ? '${body.substring(0, 300)}…' : body}';
}
