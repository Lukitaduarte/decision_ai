/// Adjusts option logits before the softmax. Implement it for other calibration schemes.
abstract interface class Calibrator {
  /// [logits] of one question; [options] is how many there are (abstention slot included).
  List<double> apply(List<double> logits, int options);
}

/// Divides logits by a temperature: one per option count when given, else a single one.
class TemperatureCalibrator implements Calibrator {
  const TemperatureCalibrator({this.temperature = 1.0, this.perOptionCount = const {}});

  /// From a manifest's `calibration` block: `{"temperature": 1.07}` and/or `{"per_option_count": {"2": 5.0, ...}}`.
  factory TemperatureCalibrator.fromJson(Map<String, dynamic> json) => TemperatureCalibrator(
    temperature: (json['temperature'] as num?)?.toDouble() ?? 1.0,
    perOptionCount: ((json['per_option_count'] as Map?) ?? const {}).map(
      (k, v) => MapEntry(int.parse(k.toString()), (v as num).toDouble()),
    ),
  );

  final double temperature;
  final Map<int, double> perOptionCount;

  @override
  List<double> apply(List<double> logits, int options) {
    final t = perOptionCount[options] ?? temperature;
    return t == 1.0 ? logits : logits.map((x) => x / t).toList();
  }
}
