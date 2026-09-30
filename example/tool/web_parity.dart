import 'dart:convert';
import 'dart:math' as math;

import 'package:decision_ai/decision_ai.dart';
import 'package:decision_ai_example/main.dart' show models;
import 'package:flutter/widgets.dart';
import 'package:http/http.dart' as http;

/// Parity check for the browser: `?model=<name in the app's list>&golden=<json served next to the page>` loads that
/// model from Hugging Face on ONNX Runtime Web, answers the golden requests and prints one line to the console.
///
///     flutter build web -t tool/web_parity.dart -o build/web_parity --no-web-resources-cdn
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final params = Uri.base.queryParameters;
  final name = params['model']!;
  try {
    final option = models.firstWhere((m) => m.name == name);
    final sw = Stopwatch()..start();
    final ai = await option.load((_, _, _) {}, '', '');
    final loadMs = sw.elapsedMilliseconds;
    final cases = (jsonDecode((await http.get(Uri.base.resolve(params['golden']!))).body) as List)
        .cast<Map<String, dynamic>>();
    var maxDiff = 0.0, agree = 0;
    final times = <int>[];
    for (final c in cases) {
      final q = Question.fromJson((c['question'] as Map).cast<String, dynamic>());
      final t = Stopwatch()..start();
      final answer = (await ai.decide(state: c['state'], questions: {'q': q}))['q']!;
      times.add(t.elapsedMilliseconds);
      final expected = (c['expected']['probabilities'] as List).cast<num>().map((x) => x.toDouble()).toList();
      final got = answer.type == 'noul' ? [1 - answer.noul!, answer.noul!] : answer.probabilities.values.toList();
      for (var i = 0; i < expected.length; i++) {
        maxDiff = math.max(maxDiff, (got[i] - expected[i]).abs());
      }
      int argmax(List<double> p) => p.indexOf(p.reduce(math.max));
      if (argmax(got) == argmax(expected)) agree++;
    }
    times.sort();
    await ai.close();
    // ignore: avoid_print
    print(
      'WEB_PARITY model="$name" cases=${cases.length} argmax_agree=$agree max_dprob=${maxDiff.toStringAsExponential(2)} '
      'load_ms=$loadMs median_ms=${times[times.length ~/ 2]} p95_ms=${times[(times.length * 0.95).floor()]}',
    );
  } catch (e, st) {
    // ignore: avoid_print
    print('WEB_PARITY_ERROR model="$name" $e\n$st');
  }
}
