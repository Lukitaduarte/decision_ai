import 'dart:convert';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:decision_ai/decision_ai.dart';
import 'package:dinah_chess/dinah_player.dart';
import 'package:dinah_chess/main.dart' show dinahRepo, dinahRevision;
import 'package:flutter/widgets.dart';

import 'golden.dart';

/// Parity check for the browser, without the test harness: Dinah-0 from Hugging Face on ONNX Runtime Web against the
/// Python reference, plus one move through the demo's player. The result goes to the console as one line.
///
///     flutter build web -t tool/web_parity.dart -o build/web_parity --no-web-resources-cdn
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    // Tokenizer check: the reader's unpadded length against Python's, per request.
    final web = await fetchModelForWeb(ModelSource.huggingFace(dinahRepo, revision: dinahRevision));
    final reader = OptionReader(web.manifest, BpeTokenizer.fromJson(web.tokenizerJson));
    final golden = (jsonDecode(goldenJson) as List).cast<Map<String, dynamic>>();
    var tokenMismatch = 0;
    final firstMismatch = <String>[];
    for (final c in golden) {
      final input = reader.encode(c['state'], Question.fromJson((c['question'] as Map).cast<String, dynamic>()));
      final n = input.tensors['pad_mask']!.bools.where((b) => b).length;
      if (n != c['expected']['tokens']) {
        tokenMismatch++;
        if (firstMismatch.length < 3) firstMismatch.add('${c['expected']['tokens']}->$n');
      }
    }
    // ignore: avoid_print
    print('WEB_PARITY_TOKENS mismatched=$tokenMismatch of ${golden.length} $firstMismatch');
    final sw = Stopwatch()..start();
    final ai = await DecisionAI.huggingFace(dinahRepo, revision: dinahRevision);
    final loadMs = sw.elapsedMilliseconds;
    final cases = (jsonDecode(goldenJson) as List).cast<Map<String, dynamic>>();
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
    final game = ch.Chess();
    final t = Stopwatch()..start();
    final turn = await DinahPlayer(ai, random: math.Random(1)).choose(game);
    // ignore: avoid_print
    print(
      'WEB_PARITY cases=${cases.length} argmax_agree=$agree max_dprob=${maxDiff.toStringAsExponential(2)} '
      'load_ms=$loadMs median_ms=${times[times.length ~/ 2]} p95_ms=${times[(times.length * 0.95).floor()]} '
      'first_move=${game.move_to_san(turn.move)} questions=${turn.questions} move_ms=${t.elapsedMilliseconds}',
    );
    await ai.close();
  } catch (e, st) {
    // ignore: avoid_print
    print('WEB_PARITY_ERROR $e\n$st');
  }
}
