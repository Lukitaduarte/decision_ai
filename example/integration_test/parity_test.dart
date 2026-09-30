import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:decision_ai/decision_ai.dart';
import 'package:decision_ai_example/gliclass_reader.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

/// The three ways to get an engine, checked on a device or simulator.
///
/// Before running: serve `example/assets/model/` on 127.0.0.1:8765 (python3 -m http.server 8765) and start
/// `tool/mock_decision_server.py` on 127.0.0.1:8766 with key `test-key`. For the LLM test, serve the folder written by
/// `tool/export_causal_lm.py` (export, then golden into `golden_llm.json`) on 127.0.0.1:8767. For the GLiClass example
/// reader, serve the folder written by `example/tool/gliclass_golden.py` (prepare, then golden) on 127.0.0.1:8768.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const card = Choice({
    'refund': 'The customer wants their money back',
    'replacement': 'The customer wants a new unit sent',
    'complaint': 'The customer only wants to complain',
  }, instructions: 'What does the customer want?');
  const cardState = 'The package arrived broken and I need it for tomorrow.';

  testWidgets('local: bundled model matches the Python reference on 64 golden decisions', (tester) async {
    final t0 = DateTime.now();
    final ai = await DecisionAI.local(threads: 4);
    final loadMs = DateTime.now().difference(t0).inMilliseconds;
    final cases = (jsonDecode(await rootBundle.loadString('assets/golden_decisions.json')) as List)
        .cast<Map<String, dynamic>>();
    var maxDiff = 0.0, maxConfDiff = 0.0, agree = 0;
    final times = <int>[];
    for (final c in cases) {
      final q = Question.fromJson((c['question'] as Map).cast<String, dynamic>());
      final sw = Stopwatch()..start();
      final answer = (await ai.decide(state: c['state'], questions: {'q': q}))['q']!;
      times.add(sw.elapsedMilliseconds);
      final expected = (c['expected']['probabilities'] as List).cast<num>().map((x) => x.toDouble()).toList();
      final got = answer.type == 'noul' ? [1 - answer.noul!, answer.noul!] : answer.probabilities.values.toList();
      for (var i = 0; i < expected.length; i++) {
        maxDiff = math.max(maxDiff, (got[i] - expected[i]).abs());
      }
      int argmax(List<double> p) => p.indexOf(p.reduce(math.max));
      if (argmax(got) == argmax(expected)) agree++;
      if (answer.confidence != null) {
        maxConfDiff = math.max(maxConfDiff, (answer.confidence! - (c['expected']['confidence'] as num)).abs());
      }
    }
    times.sort();
    // ignore: avoid_print
    print(
      'DECISION_AI_LOCAL cases=${cases.length} argmax_agree=$agree max_dprob=${maxDiff.toStringAsExponential(2)} '
      'max_dconf=${maxConfDiff.toStringAsExponential(2)} load_ms=$loadMs median_ms=${times[times.length ~/ 2]} p95_ms=${times[(times.length * 0.95).floor()]}',
    );
    expect(agree, cases.length);
    expect(maxDiff, lessThan(2e-3));
    expect(maxConfDiff, lessThan(2e-3));
    await ai.close();
  });

  testWidgets('remote: downloads a model folder from a URL, checks it, caches it and answers', (tester) async {
    final cache = await Directory.systemTemp.createTemp('decision_ai_remote');
    final source = ModelSource.url('http://127.0.0.1:8765/');
    final t0 = DateTime.now();
    final ai = await DecisionAI.remote(source, cacheDir: cache, threads: 4);
    final firstMs = DateTime.now().difference(t0).inMilliseconds;
    final answer = (await ai.decide(state: cardState, questions: {'intent': card}))['intent']!;
    await ai.close();
    final t1 = DateTime.now();
    final again = await DecisionAI.remote(source, cacheDir: cache, threads: 4);
    final cachedMs = DateTime.now().difference(t1).inMilliseconds;
    await again.close();
    // ignore: avoid_print
    print('DECISION_AI_REMOTE first_load_ms=$firstMs cached_load_ms=$cachedMs answer=${jsonEncode(answer.toJson())}');
    expect(answer.choice, 'replacement');
    await cache.delete(recursive: true);
  });

  testWidgets('huggingFace: downloads Dinah-0 through its decision_ai.json manifest and answers', (tester) async {
    final cache = await Directory.systemTemp.createTemp('decision_ai_hf');
    var shown = 0;
    final t0 = DateTime.now();
    final ai = await DecisionAI.huggingFace(
      'https://huggingface.co/Lukitaduarte/dinah-0',
      cacheDir: cache,
      threads: 4,
      onProgress: (file, received, total) {
        if (received - shown > 50 << 20 || received == total) {
          shown = received;
          // ignore: avoid_print
          print('download $file $received/${total ?? '?'}');
        }
      },
    );
    final firstMs = DateTime.now().difference(t0).inMilliseconds;
    final answer = (await ai.decide(state: cardState, questions: {'intent': card}))['intent']!;
    await ai.close();
    final t1 = DateTime.now();
    final again = await DecisionAI.huggingFace('Lukitaduarte/dinah-0', cacheDir: cache, threads: 4);
    final cachedMs = DateTime.now().difference(t1).inMilliseconds;
    await again.close();
    // ignore: avoid_print
    print('DECISION_AI_HF first_load_ms=$firstMs cached_load_ms=$cachedMs answer=${jsonEncode(answer.toJson())}');
    expect(answer.choice, 'replacement');
    await cache.delete(recursive: true);
  });

  testWidgets('inline manifest: a repo revision without decision_ai.json, described in code', (tester) async {
    final cache = await Directory.systemTemp.createTemp('decision_ai_inline');
    // Dinah-0 before the manifest was published: only onnx/model_int8.onnx and tokenizer.json.
    final ai = await DecisionAI.huggingFace(
      'Lukitaduarte/dinah-0',
      revision: 'e6bfc332fd31b6625cc912441c92f4a11dcb1266',
      manifest: ModelManifest(
        model: 'onnx/model_int8.onnx',
        tokenizer: 'tokenizer.json',
        maxLength: 8192,
        padMultiple: 64,
      ),
      cacheDir: cache,
      threads: 4,
    );
    final answer = (await ai.decide(state: cardState, questions: {'intent': card}))['intent']!;
    await ai.close();
    // ignore: avoid_print
    print('DECISION_AI_INLINE answer=${jsonEncode(answer.toJson())}');
    expect(answer.choice, 'replacement');
    await cache.delete(recursive: true);
  });

  testWidgets('label-logits: a causal LM exported by tool/export_causal_lm.py matches the Python reference', (
    tester,
  ) async {
    final cache = await Directory.systemTemp.createTemp('decision_ai_llm');
    final t0 = DateTime.now();
    final ai = await DecisionAI.remote(ModelSource.url('http://127.0.0.1:8767/'), cacheDir: cache, threads: 4);
    final loadMs = DateTime.now().difference(t0).inMilliseconds;
    final http = HttpClient();
    final res = await (await http.getUrl(Uri.parse('http://127.0.0.1:8767/golden_llm.json'))).close();
    final cases = (jsonDecode(await res.transform(utf8.decoder).join()) as List).cast<Map<String, dynamic>>();
    http.close();
    var maxDiff = 0.0, agree = 0;
    final times = <int>[];
    for (final c in cases) {
      final q = Question.fromJson((c['question'] as Map).cast<String, dynamic>());
      final sw = Stopwatch()..start();
      final answer = (await ai.decide(state: c['state'], questions: {'q': q}))['q']!;
      times.add(sw.elapsedMilliseconds);
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
      'DECISION_AI_LLM cases=${cases.length} argmax_agree=$agree max_dprob=${maxDiff.toStringAsExponential(2)} '
      'first_load_ms=$loadMs median_ms=${times[times.length ~/ 2]} p95_ms=${times[(times.length * 0.95).floor()]}',
    );
    expect(agree, cases.length);
    expect(maxDiff, lessThan(2e-3));
    await cache.delete(recursive: true);
  });

  testWidgets('custom reader: Verdict (GLiClass) through a reader registered by the app', (tester) async {
    Future<Object?> getJson(String path) async {
      final http = HttpClient();
      final res = await (await http.getUrl(Uri.parse('http://127.0.0.1:8768/$path'))).close();
      final body = await res.transform(utf8.decoder).join();
      http.close();
      return jsonDecode(body);
    }

    DecisionAI.registerReader('gliclass', GliClassReader.new);
    final cache = await Directory.systemTemp.createTemp('decision_ai_gliclass');
    final manifest = ModelManifest.custom(
      reader: 'gliclass',
      files: {'model': 'model.onnx', 'tokenizer': 'tokenizer.json'},
      maxLength: 512,
      name: 'verdict',
    );
    final t0 = DateTime.now();
    final ai = await DecisionAI.remote(
      ModelSource.url('http://127.0.0.1:8768/', manifest: manifest),
      cacheDir: cache,
      threads: 4,
      calibrator: verdictCalibrator((await getJson('calibrator.json'))! as Map<String, dynamic>),
    );
    final loadMs = DateTime.now().difference(t0).inMilliseconds;
    final cases = ((await getJson('golden_gliclass.json'))! as List).cast<Map<String, dynamic>>();
    var maxDiff = 0.0, agree = 0;
    final times = <int>[];
    for (final c in cases) {
      final q = Question.fromJson((c['question'] as Map).cast<String, dynamic>());
      final sw = Stopwatch()..start();
      final answer = (await ai.decide(state: c['state'], questions: {'q': q}))['q']!;
      times.add(sw.elapsedMilliseconds);
      // Verdict's order: its options (noul: true, false), then abstention. Compare conditional probabilities.
      final p = (c['expected']['probabilities'] as List).cast<num>().map((x) => x.toDouble()).toList();
      final abst = p.removeLast();
      final expected = [for (final x in p) x / (1 - abst)];
      final got = answer.type == 'noul' ? [answer.noul!, 1 - answer.noul!] : answer.probabilities.values.toList();
      for (var i = 0; i < expected.length; i++) {
        maxDiff = math.max(maxDiff, (got[i] - expected[i]).abs());
      }
      maxDiff = math.max(maxDiff, (answer.abstention! - abst).abs());
      int argmax(List<double> v) => v.indexOf(v.reduce(math.max));
      if (argmax(got) == argmax(expected)) agree++;
    }
    times.sort();
    await ai.close();
    // ignore: avoid_print
    print(
      'DECISION_AI_GLICLASS cases=${cases.length} argmax_agree=$agree max_dprob=${maxDiff.toStringAsExponential(2)} '
      'first_load_ms=$loadMs median_ms=${times[times.length ~/ 2]} p95_ms=${times[(times.length * 0.95).floor()]}',
    );
    expect(agree, cases.length);
    expect(maxDiff, lessThan(2e-3));
    await cache.delete(recursive: true);
  });

  testWidgets('api: any Decision API provider (local mock), bearer key checked', (tester) async {
    final ai = DecisionAI.api(endpoint: 'http://127.0.0.1:8766/v1/systemone', apiKey: 'test-key', model: 'dinah-0');
    final answer = (await ai.decide(
      state: cardState,
      questions: {
        'intent': card,
        'cancel': const Noul(
          'Does the customer want to cancel?',
          whenTrue: 'The customer wants to cancel.',
          whenFalse: 'The customer does not want to cancel.',
        ),
      },
    ));
    // ignore: avoid_print
    print('DECISION_AI_API answers=${jsonEncode(answer.map((k, v) => MapEntry(k, v.toJson())))}');
    expect(answer['intent']!.choice, 'replacement');
    expect(answer['cancel']!.noul, isNotNull);
    await ai.close();
    final wrongKey = DecisionAI.api(endpoint: 'http://127.0.0.1:8766/v1/systemone', apiKey: 'wrong');
    await expectLater(
      wrongKey.decide(state: cardState, questions: {'intent': card}),
      throwsA(isA<DecisionApiException>().having((e) => e.statusCode, 'status', 401)),
    );
    await wrongKey.close();
  });
}
