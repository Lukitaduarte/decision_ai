import 'dart:convert';
import 'dart:io';

import 'package:decision_ai/decision_ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

/// A [Runtime] that returns fixed outputs and records what it was given.
class FakeRuntime implements Runtime {
  FakeRuntime(this.outputs);
  final Map<String, Tensor> Function(Map<String, Tensor> inputs) outputs;
  final calls = <Map<String, Tensor>>[];
  var closed = false;

  @override
  List<String> get inputNames => const [];
  @override
  List<String> get outputNames => const [];
  @override
  Future<Map<String, Tensor>> run(Map<String, Tensor> inputs) async {
    calls.add(inputs);
    return outputs(inputs);
  }

  @override
  Future<void> close() async => closed = true;
}

void main() {
  final dinahTok = BpeTokenizer.fromJson(File('test/fixtures/tokenizer.json').readAsStringSync());
  final smolTok = BpeTokenizer.fromJson(File('test/fixtures/tokenizer_smollm2.json').readAsStringSync());
  const card = Choice({
    'refund': 'The customer wants their money back',
    'replacement': 'The customer wants a new unit sent',
  }, instructions: 'What does the customer want?');

  group('decisions', () {
    test('questions round-trip through the wire format', () {
      for (final q in [
        card,
        const Noul('It is urgent.', whenTrue: 'Urgent.', whenFalse: 'Not urgent.'),
        const Score(['low', 'high'], instructions: 'Rate it.'),
      ]) {
        final back = Question.fromJson(q.toJson());
        expect(back.toJson(), q.toJson());
        expect(back.type, q.type);
      }
      expect(() => Question.fromJson({'type': 'rank'}), throwsArgumentError);
    });

    test('answers parse provider JSON and keep it raw', () {
      final a = Answer.fromJson({
        'choice': 'refund',
        'confidence': 0.9,
        'probabilities': {'refund': 0.9, 'replacement': 0.1},
        'provider_field': 1,
      });
      expect(a.type, 'choice');
      expect(a.probabilities['refund'], 0.9);
      expect(a.raw!['provider_field'], 1);
      expect(Answer.fromJson({'noul': 0.3}).type, 'noul');
      expect(Answer.noul(0.3, abstention: 0.1).toJson(), {'type': 'noul', 'noul': 0.3, 'abstention': 0.1});
    });
  });

  group('manifest', () {
    test('reads the current format', () {
      final m = ModelManifest.parse(
        jsonEncode({
          'format': 'decision-ai/model@1',
          'name': 'm',
          'reader': 'label-logits',
          'files': {'model': 'a.onnx', 'tokenizer': 't.json'},
          'max_length': 512,
          'calibration': {'temperature': 2.0},
        }),
      );
      expect(m.reader, 'label-logits');
      expect(m.modelFile, 'a.onnx');
      expect(m.files, ['a.onnx', 't.json']);
      expect(m.calibration, {'temperature': 2.0});
    });

    test('reads the legacy option-reader format', () {
      final m = ModelManifest.parse(
        jsonEncode({
          'format': 'decision-ai/option-reader@1',
          'files': {'model': 'a.onnx', 'tokenizer': 't.json'},
          'max_length': 8192,
          'pad_multiple': 64,
        }),
      );
      expect(m.reader, 'option-reader');
      expect(m.readerConfig['pad_multiple'], 64);
    });

    test('rejects unknown formats', () {
      expect(() => ModelManifest.parse('{"format": "x"}'), throwsFormatException);
    });
  });

  group('calibration and answers', () {
    test('temperature per option count, else global', () {
      final c = TemperatureCalibrator.fromJson({
        'temperature': 2.0,
        'per_option_count': {'3': 4.0},
      });
      expect(c.apply([4.0, 2.0], 2), [2.0, 1.0]);
      expect(c.apply([4.0, 2.0, 0.0], 3), [1.0, 0.5, 0.0]);
    });

    test('abstention is split off and the rest renormalized', () {
      final input = ReaderInput(tensors: const {}, optionCount: 3, abstentionIndex: 2);
      final a = answerFrom(card, input, Readout([0.0, 0.0, 0.0]), null);
      expect(a.abstention, closeTo(1 / 3, 1e-12));
      expect(a.probabilities.values, [closeTo(0.5, 1e-12), closeTo(0.5, 1e-12)]);
      expect(a.confidence, closeTo(1 / 3, 1e-12));
    });

    test('confidence head, noul and expected score', () {
      final two = ReaderInput(tensors: const {}, optionCount: 2);
      expect(answerFrom(card, two, Readout([0.0, 1.0], confidenceLogit: 0.0), null).confidence, 0.5);
      expect(answerFrom(const Noul('x'), two, Readout([0.0, 0.0]), null).noul, closeTo(0.5, 1e-12));
      final score = answerFrom(
        const Score(['a', 'b', 'c']),
        ReaderInput(tensors: const {}, optionCount: 3),
        Readout([0.0, 0.0, 0.0]),
        null,
      );
      expect(score.score, closeTo(1.0, 1e-12));
    });
  });

  group('option-reader', () {
    final manifest = ModelManifest(model: 'm.onnx', tokenizer: 't.json', maxLength: 8192, padMultiple: 64);

    test('builds the Dinah protocol inputs', () {
      final input = OptionReader(manifest, dinahTok).encode('The package arrived broken.', card);
      final ids = input.tensors['ids']!;
      expect(ids.shape, [1, 64]);
      expect(ids.ints.first, dinahTok.tokenId('[CLS]'));
      final positions = input.tensors['opt_positions']!.ints;
      expect(positions.length, 2);
      for (final p in positions) {
        expect(ids.ints[p], dinahTok.tokenId('[OPT]'));
      }
      expect(input.tensors['qtype']!.ints, [1]);
      expect(input.tensors['pad_mask']!.bools.where((b) => b).length, lessThan(64));
    });

    test('reads logits and the confidence head, and never truncates', () {
      final reader = OptionReader(manifest, dinahTok);
      final input = reader.encode('x', card);
      final out = reader.decode(input, {
        'logits': Tensor.float32([1, 2, -1e4], [1, 3]),
        'conf_logit': Tensor.float32([0.5], [1]),
      });
      expect(out.logits, [1.0, 2.0]);
      expect(out.confidenceLogit, 0.5);
      final tiny = ModelManifest(model: 'm', tokenizer: 't', maxLength: 8);
      expect(
        () => OptionReader(tiny, dinahTok).encode('a long state that does not fit', card),
        throwsA(isA<RequestTooLong>()),
      );
    });
  });

  group('label-logits', () {
    ModelManifest manifest([Map<String, dynamic> cfg = const {}]) => ModelManifest.custom(
      reader: 'label-logits',
      files: {'model': 'm', 'tokenizer': 't'},
      maxLength: 2048,
      readerConfig: cfg,
    );

    test('lists options under keys and scores the key tokens', () {
      final reader = LabelLogitsReader(manifest(), smolTok);
      final input = reader.encode('The package arrived broken.', card);
      final ids = input.tensors['input_ids']!.ints;
      expect(ids.length, greaterThan(20));
      final keys = input.extra as List<int>;
      expect(keys, [smolTok.encode('A').single, smolTok.encode('B').single]);
      final vocab = 49152;
      final last = List<double>.filled(vocab, 0)
        ..[keys[0]] = 1
        ..[keys[1]] = 3;
      expect(
        reader.decode(input, {
          'logits': Tensor.float32(last, [1, vocab]),
        }).logits,
        [1.0, 3.0],
      );
      // Full [1, length, vocab] logits: the last position is used.
      final full = [...List<double>.filled(vocab, 9), ...last];
      expect(
        reader.decode(input, {
          'logits': Tensor.float32(full, [1, 2, vocab]),
        }).logits,
        [1.0, 3.0],
      );
    });

    test('per-type settings, position ids and single-token keys', () {
      final reader = LabelLogitsReader(
        manifest({
          'by_type': {
            'noul': {
              'keys': ['No', 'Yes'],
              'key_prefix': ' ',
            },
          },
          'inputs': {'position_ids': 'position_ids'},
        }),
        smolTok,
      );
      final input = reader.encode('x', const Noul('It is urgent.'));
      expect(input.extra, [smolTok.encode(' No').single, smolTok.encode(' Yes').single]);
      expect(input.tensors['position_ids']!.ints.first, 0);
      final bad = LabelLogitsReader(
        manifest({
          'keys': ['Refund', 'Replacement is a long key'],
        }),
        smolTok,
      );
      expect(() => bad.encode('x', card), throwsStateError);
    });
  });

  group('engine', () {
    test('custom engine: one pass per question, calibrated answers', () async {
      final runtime = FakeRuntime(
        (inputs) => {
          'logits': Tensor.float32([0, 2, 0], [1, 3]),
          'conf_logit': Tensor.float32([3], [1]),
        },
      );
      final reader = OptionReader(
        ModelManifest(model: 'm', tokenizer: 't', maxLength: 8192, padMultiple: 64),
        dinahTok,
      );
      final ai = DecisionAI.custom(runtime: runtime, reader: reader, calibrator: const TemperatureCalibrator());
      final answers = await ai.decide(
        state: {'order': 1},
        questions: {'intent': card, 'urgent': const Noul('It is urgent.')},
      );
      expect(runtime.calls.length, 2);
      expect(answers['intent']!.choice, 'replacement');
      expect(answers['urgent']!.noul, greaterThan(0.5));
      await ai.close();
      expect(runtime.closed, isTrue);
    });

    test('readers are registered by name', () {
      expect(DecisionAI.readers, containsAll(['option-reader', 'label-logits']));
      DecisionAI.registerReader('mine', LabelLogitsReader.new);
      expect(DecisionAI.readers, contains('mine'));
    });

    test('tensors expose typed views', () {
      expect(Tensor.int64([1, 2], [2]).ints, [1, 2]);
      expect(Tensor.bool([true], [1]).bools, [true]);
      expect(Tensor.float32([0.5], [1]).doubles, [0.5]);
    });
  });

  group('api', () {
    test('posts the wire format with a bearer key and reads usage', () async {
      late http.Request sent;
      final client = MockClient((req) async {
        sent = req;
        return http.Response(
          jsonEncode({
            'answers': {
              'intent': {
                'type': 'choice',
                'choice': 'refund',
                'confidence': 0.8,
                'probabilities': {'refund': 0.8, 'replacement': 0.2},
              },
            },
            'usage': {'cost': 0.001},
          }),
          200,
        );
      });
      final ai = DecisionAI.api(
        endpoint: 'https://example.com/v1/decide',
        apiKey: 'k',
        model: 'm',
        extra: {'provider': 'x'},
        client: client,
      );
      final answers = await ai.decide(state: 'hi', questions: {'intent': card});
      final body = jsonDecode(sent.body) as Map<String, dynamic>;
      expect(sent.headers['Authorization'], 'Bearer k');
      expect(body['model'], 'm');
      expect(body['provider'], 'x');
      expect(body['questions']['intent']['type'], 'choice');
      expect(answers['intent']!.choice, 'refund');
      expect((ai as ApiEngine).lastUsage, {'cost': 0.001});
    });

    test('errors carry the status code', () async {
      final ai = DecisionAI.openRouter(
        apiKey: 'k',
        model: 'm',
        appName: 'test',
        client: MockClient((req) async => http.Response('nope', 401)),
      );
      await expectLater(
        ai.decide(questions: {'intent': card}),
        throwsA(isA<DecisionApiException>().having((e) => e.statusCode, 'status', 401)),
      );
      final empty = DecisionAI.api(endpoint: 'https://e', client: MockClient((req) async => http.Response('{}', 200)));
      await expectLater(empty.decide(questions: {'intent': card}), throwsA(isA<DecisionApiException>()));
    });
  });
}
