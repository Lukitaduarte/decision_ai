import 'dart:convert';
import 'dart:io';

import 'package:decision_ai/decision_ai.dart';
import 'package:flutter_test/flutter_test.dart';

/// The `laya` reader against Laya's own `build_sequence` (github.com/NandhaKishorM/laya), from
/// `tool/port_laya.py golden`: the same token ids and marker positions for every request.
void main() {
  final tokenizer = BpeTokenizer.fromJson(File('test/fixtures/tokenizer_laya.json').readAsStringSync());
  final manifest = ModelManifest.custom(
    reader: 'laya',
    files: {'model': 'm', 'tokenizer': 't'},
    maxLength: 512,
    readerConfig: {
      'head_max_len': 192,
      'temperature': [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
      'temperature_by_options': {'choice:11+': 0.10058280825614929, 'choice:2': 1.9063563346862793},
    },
  );
  final reader = LayaReader(manifest, tokenizer);

  test('inputs match Laya token for token', () {
    final cases = (jsonDecode(File('test/golden/laya_sequences.json').readAsStringSync()) as List)
        .cast<Map<String, dynamic>>();
    for (final c in cases) {
      final input = reader.encode(c['state'], Question.fromJson((c['question'] as Map).cast<String, dynamic>()));
      expect(input.tensors['input_ids']!.ints, c['ids'], reason: jsonEncode(c['question']));
      expect(input.tensors['marker_pos']!.ints, c['markers'], reason: jsonEncode(c['question']));
    }
    expect(cases, hasLength(64));
  });

  test('temperatures follow the type and option-count buckets, bounded to [0.5, 5]', () {
    expect(reader.temperature(0, 2), closeTo(1.906, 1e-3));
    expect(reader.temperature(0, 12), 0.5, reason: 'choice:11+ ships 0.1006; Laya refuses to sharpen that hard');
    expect(reader.temperature(2, 2), closeTo(1.983, 1e-3), reason: 'no bucket: the per-type temperature');
  });

  test('a state that does not fit raises instead of being cut', () {
    expect(() => reader.encode('word ' * 600, const Noul('It is long.')), throwsA(isA<RequestTooLong>()));
  });
}
