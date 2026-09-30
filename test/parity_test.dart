import 'dart:convert';
import 'dart:io';

import 'package:decision_ai/src/render.dart';
import 'package:decision_ai/src/tokenizer.dart';
import 'package:flutter_test/flutter_test.dart';

/// Parity with the Python reference (tool/make_golden.py): every id and every rendered string must match.
void main() {
  final tokenizer = BpeTokenizer.fromJson(File('test/fixtures/tokenizer.json').readAsStringSync());

  bool same(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  for (final name in ['', 'smollm2', 'qwen']) {
    final suffix = name.isEmpty ? '' : '_$name';
    test('tokenizer${name.isEmpty ? '' : ' ($name)'} matches Hugging Face tokenizers on every golden text', () {
      final tok = name.isEmpty
          ? tokenizer
          : BpeTokenizer.fromJson(File('test/fixtures/tokenizer$suffix.json').readAsStringSync());
      final cases = (jsonDecode(File('test/golden/tokenizer$suffix.json').readAsStringSync()) as List)
          .cast<Map<String, dynamic>>();
      final failures = <String>[];
      for (final c in cases) {
        final expected = (c['ids'] as List).cast<int>();
        final got = tok.encode(c['text'] as String);
        if (!same(got, expected)) {
          final shown = jsonEncode(c['text']);
          failures.add(
            '${shown.substring(0, shown.length.clamp(0, 80))}\n  expected ${expected.take(20).toList()}\n  got      ${got.take(20).toList()}',
          );
        }
      }
      expect(
        failures,
        isEmpty,
        reason: '${failures.length} of ${cases.length} differ:\n${failures.take(8).join('\n')}',
      );
    });
  }

  test('render matches Python json.dumps(sort_keys, compact, ensure_ascii=False) and str.strip', () {
    final cases = (jsonDecode(File('test/golden/render.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();
    final failures = <String>[];
    for (final c in cases) {
      final got = render(c['value']);
      if (got != c['rendered']) failures.add('expected ${c['rendered']}\n  got      $got');
    }
    expect(failures, isEmpty, reason: '${failures.length} of ${cases.length} differ:\n${failures.take(8).join('\n')}');
  });

  test('special tokens resolve to the ids the model was trained with', () {
    expect(tokenizer.tokenId('[CLS]'), 50281);
    expect(tokenizer.tokenId('[SEP]'), 50282);
    expect(tokenizer.tokenId('[PAD]'), 50283);
    expect(tokenizer.tokenId('[OPT]'), 50368);
  });
}
