import 'dart:math' as math;

import '../decisions.dart';
import '../manifest.dart';
import '../reader.dart';
import '../render.dart';
import '../runtime.dart';
import '../tokenizer.dart';

/// Reader for Laya (`laya`, Convai Innovations, Apache-2.0), following `laya.common.build_sequence` and
/// `laya.onnx_agent`: `[CLS] "{type} question: {instructions}" [SEP] [MASK] option_1 ... [MASK] option_n [SEP] state
/// [SEP]`, one logit per `[MASK]` marker, then a temperature per question type and option count.
///
/// Options are capped at `option_max_tokens` and shortened evenly when they exceed `head_max_len`, as Laya does. A
/// state that does not fit in the rest of the context raises [RequestTooLong] (Laya would cut it).
///
/// `reader_config`: `head_max_len`, `option_max_tokens`, `temperature` (one per type: choice, score, noul),
/// `temperature_by_options` (`"choice:3-5"` style buckets), `temperature_range` (applied bounds, default 0.5 to 5),
/// `tokens` {cls, sep, mask}, `noul_labels` {false, true}, `inputs` / `outputs` (tensor names).
class LayaReader implements Reader {
  LayaReader(this.manifest, this.tokenizer) : _cfg = manifest.readerConfig {
    final tokens = (_cfg['tokens'] as Map?)?.cast<String, String>() ?? const {};
    int id(String role, String fallback) {
      final t = tokens[role] ?? fallback;
      return tokenizer.tokenId(t) ?? (throw StateError('special token $t is not in the tokenizer'));
    }

    _mask = tokens['mask'] ?? '[MASK]';
    _cls = id('cls', '[CLS]');
    _sep = id('sep', '[SEP]');
    _maskId = id('mask', '[MASK]');
  }

  static const questionTypes = {'choice': 0, 'score': 1, 'noul': 2};
  static const defaultInputs = {
    'input_ids': 'input_ids',
    'attention_mask': 'attention_mask',
    'marker_pos': 'marker_pos',
    'marker_mask': 'marker_mask',
    'qtype': 'qtype',
  };

  final ModelManifest manifest;
  final Tokenizer tokenizer;
  final Map<String, dynamic> _cfg;
  late final String _mask;
  late final int _cls, _sep, _maskId;

  int get _headMaxLen => _cfg['head_max_len'] as int? ?? 192;
  int get _optionMaxTokens => _cfg['option_max_tokens'] as int? ?? 48;
  String _in(String role) => ((_cfg['inputs'] as Map?)?[role] as String?) ?? defaultInputs[role]!;
  String get _logits => ((_cfg['outputs'] as Map?)?['logits'] as String?) ?? 'logits';

  /// Laya's `render_criterion`: strings as they are, anything else as Python-default JSON.
  static String criterion(Object? v) => v is String ? v : pythonJson(v);

  /// The option texts, in the library's order (noul: false, true).
  List<String> options(Question q) {
    switch (q) {
      case Choice():
        return [
          for (final e in q.options.entries)
            e.value == null || e.value == '' ? e.key : '${e.key}: ${criterion(e.value)}',
        ];
      case Score():
        return [for (var i = 0; i < q.levels.length; i++) 'level $i: ${criterion(q.levels[i])}'];
      case Noul():
        final labels = (_cfg['noul_labels'] as Map?)?.cast<String, String>() ?? const {};
        String text(Object? v, String fallback) => v == null || v == '' ? fallback : criterion(v);
        return [
          '${labels['false'] ?? 'false'}: ${text(q.whenFalse, 'no, the statement does not hold')}',
          '${labels['true'] ?? 'true'}: ${text(q.whenTrue, 'yes, the statement holds')}',
        ];
    }
  }

  @override
  ReaderInput encode(Object? state, Question question) {
    final ins = question.instructions;
    final insText = (ins is String ? ins : (ins == null ? '' : pythonJson(ins))).replaceAll(_mask, ' ');
    var head = tokenizer.encode('${question.type} question: $insText');
    var opts = [
      for (final o in options(question))
        [_maskId, ...tokenizer.encode(' ${o.replaceAll(_mask, ' ')}').take(_optionMaxTokens)],
    ];
    var budget = _headMaxLen - opts.fold<int>(0, (a, o) => a + o.length);
    if (budget < 16) {
      // Too many or too long options: every option is shortened evenly.
      final per = math.max(4, (_headMaxLen - 16) ~/ math.max(1, opts.length));
      opts = [for (final o in opts) o.take(per).toList()];
      budget = _headMaxLen - opts.fold<int>(0, (a, o) => a + o.length);
    }
    head = head.take(math.max(8, budget)).toList();
    final ids = <int>[_cls, ...head, _sep];
    final markers = <int>[];
    for (final o in opts) {
      markers.add(ids.length);
      ids.addAll(o);
    }
    ids.add(_sep);
    final stateText = state is String ? state : pythonJson(state);
    final stateIds = tokenizer.encode(stateText.replaceAll(_mask, ' '));
    final total = ids.length + stateIds.length + 1;
    if (total > manifest.maxLength) throw RequestTooLong(total, manifest.maxLength);
    ids
      ..addAll(stateIds)
      ..add(_sep);
    final k = markers.length;
    return ReaderInput(
      optionCount: k,
      extra: questionTypes[question.type],
      tensors: {
        _in('input_ids'): Tensor.int64(ids, [1, ids.length]),
        _in('attention_mask'): Tensor.int64(List.filled(ids.length, 1), [1, ids.length]),
        _in('marker_pos'): Tensor.int64(markers, [1, k]),
        _in('marker_mask'): Tensor.bool(List.filled(k, true), [1, k]),
        _in('qtype'): Tensor.int64([questionTypes[question.type]!], [1]),
      },
    );
  }

  /// Laya's temperature for a question type and option count, bounded like `laya.common.clamp_temperature`.
  double temperature(int qtype, int k) {
    final range = ((_cfg['temperature_range'] as List?) ?? const [0.5, 5.0]).cast<num>();
    final byType = ((_cfg['temperature'] as List?) ?? const [1.0, 1.0, 1.0]).cast<num>();
    final name = questionTypes.entries.firstWhere((e) => e.value == qtype).key;
    final size = k <= 2 ? '2' : (k <= 5 ? '3-5' : (k <= 10 ? '6-10' : '11+'));
    final t = ((_cfg['temperature_by_options'] as Map?)?['$name:$size'] as num? ?? byType[qtype]).toDouble();
    if (t.isNaN || t.isInfinite) return 1.0;
    return t.clamp(range[0].toDouble(), range[1].toDouble());
  }

  @override
  Readout decode(ReaderInput input, Map<String, Tensor> outputs) {
    final k = input.optionCount;
    final t = temperature(input.extra! as int, k);
    return Readout([for (final z in outputs[_logits]!.doubles.take(k)) z / t]);
  }
}
