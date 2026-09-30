import '../decisions.dart';
import '../manifest.dart';
import '../reader.dart';
import '../render.dart';
import '../runtime.dart';
import '../tokenizer.dart';

/// Reader for encoders that score each option at a marker token (the Dinah protocol, `option-reader`):
/// `[CLS] instructions [SEP] state [SEP] [OPT] option_1 ... [OPT] option_n`, padded to `pad_multiple`, with the
/// question type as an input; outputs one logit per option and a confidence logit.
///
/// `reader_config`: `pad_multiple`, `tokens` {cls, sep, pad, option}, `question_types`, `noul_false_default`,
/// `inputs` / `outputs` (tensor names, when they differ from [defaultInputs] / [defaultOutputs]).
class OptionReader implements Reader {
  OptionReader(this.manifest, this.tokenizer) : _cfg = manifest.readerConfig {
    final tokens = (_cfg['tokens'] as Map?)?.cast<String, String>() ?? const {};
    int id(String role, String fallback) {
      final t = tokens[role] ?? fallback;
      return tokenizer.tokenId(t) ?? (throw StateError('special token $t is not in the tokenizer'));
    }

    _cls = id('cls', '[CLS]');
    _sep = id('sep', '[SEP]');
    _pad = id('pad', '[PAD]');
    _opt = id('option', '[OPT]');
  }

  static const defaultInputs = {
    'ids': 'ids',
    'pad_mask': 'pad_mask',
    'option_positions': 'opt_positions',
    'option_mask': 'opt_mask',
    'question_type': 'qtype',
  };
  static const defaultOutputs = {'logits': 'logits', 'confidence_logit': 'conf_logit'};

  final ModelManifest manifest;
  final Tokenizer tokenizer;
  final Map<String, dynamic> _cfg;
  late final int _cls, _sep, _pad, _opt;

  int get _padMultiple => _cfg['pad_multiple'] as int? ?? 1;
  Map<String, int> get _types =>
      ((_cfg['question_types'] as Map?) ?? const {'noul': 0, 'choice': 1, 'score': 2}).cast<String, int>();
  String _in(String role) => ((_cfg['inputs'] as Map?)?[role] as String?) ?? defaultInputs[role]!;
  String _out(String role) => ((_cfg['outputs'] as Map?)?[role] as String?) ?? defaultOutputs[role]!;

  @override
  ReaderInput encode(Object? state, Question question) {
    final options = optionTexts(
      question,
      noulFalseDefault: _cfg['noul_false_default'] ?? 'The statement above is false.',
    );
    if (options.length < 2) throw ArgumentError('a question needs at least 2 options');
    final type =
        _types[question.type] ??
        (throw UnsupportedError('${manifest.name} does not answer ${question.type} questions'));
    final ids = <int>[
      _cls,
      ...tokenizer.encode(render(question.instructions)),
      _sep,
      ...tokenizer.encode(render(state)),
      _sep,
    ];
    final positions = <int>[];
    for (final o in options) {
      positions.add(ids.length);
      ids
        ..add(_opt)
        ..addAll(tokenizer.encode(render(o)));
    }
    if (ids.length > manifest.maxLength) throw RequestTooLong(ids.length, manifest.maxLength);
    final m = _padMultiple;
    final length = ((ids.length + m - 1) ~/ m) * m;
    final n = positions.length;
    return ReaderInput(
      optionCount: n,
      tensors: {
        _in('ids'): Tensor.int64([...ids, ...List.filled(length - ids.length, _pad)], [1, length]),
        _in('pad_mask'): Tensor.bool(List.generate(length, (i) => i < ids.length), [1, length]),
        _in('option_positions'): Tensor.int64(positions, [1, n]),
        _in('option_mask'): Tensor.bool(List.filled(n, true), [1, n]),
        _in('question_type'): Tensor.int64([type], [1]),
      },
    );
  }

  @override
  Readout decode(ReaderInput input, Map<String, Tensor> outputs) {
    final logits = outputs[_out('logits')]!.doubles.take(input.optionCount).toList();
    final conf = outputs[_out('confidence_logit')];
    return Readout(logits, confidenceLogit: conf?.doubles.first);
  }
}
