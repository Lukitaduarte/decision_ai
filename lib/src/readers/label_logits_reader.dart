import '../decisions.dart';
import '../manifest.dart';
import '../reader.dart';
import '../render.dart';
import '../runtime.dart';
import '../tokenizer.dart';

/// Reader for causal language models (`label-logits`): the question goes in a prompt that lists the options under
/// keys (`A.`, `B.`, ...), and the scores are the next-token logits of those keys after one forward pass. No text is
/// generated.
///
/// `reader_config`:
/// - `template`: the prompt, chat markup included, with `{instructions}`, `{state}` and `{options}` placeholders.
/// - `option_line`: how one option is written, default `{key}. {text}`; lines are joined with `\n`.
/// - `keys`: the option keys, default `A` to `Z`.
/// - `key_prefix`: text before the key when it is scored (`" "` for tokenizers that fold the space into the token).
///   Each `key_prefix + key` must be a single token.
/// - `noul_false_default`: the false option of a noul question without one.
/// - `by_type`: per question type overrides of the settings above (`{"noul": {"keys": ["No", "Yes"], ...}}`).
/// - `bos`: a token to put before the prompt, if the model expects one.
/// - `inputs` / `outputs`: tensor names, when they differ from [defaultInputs] / [defaultOutputs]. Set the
///   `position_ids` input to a name to feed positions. The logits output is either the last position, `[1, vocab]`
///   (what `tool/export_causal_lm.py` exports), or every position, `[1, length, vocab]` (Hugging Face's own ONNX
///   exports, such as the `onnx/` folder many model repositories ship).
/// - `past_key_values`: for graphs with an attention cache input, `{"layers": n, "heads": kv_heads, "head_dim": d}`
///   (from the model's `config.json`); the cache is fed empty. `names` changes the input names, default
///   `past_key_values.{layer}.{kind}` with `kind` `key` and `value`.
class LabelLogitsReader implements Reader {
  LabelLogitsReader(this.manifest, this.tokenizer) : _cfg = manifest.readerConfig;

  static const defaultInputs = {'input_ids': 'input_ids', 'attention_mask': 'attention_mask'};
  static const defaultOutputs = {'logits': 'logits'};
  static const defaultTemplate = '{instructions}\n\nContext:\n{state}\n\nOptions:\n{options}\n\nAnswer:';

  final ModelManifest manifest;
  final Tokenizer tokenizer;
  final Map<String, dynamic> _cfg;
  final Map<String, int> _keyIds = {};

  Object? _setting(String type, String key) => ((_cfg['by_type'] as Map?)?[type] as Map?)?[key] ?? _cfg[key];

  String _in(String role) => ((_cfg['inputs'] as Map?)?[role] as String?) ?? defaultInputs[role]!;
  String? get _positions => (_cfg['inputs'] as Map?)?['position_ids'] as String?;
  String get _logits => ((_cfg['outputs'] as Map?)?['logits'] as String?) ?? defaultOutputs['logits']!;

  /// An empty attention cache, for graphs that take one (the first forward pass of a generation).
  late final Map<String, Tensor> _emptyCache = () {
    final cfg = _cfg['past_key_values'] as Map?;
    if (cfg == null) return const <String, Tensor>{};
    final names = cfg['names'] as String? ?? 'past_key_values.{layer}.{kind}';
    final shape = [1, cfg['heads'] as int, 0, cfg['head_dim'] as int];
    return {
      for (var layer = 0; layer < (cfg['layers'] as int); layer++)
        for (final kind in const ['key', 'value'])
          names.replaceAll('{layer}', '$layer').replaceAll('{kind}', kind): Tensor.float32(const [], shape),
    };
  }();

  int _keyId(String scored) => _keyIds.putIfAbsent(scored, () {
    final ids = tokenizer.encode(scored);
    if (ids.length != 1) throw StateError('option key "$scored" is ${ids.length} tokens; keys must be single tokens');
    return ids.single;
  });

  @override
  ReaderInput encode(Object? state, Question question) {
    final type = question.type;
    final options = optionTexts(
      question,
      noulFalseDefault: _setting(type, 'noul_false_default') ?? 'The statement is false.',
    );
    final keys = ((_setting(type, 'keys') as List?) ?? [for (var c = 65; c <= 90; c++) String.fromCharCode(c)])
        .cast<String>();
    if (options.length > keys.length) throw ArgumentError('${options.length} options, but only ${keys.length} keys');
    final line = _setting(type, 'option_line') as String? ?? '{key}. {text}';
    final listed = [
      for (var i = 0; i < options.length; i++)
        line.replaceAll('{key}', keys[i]).replaceAll('{text}', render(options[i])),
    ].join('\n');
    final prompt = (_setting(type, 'template') as String? ?? defaultTemplate)
        .replaceAll('{instructions}', render(question.instructions))
        .replaceAll('{state}', render(state))
        .replaceAll('{options}', listed);
    final prefix = _setting(type, 'key_prefix') as String? ?? '';
    final keyIds = [for (var i = 0; i < options.length; i++) _keyId('$prefix${keys[i]}')];

    final bos = _cfg['bos'] as String?;
    final bosId = bos == null
        ? null
        : tokenizer.tokenId(bos) ?? (throw StateError('bos token $bos is not in the tokenizer'));
    final ids = [?bosId, ...tokenizer.encode(prompt)];
    if (ids.length > manifest.maxLength) throw RequestTooLong(ids.length, manifest.maxLength);
    final n = ids.length;
    return ReaderInput(
      optionCount: options.length,
      extra: keyIds,
      outputs: [_logits],
      tensors: {
        _in('input_ids'): Tensor.int64(ids, [1, n]),
        _in('attention_mask'): Tensor.int64(List.filled(n, 1), [1, n]),
        ?_positions: Tensor.int64(List.generate(n, (i) => i), [1, n]),
        ..._emptyCache,
      },
    );
  }

  @override
  Readout decode(ReaderInput input, Map<String, Tensor> outputs) {
    final out = outputs[_logits]!;
    final all = out.doubles;
    final vocab = out.shape.last;
    final offset = all.length - vocab; // last position, whichever shape the model returns
    return Readout([for (final id in input.extra as List<int>) all[offset + id]]);
  }
}
