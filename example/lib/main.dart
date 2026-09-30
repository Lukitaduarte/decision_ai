import 'package:decision_ai/decision_ai.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import 'gliclass_reader.dart';

void main() => runApp(const DemoApp());

class DemoApp extends StatelessWidget {
  const DemoApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'decision_ai',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(colorSchemeSeed: const Color(0xFF7C5CFF), useMaterial3: true, fontFamily: 'Roboto'),
    darkTheme: ThemeData(
      colorSchemeSeed: const Color(0xFF7C5CFF),
      brightness: Brightness.dark,
      useMaterial3: true,
      fontFamily: 'Roboto',
    ),
    home: const DecisionPage(),
  );
}

/// A model the app can load, and how.
class ModelOption {
  const ModelOption(this.name, this.detail, this.load, {this.needsKey = false});
  final String name;
  final String detail;
  final bool needsKey;

  /// [key] and [modelId] are only used by API models.
  final Future<DecisionEngine> Function(DownloadProgress onProgress, String key, String modelId) load;
}

/// SmolLM2's chat format, ending where the answer (an option letter) starts.
const _smolTemplate =
    '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n'
    '<|im_start|>user\n{instructions}\n\nContext:\n{state}\n\nOptions:\n{options}\n\n'
    'Answer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n';

/// Verdict's calibrator.json (heman10x/rlcd-modernbert-151m): a temperature per label count.
final _verdictCalibrator = verdictCalibrator({
  'temperature': 2.8039,
  'per_k': {
    '2': 5.0069,
    '3': 5.0069,
    '4': 4.0314,
    '5': 3.0560,
    '6': 2.3898,
    '7': 2.3898,
    '9': 1.6668,
    '11': 3.3919,
    '17': 1.7200,
    '25': 1.5144,
  },
});

/// On the web the bundled model is not shipped (GitHub Pages takes files up to 100 MB): models come from Hugging Face.
final models = <ModelOption>[
  if (!kIsWeb)
    ModelOption(
      'Dinah-0, in the app',
      'Encoder, 150M. Bundled (165 MB), works offline.',
      (_, _, _) => DecisionAI.local(),
    ),
  ModelOption(
    'Dinah-0, from Hugging Face',
    'The same model, downloaded once (165 MB) and cached.',
    (onProgress, _, _) => DecisionAI.huggingFace(
      'Lukitaduarte/dinah-0',
      revision: '07c6884439df7c3d2cd01eea16924ed07b866dea',
      onProgress: onProgress,
    ),
  ),
  ModelOption(
    'SmolLM2-135M-Instruct',
    'A small LLM, straight from its own repository (onnx/model_q4.onnx, 182 MB): no export.',
    (onProgress, _, _) => DecisionAI.huggingFace(
      'HuggingFaceTB/SmolLM2-135M-Instruct',
      revision: '12fd25f77366fa6b3b4b768ec3050bf629380bac',
      onProgress: onProgress,
      manifest: ModelManifest.custom(
        reader: 'label-logits',
        files: {'model': 'onnx/model_q4.onnx', 'tokenizer': 'tokenizer.json'},
        maxLength: 2048,
        readerConfig: {
          'template': _smolTemplate,
          'inputs': {'position_ids': 'position_ids'},
          'past_key_values': {'layers': 30, 'heads': 3, 'head_dim': 64},
        },
      ),
    ),
  ),
  ModelOption(
    'Laya',
    'ModernBERT-large encoder, 421M, a community 4-bit port (techtheist/laya-onnx, 275 MB). Matches '
        'Laya exactly on iOS; its 4-bit kernels change 2 of 64 answers on Android.',
    (onProgress, _, _) => DecisionAI.huggingFace(
      'techtheist/laya-onnx',
      revision: '6ca7822091b315d260f5cb7bfa82f6acc94bbffc',
      onProgress: onProgress,
      manifest: ModelManifest.custom(
        reader: 'laya',
        files: {'model': 'en/model_int4.onnx', 'tokenizer': 'en/tokenizer.json'},
        maxLength: 512,
        // From en/rl_agent_config.json.
        readerConfig: {
          'head_max_len': 192,
          'temperature': [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
          'temperature_by_options': {
            'choice:3-5': 1.7601518630981445,
            'choice:6-10': 1.0000158548355103,
            'score:3-5': 1.2514300346374512,
            'noul:2': 1.983399510383606,
            'choice:11+': 0.10058280825614929,
            'choice:2': 1.9063563346862793,
          },
        },
      ),
    ),
  ),
  ModelOption(
    'Verdict',
    'GLiClass encoder, 151M, the authors\' fp32 model (606 MB), read by a reader this app registers.',
    (onProgress, _, _) {
      DecisionAI.registerReader('gliclass', GliClassReader.new);
      return DecisionAI.huggingFace(
        'heman10x/rlcd-modernbert-151m',
        revision: '8af2496eb63c7fa66d7d234e1f62629380030eb4',
        onProgress: onProgress,
        calibrator: _verdictCalibrator,
        manifest: ModelManifest.custom(
          reader: 'gliclass',
          files: {'model': 'model.onnx', 'tokenizer': 'tokenizer.json'},
          maxLength: 512,
        ),
      );
    },
  ),
  ModelOption(
    'OpenRouter',
    'A large model behind OpenRouter\'s Decisions API. Needs your OpenRouter key.',
    (_, key, modelId) async => DecisionAI.openRouter(apiKey: key, model: modelId, appName: 'decision_ai example'),
    needsKey: true,
  ),
];

/// A ready-made question to start from.
class Preset {
  const Preset(this.name, this.state, this.question);
  final String name;
  final String state;
  final Question question;
}

const presets = [
  Preset(
    'Customer intent',
    'The package arrived broken and I need it for tomorrow.',
    Choice({
      'refund': 'The customer wants their money back',
      'replacement': 'The customer wants a new unit sent',
      'complaint': 'The customer only wants to complain',
    }, instructions: 'What does the customer want?'),
  ),
  Preset(
    'Cancel?',
    'Hi, please cancel my subscription starting today.',
    Noul(
      'The customer wants to cancel.',
      whenTrue: 'The customer wants to cancel.',
      whenFalse: 'The customer does not want to cancel.',
    ),
  ),
  Preset(
    'Review',
    'Great product, but shipping took three weeks.',
    Score(['very negative', 'negative', 'neutral', 'positive', 'very positive'], instructions: 'Rate the review.'),
  ),
];

enum Kind { choice, noul, score }

class DecisionPage extends StatefulWidget {
  const DecisionPage({super.key});

  @override
  State<DecisionPage> createState() => _DecisionPageState();
}

class _DecisionPageState extends State<DecisionPage> {
  var _model = models.first;
  DecisionEngine? _engine;
  String? _loadedName;
  var _loading = false;
  String _status = 'Pick a model and load it.';
  double? _progress;

  final _key = TextEditingController();
  final _modelId = TextEditingController();
  final _state = TextEditingController();
  final _instructions = TextEditingController();
  final _options = TextEditingController();
  final _whenTrue = TextEditingController();
  final _whenFalse = TextEditingController();
  var _kind = Kind.choice;

  var _asking = false;
  Answer? _answer;
  List<String> _labels = const [];
  int? _answerMs;
  Object? _error;

  @override
  void initState() {
    super.initState();
    _apply(presets.first);
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _progress = null;
      _error = null;
      _status = 'Loading ${_model.name}...';
    });
    await _engine?.close();
    _engine = null;
    final sw = Stopwatch()..start();
    try {
      final engine = await _model.load(
        (file, received, total) {
          if (!mounted) return;
          setState(() {
            _progress = total == null ? null : received / total;
            _status = 'Downloading $file (${(received / 1e6).toStringAsFixed(0)} MB)';
          });
        },
        _key.text.trim(),
        _modelId.text.trim(),
      );
      setState(() {
        _engine = engine;
        _loadedName = _model.name;
        _status = '${_model.name} ready in ${sw.elapsedMilliseconds} ms.';
      });
    } catch (e) {
      setState(() {
        _error = e;
        _status = 'Could not load ${_model.name}.';
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _apply(Preset p) {
    _state.text = p.state;
    _answer = null;
    switch (p.question) {
      case Choice(:final options, :final instructions):
        _kind = Kind.choice;
        _instructions.text = '${instructions ?? ''}';
        _options.text = [for (final e in options.entries) '${e.key}: ${e.value ?? ''}'].join('\n');
      case Noul(:final instructions, :final whenTrue, :final whenFalse):
        _kind = Kind.noul;
        _instructions.text = '${instructions ?? ''}';
        _whenTrue.text = '${whenTrue ?? ''}';
        _whenFalse.text = '${whenFalse ?? ''}';
      case Score(:final levels, :final instructions):
        _kind = Kind.score;
        _instructions.text = '${instructions ?? ''}';
        _options.text = levels.join('\n');
    }
    setState(() {});
  }

  /// The question on screen. Choice lines are `label: description` (or just a label); score lines are levels.
  Question _question() {
    final lines = _options.text.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();
    switch (_kind) {
      case Kind.choice:
        return Choice({
          for (final l in lines)
            if (l.contains(':'))
              l.substring(0, l.indexOf(':')).trim(): l.substring(l.indexOf(':') + 1).trim()
            else
              l: null,
        }, instructions: _instructions.text.trim());
      case Kind.noul:
        return Noul(
          _instructions.text.trim(),
          whenTrue: _whenTrue.text.trim().isEmpty ? null : _whenTrue.text.trim(),
          whenFalse: _whenFalse.text.trim().isEmpty ? null : _whenFalse.text.trim(),
        );
      case Kind.score:
        return Score(lines, instructions: _instructions.text.trim());
    }
  }

  Future<void> _ask() async {
    final engine = _engine;
    if (engine == null) return;
    final q = _question();
    setState(() {
      _asking = true;
      _error = null;
    });
    final sw = Stopwatch()..start();
    try {
      final answer = (await engine.decide(state: _state.text.trim(), questions: {'q': q}))['q']!;
      setState(() {
        _answer = answer;
        _answerMs = sw.elapsedMilliseconds;
        _labels = switch (q) {
          Choice(:final options) => options.keys.toList(),
          Noul() => const ['false', 'true'],
          Score(:final levels) => [for (final l in levels) '$l'],
        };
      });
    } catch (e) {
      setState(() => _error = e);
    } finally {
      if (mounted) setState(() => _asking = false);
    }
  }

  @override
  void dispose() {
    _engine?.close();
    for (final c in [_key, _modelId, _state, _instructions, _options, _whenTrue, _whenFalse]) {
      c.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('decision_ai')),
      // Tapping outside a field, or dragging the list, closes the keyboard.
      body: GestureDetector(
        behavior: HitTestBehavior.translucent,
        onTap: () => FocusManager.instance.primaryFocus?.unfocus(),
        child: SafeArea(
          // A phone-width column, also on a desktop browser.
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 720),
              child: ListView(
                keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
                padding: const EdgeInsets.all(16),
                children: [
                  Text('1. Model', style: text.titleMedium),
                  const SizedBox(height: 8),
                  DropdownButtonFormField<ModelOption>(
                    initialValue: _model,
                    isExpanded: true,
                    items: [for (final m in models) DropdownMenuItem(value: m, child: Text(m.name))],
                    onChanged: _loading ? null : (m) => setState(() => _model = m!),
                  ),
                  const SizedBox(height: 4),
                  Text(_model.detail, style: text.bodySmall),
                  if (_model.needsKey) ...[
                    const SizedBox(height: 8),
                    TextField(
                      controller: _key,
                      obscureText: true,
                      decoration: const InputDecoration(labelText: 'OpenRouter API key', border: OutlineInputBorder()),
                    ),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _modelId,
                      decoration: const InputDecoration(
                        labelText: 'Model id',
                        hintText: 'provider/model, as OpenRouter lists it',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ],
                  const SizedBox(height: 8),
                  FilledButton.icon(
                    onPressed: _loading ? null : _load,
                    icon: const Icon(Icons.download),
                    label: Text(_loadedName == _model.name ? 'Reload' : 'Load'),
                  ),
                  const SizedBox(height: 8),
                  if (_loading) LinearProgressIndicator(value: _progress),
                  Text(_status, style: text.bodySmall),
                  const Divider(height: 32),
                  Text('2. Question', style: text.titleMedium),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    children: [for (final p in presets) ActionChip(label: Text(p.name), onPressed: () => _apply(p))],
                  ),
                  const SizedBox(height: 8),
                  TextField(
                    controller: _state,
                    minLines: 2,
                    maxLines: 6,
                    decoration: const InputDecoration(labelText: 'State (the data)', border: OutlineInputBorder()),
                  ),
                  const SizedBox(height: 12),
                  SegmentedButton<Kind>(
                    segments: const [
                      ButtonSegment(value: Kind.choice, label: Text('Choice')),
                      ButtonSegment(value: Kind.noul, label: Text('Noul')),
                      ButtonSegment(value: Kind.score, label: Text('Score')),
                    ],
                    selected: {_kind},
                    onSelectionChanged: (s) => setState(() {
                      _kind = s.first;
                      _answer = null;
                    }),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _instructions,
                    decoration: InputDecoration(
                      labelText: _kind == Kind.noul ? 'Statement' : 'Question',
                      border: const OutlineInputBorder(),
                    ),
                  ),
                  const SizedBox(height: 12),
                  if (_kind == Kind.noul) ...[
                    TextField(
                      controller: _whenTrue,
                      decoration: const InputDecoration(
                        labelText: 'What "true" looks like (optional)',
                        border: OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _whenFalse,
                      decoration: const InputDecoration(
                        labelText: 'What "false" looks like (optional)',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ] else
                    TextField(
                      controller: _options,
                      minLines: 3,
                      maxLines: 10,
                      decoration: InputDecoration(
                        labelText: _kind == Kind.choice
                            ? 'Options, one per line: label: description'
                            : 'Levels, one per line, lowest first',
                        border: const OutlineInputBorder(),
                      ),
                    ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: _engine == null || _asking ? null : _ask,
                    icon: _asking
                        ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                        : const Icon(Icons.send),
                    label: Text(_engine == null ? 'Load a model first' : 'Ask ${_loadedName ?? ''}'),
                  ),
                  if (_error != null) ...[
                    const SizedBox(height: 12),
                    Text('$_error', style: text.bodySmall?.copyWith(color: Theme.of(context).colorScheme.error)),
                  ],
                  if (_answer != null) ...[const Divider(height: 32), _AnswerView(_answer!, _labels, _answerMs!)],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _AnswerView extends StatelessWidget {
  const _AnswerView(this.answer, this.labels, this.ms);
  final Answer answer;
  final List<String> labels;
  final int ms;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final Map<String, double> bars = switch (answer.type) {
      'noul' => {'false': 1 - answer.noul!, 'true': answer.noul!},
      'score' => {for (final e in answer.probabilities.entries) labels[int.parse(e.key)]: e.value},
      _ => answer.probabilities,
    };
    final headline = switch (answer.type) {
      'noul' => 'True with probability ${(answer.noul! * 100).toStringAsFixed(1)}%',
      'score' => 'Score ${answer.score!.toStringAsFixed(2)} (0 to ${labels.length - 1})',
      _ => answer.choice!,
    };
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('3. Answer', style: text.titleMedium),
        const SizedBox(height: 8),
        Text(headline, style: text.headlineSmall),
        const SizedBox(height: 4),
        Text(
          [
            if (answer.confidence != null) 'confidence ${(answer.confidence! * 100).toStringAsFixed(1)}%',
            if (answer.abstention != null) 'insufficient evidence ${(answer.abstention! * 100).toStringAsFixed(1)}%',
            '$ms ms on this device',
          ].join(' · '),
          style: text.bodySmall,
        ),
        const SizedBox(height: 12),
        for (final e in bars.entries)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(child: Text(e.key, overflow: TextOverflow.ellipsis)),
                    Text('${(e.value * 100).toStringAsFixed(1)}%'),
                  ],
                ),
                const SizedBox(height: 2),
                LinearProgressIndicator(value: e.value, minHeight: 6, borderRadius: BorderRadius.circular(3)),
              ],
            ),
          ),
      ],
    );
  }
}
