import 'dart:convert';

import 'package:decision_ai/decision_ai.dart';
import 'package:flutter/material.dart';

void main() => runApp(const DemoApp());

class DemoApp extends StatelessWidget {
  const DemoApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'decision_ai',
    theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
    home: const DecisionPage(),
  );
}

/// Loads the bundled model (Dinah-0 in this example) and answers one typed decision about the text you type.
class DecisionPage extends StatefulWidget {
  const DecisionPage({super.key});

  @override
  State<DecisionPage> createState() => _DecisionPageState();
}

class _DecisionPageState extends State<DecisionPage> {
  final _state = TextEditingController(text: 'The package arrived broken and I need it for tomorrow.');
  DecisionEngine? _engine;
  String _status = 'Loading model…';
  String _result = '';

  static const _question = Choice({
    'refund': 'The customer wants their money back',
    'replacement': 'The customer wants a new unit sent',
    'complaint': 'The customer only wants to complain',
  }, instructions: 'What does the customer want?');

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final sw = Stopwatch()..start();
    final engine = await DecisionAI.local();
    setState(() {
      _engine = engine;
      _status = 'Model loaded in ${sw.elapsedMilliseconds} ms';
    });
  }

  Future<void> _decide() async {
    final engine = _engine;
    if (engine == null) return;
    final sw = Stopwatch()..start();
    final answers = await engine.decide(state: _state.text, questions: {'intent': _question});
    setState(() {
      _result =
          '${const JsonEncoder.withIndent('  ').convert(answers['intent']!.toJson())}\n\n${sw.elapsedMilliseconds} ms on device';
    });
  }

  @override
  void dispose() {
    _engine?.close();
    _state.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('decision_ai')),
    body: ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Text(_status, style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 12),
        TextField(
          controller: _state,
          maxLines: 4,
          decoration: const InputDecoration(labelText: 'Customer message', border: OutlineInputBorder()),
        ),
        const SizedBox(height: 12),
        const Text('What does the customer want? refund · replacement · complaint'),
        const SizedBox(height: 12),
        FilledButton(onPressed: _engine == null ? null : _decide, child: const Text('Decide')),
        const SizedBox(height: 16),
        SelectableText(_result, style: const TextStyle(fontFamily: 'Menlo')),
      ],
    ),
  );
}
