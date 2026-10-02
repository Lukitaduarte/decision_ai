@Tags(['live'])
library;

import 'dart:convert';
import 'dart:io';

import 'package:decision_ai/decision_ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('OpenRouter decisions, for real', () async {
    final key = Platform.environment['OPENROUTER_API_KEY'];
    if (key == null || key.isEmpty) return markTestSkipped('no OPENROUTER_API_KEY');
    final ai = DecisionAI.openRouter(
      apiKey: key,
      model: 'typesafe/jev-1.13',
      appName: 'decision_ai live test',
      extra: {
        'provider': {'zdr': true, 'data_collection': 'deny'},
      },
    );
    final answers = await ai.decide(
      state: 'The package arrived broken and I need it for tomorrow.',
      questions: {
        'intent': const Choice({
          'refund': 'The customer wants their money back',
          'replacement': 'The customer wants a new unit sent',
          'complaint': 'The customer only wants to complain',
        }, instructions: 'What does the customer want?'),
        'urgent': const Noul('The customer needs it soon.'),
        'mood': const Score(['angry', 'upset', 'neutral', 'happy'], instructions: 'How does the customer feel?'),
      },
    );
    // ignore: avoid_print
    print('LIVE ${jsonEncode(answers.map((k, v) => MapEntry(k, v.toJson())))}');
    // ignore: avoid_print
    print('USAGE ${jsonEncode((ai as ApiEngine).lastUsage)}');
    expect(answers['intent']!.choice, isNotNull);
    expect(answers['urgent']!.noul, isNotNull);
    expect(answers['mood']!.score, isNotNull);
    await ai.close();
  });
}
