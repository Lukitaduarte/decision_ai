import 'dart:convert';
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:dinah_chess/dinah_player.dart';
import 'package:flutter_test/flutter_test.dart';

/// Positions from Dinah-0's chess test set (Lichess puzzles, CC0), written by python-chess in training. The demo must
/// describe them with the same text, or Dinah would be playing a format it never saw.
void main() {
  final rows = (jsonDecode(File('test/dinah_positions.json').readAsStringSync()) as List).cast<Map<String, dynamic>>();

  test('state and option texts match the training data', () {
    var options = 0;
    for (final r in rows) {
      final state = (r['state'] as Map).cast<String, Object>();
      final game = ch.Chess.fromFEN(state['fen']! as String);
      expect(DinahPlayer.stateOf(game), state);
      final described = {for (final m in game.generate_moves()) DinahPlayer.describe(game, m)};
      for (final c in (r['criteria'] as List).cast<String>()) {
        expect(described, contains(c), reason: state['fen'] as String);
        options++;
      }
    }
    expect(options, greaterThan(3000));
  });

  test('descriptions translate to English for display, fact by fact', () {
    expect(DinahPlayer.toEnglish('Nf3: lance comum'), 'Nf3: quiet move');
    expect(
      DinahPlayer.toEnglish('Qxg5: captura bispo, dá xeque, a peça fica pendurada'),
      'Qxg5: captures a bishop, gives check, leaves the piece hanging',
    );
    expect(DinahPlayer.toEnglish('exd6: captura peão (en passant)'), 'exd6: captures a pawn (en passant)');
    // Every fact Dinah can see has an English text.
    for (final r in rows) {
      for (final c in (r['criteria'] as List).cast<String>()) {
        expect(DinahPlayer.toEnglish(c), isNot(matches(RegExp('captura|lance|xeque|pendurada|promove'))), reason: c);
      }
    }
  });
}
