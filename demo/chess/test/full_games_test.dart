import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:decision_ai/decision_ai.dart';
import 'package:dinah_chess/dinah_player.dart';
import 'package:flutter_test/flutter_test.dart';

/// Answers every choice with a random option, so games reach castling, promotion, en passant and mate.
class RandomEngine implements DecisionEngine {
  RandomEngine(this.random);
  final math.Random random;
  var questions = 0;

  @override
  Future<Map<String, Answer>> decide({Object? state, required Map<String, Question> questions}) async {
    this.questions += questions.length;
    return {
      for (final e in questions.entries)
        e.key: () {
          final options = (e.value as Choice).options.keys.toList();
          expect(options.length, inInclusiveRange(2, DinahPlayer.maxOptions));
          final pick = options[random.nextInt(options.length)];
          return Answer.choice(pick, 1.0, {for (final o in options) o: o == pick ? 1.0 : 0.0});
        }(),
    };
  }

  @override
  Future<void> close() async {}
}

void main() {
  test('Dinah\'s player finishes 200 random games without an error', () async {
    final random = math.Random(7);
    final engine = RandomEngine(random);
    final player = DinahPlayer(engine, random: random);
    var plies = 0, mates = 0;
    for (var g = 0; g < 200; g++) {
      final game = ch.Chess();
      while (!game.game_over && game.history.length < 400) {
        final fenBefore = game.fen;
        final turn = await player.choose(game);
        expect(game.fen, fenBefore, reason: 'choosing must not change the position');
        expect(game.generate_moves().map((m) => m.toString()), contains(turn.move.toString()));
        game.make_move(turn.move);
        plies++;
      }
      if (game.in_checkmate) mates++;
    }
    expect(plies, greaterThan(10000));
    expect(mates, greaterThan(0));
  });
}
