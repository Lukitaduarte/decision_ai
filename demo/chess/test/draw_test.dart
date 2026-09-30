import 'package:chess/chess.dart' as ch;
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('a lone king with no legal move and no check is stalemate (a game played against Dinah, after 34. Be3)', () {
    final game = ch.Chess.fromFEN('8/8/8/2K5/4PQ2/4B3/PPP1k2P/R4N2 b - - 0 34');
    expect(game.in_check, isFalse);
    expect(game.generate_moves(), isEmpty);
    expect(game.in_stalemate, isTrue);
    expect(game.in_draw, isTrue);
  });
}
