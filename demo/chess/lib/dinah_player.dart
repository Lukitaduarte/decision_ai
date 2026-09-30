import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:decision_ai/decision_ai.dart';

/// Dinah-0 learned chess from Lichess puzzles with one question: "Qual é o melhor lance para o lado que joga?"
/// (what is the best move for the side to play?), a state with the position, and at most 12 candidate moves, each
/// written as SAN plus facts a fair harness can compute (captures, checks, promotions, hanging pieces; never mate).
/// This player asks exactly that question, so Dinah plays in the format it was trained and measured on.
class DinahPlayer {
  DinahPlayer(this.engine, {math.Random? random}) : _random = random ?? math.Random();

  /// Dinah saw at most 12 options per question.
  static const maxOptions = 12;
  static const instructions = 'Qual é o melhor lance para o lado que joga?';

  final DecisionEngine engine;
  final math.Random _random;

  /// Picks a move. With more than [maxOptions] legal moves, candidates play in rounds of up to 12 and the winners meet
  /// in a final question, so every question stays within what Dinah was trained on.
  Future<DinahTurn> choose(ch.Chess game) async {
    final legal = game.generate_moves();
    if (legal.length == 1) return DinahTurn(legal.single, {describe(game, legal.single): 1.0}, 0);
    final state = stateOf(game);
    var pool = [...legal]..shuffle(_random);
    var questions = 0;
    while (pool.length > maxOptions) {
      final groups = _split(pool);
      final answers = await engine.decide(
        state: state,
        questions: {for (var i = 0; i < groups.length; i++) 'group$i': _question(game, groups[i])},
      );
      questions += groups.length;
      pool = [for (var i = 0; i < groups.length; i++) groups[i][_index(answers['group$i']!.choice!)]];
    }
    final answer = (await engine.decide(state: state, questions: {'final': _question(game, pool)}))['final']!;
    final probabilities = {
      for (var i = 0; i < pool.length; i++) describe(game, pool[i]): answer.probabilities['m$i'] ?? 0,
    };
    return DinahTurn(pool[_index(answer.choice!)], probabilities, questions + 1);
  }

  static int _index(String label) => int.parse(label.substring(1));

  Choice _question(ch.Chess game, List<ch.Move> moves) =>
      Choice({for (var i = 0; i < moves.length; i++) 'm$i': describe(game, moves[i])}, instructions: instructions);

  /// Splits [moves] into the fewest groups of at most [maxOptions], with sizes as even as possible.
  static List<List<ch.Move>> _split(List<ch.Move> moves) {
    final n = (moves.length / maxOptions).ceil();
    return [for (var i = 0; i < n; i++) moves.sublist(i * moves.length ~/ n, (i + 1) * moves.length ~/ n)];
  }

  static const _names = {'p': 'peão', 'n': 'cavalo', 'b': 'bispo', 'r': 'torre', 'q': 'dama', 'k': 'rei'};
  static const _values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0};

  /// A move as Dinah saw it in training: `"Nxe5: captura peão, dá xeque"`, or `"a4: lance comum"`.
  static String describe(ch.Chess game, ch.Move move) {
    final san = game.move_to_san(move);
    final parts = <String>[];
    if (move.flags & ch.Chess.BITS_EP_CAPTURE != 0) {
      parts.add('captura peão (en passant)');
    } else if (move.captured != null) {
      parts.add('captura ${_names[move.captured!.name]}');
    }
    if (move.promotion != null) parts.add('promove a ${_names[move.promotion!.name]}');
    game.make_move(move);
    try {
      if (game.in_check) parts.add('dá xeque');
      // Hanging: the opponent attacks the destination and the mover no longer defends it.
      if (game.attacked(game.turn, move.to) && !game.attacked(ch.Chess.swap_color(game.turn), move.to)) {
        parts.add('a peça fica pendurada');
      }
    } finally {
      game.undo_move();
    }
    return '$san: ${parts.isEmpty ? 'lance comum' : parts.join(', ')}';
  }

  static const _english = {
    'captura peão (en passant)': 'captures a pawn (en passant)',
    'captura peão': 'captures a pawn',
    'captura cavalo': 'captures a knight',
    'captura bispo': 'captures a bishop',
    'captura torre': 'captures a rook',
    'captura dama': 'captures the queen',
    'promove a cavalo': 'promotes to a knight',
    'promove a bispo': 'promotes to a bishop',
    'promove a torre': 'promotes to a rook',
    'promove a dama': 'promotes to a queen',
    'dá xeque': 'gives check',
    'a peça fica pendurada': 'leaves the piece hanging',
    'lance comum': 'quiet move',
  };

  /// A [describe] text in English, for display. Dinah itself reads the Portuguese it was trained on.
  static String toEnglish(String description) {
    final colon = description.indexOf(': ');
    if (colon < 0) return description;
    final facts = description.substring(colon + 2).split(', ').map((f) => _english[f] ?? f).join(', ');
    return '${description.substring(0, colon)}: $facts';
  }

  /// The position as Dinah saw it: FEN (python-chess style), side to move, material balance, check.
  static Map<String, Object> stateOf(ch.Chess game) {
    var material = 0;
    for (var i = 0; i < 128; i++) {
      if (i & 0x88 != 0) continue;
      final p = game.board[i];
      if (p != null) material += _values[p.type.name]! * (p.color == ch.Color.WHITE ? 1 : -1);
    }
    return {
      'fen': fen(game),
      'vez': game.turn == ch.Color.WHITE ? 'brancas' : 'pretas',
      'material_brancas_menos_pretas': material,
      'em_xeque': game.in_check,
    };
  }

  /// python-chess writes the en passant square only when an en passant capture is legal; chess.dart always does.
  static String fen(ch.Chess game) {
    final parts = game.fen.split(' ');
    if (parts[3] != '-' && !game.generate_moves().any((m) => m.flags & ch.Chess.BITS_EP_CAPTURE != 0)) {
      parts[3] = '-';
    }
    return parts.join(' ');
  }
}

/// Dinah's move, the probabilities of the final question (by move description), and how many questions it took.
class DinahTurn {
  DinahTurn(this.move, this.probabilities, this.questions);
  final ch.Move move;
  final Map<String, double> probabilities;
  final int questions;
}
