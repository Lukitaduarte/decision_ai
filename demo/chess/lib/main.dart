import 'package:chess/chess.dart' as ch;
import 'package:decision_ai/decision_ai.dart';
import 'package:flutter/material.dart';

import 'dinah_player.dart';

/// Dinah-0 at the commit this demo was checked against.
const dinahRepo = 'Lukitaduarte/dinah-0';
const dinahRevision = '07c6884439df7c3d2cd01eea16924ed07b866dea';

void main() => runApp(const DinahChessApp());

class DinahChessApp extends StatelessWidget {
  const DinahChessApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Play chess against Dinah',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      colorSchemeSeed: const Color(0xFF7C5CFF),
      brightness: Brightness.dark,
      useMaterial3: true,
      fontFamily: 'Roboto',
    ),
    home: const GamePage(),
  );
}

class GamePage extends StatefulWidget {
  const GamePage({super.key});

  @override
  State<GamePage> createState() => _GamePageState();
}

class _GamePageState extends State<GamePage> {
  DinahPlayer? _dinah;
  Object? _loadError;
  var _game = ch.Chess();
  var _humanIsWhite = true;
  String? _selected;
  (String, String)? _lastMove;
  var _thinking = false;
  DinahTurn? _lastTurn;
  Duration? _lastThinkTime;
  final _history = <String>[];

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      // The model runs in the browser (ONNX Runtime Web); nothing is sent to a server.
      final engine = await DecisionAI.huggingFace(dinahRepo, revision: dinahRevision);
      setState(() => _dinah = DinahPlayer(engine));
      _maybeDinahMoves();
    } catch (e) {
      setState(() => _loadError = e);
    }
  }

  bool get _humansTurn => (_game.turn == ch.Color.WHITE) == _humanIsWhite;

  void _newGame({required bool humanIsWhite}) {
    if (_thinking) return;
    setState(() {
      _game = ch.Chess();
      _humanIsWhite = humanIsWhite;
      _selected = null;
      _lastMove = null;
      _lastTurn = null;
      _history.clear();
    });
    _maybeDinahMoves();
  }

  void _play(ch.Move m) {
    final san = _game.move_to_san(m);
    _game.make_move(m);
    _history.add(san);
    _lastMove = (ch.Chess.algebraic(m.from), ch.Chess.algebraic(m.to));
  }

  Future<void> _maybeDinahMoves() async {
    final dinah = _dinah;
    if (dinah == null || _game.game_over || _humansTurn || _thinking) return;
    setState(() => _thinking = true);
    final sw = Stopwatch()..start();
    try {
      final turn = await dinah.choose(_game);
      if (!mounted) return;
      setState(() {
        _play(turn.move);
        _lastTurn = turn;
        _lastThinkTime = sw.elapsed;
      });
    } catch (e) {
      setState(() => _loadError = e);
    } finally {
      if (mounted) setState(() => _thinking = false);
    }
  }

  void _tap(String square) {
    if (_dinah == null || !_humansTurn || _thinking || _game.game_over) return;
    final piece = _game.get(square);
    if (_selected != null) {
      final moves = _game.generate_moves().where(
        (m) => ch.Chess.algebraic(m.from) == _selected && ch.Chess.algebraic(m.to) == square,
      );
      if (moves.isNotEmpty) {
        // Promotions go to a queen.
        final m = moves.firstWhere((m) => m.promotion == null || m.promotion == ch.PieceType.QUEEN);
        setState(() {
          _play(m);
          _selected = null;
        });
        _maybeDinahMoves();
        return;
      }
    }
    setState(() => _selected = piece != null && piece.color == _game.turn ? square : null);
  }

  String get _status {
    if (_loadError != null) return 'Something went wrong: $_loadError';
    if (_dinah == null) return 'Downloading Dinah-0 from Hugging Face (165 MB, cached by the browser)...';
    if (_game.in_checkmate) return _humansTurn ? 'Checkmate. Dinah wins.' : 'Checkmate. You win!';
    if (_game.in_draw) return _drawReason;
    if (_thinking) return 'Dinah is thinking...';
    if (_game.in_check) return _humansTurn ? 'Check! Your king is attacked.' : 'Check! Dinah must answer.';
    return _humansTurn ? 'Your move.' : "Dinah's move.";
  }

  /// Why the game is drawn, with the official rule behind it (FIDE Laws of Chess, articles 5.2 and 9).
  String get _drawReason {
    if (_game.in_stalemate) {
      return 'Draw by stalemate, an official rule of chess: ${_humansTurn ? 'you have' : 'Dinah has'} no legal move '
          'and ${_humansTurn ? 'are' : 'is'} not in check, so the game ends in a draw, whatever the material.';
    }
    if (_game.insufficient_material) {
      return 'Draw by insufficient material, an official rule of chess: neither side has enough pieces left to mate.';
    }
    if (_game.in_threefold_repetition) {
      return 'Draw by threefold repetition, an official rule of chess: the same position appeared three times.';
    }
    return 'Draw by the 50-move rule, an official rule of chess: 50 moves each without a capture or a pawn move.';
  }

  @override
  Widget build(BuildContext context) {
    final wide = MediaQuery.sizeOf(context).width > 900;
    final board = ConstrainedBox(
      constraints: const BoxConstraints(maxWidth: 560),
      child: AspectRatio(aspectRatio: 1, child: _board()),
    );
    final panel = ConstrainedBox(
      constraints: BoxConstraints(maxWidth: wide ? 360 : 560),
      child: _panel(context),
    );
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Center(
            child: Column(
              children: [
                Text(
                  'Play chess against Dinah',
                  style: Theme.of(context).textTheme.headlineMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 4),
                Text(
                  'A 150M-parameter decision model running in your browser with decision_ai',
                  style: Theme.of(context).textTheme.bodyMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                if (wide)
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Flexible(child: board),
                      const SizedBox(width: 24),
                      panel,
                    ],
                  )
                else
                  Column(children: [board, const SizedBox(height: 16), panel]),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _board() {
    final targets = _selected == null
        ? const <String>{}
        : {
            for (final m in _game.generate_moves())
              if (ch.Chess.algebraic(m.from) == _selected) ch.Chess.algebraic(m.to),
          };
    return GridView.builder(
      physics: const NeverScrollableScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 8),
      itemCount: 64,
      itemBuilder: (context, i) {
        final row = i ~/ 8, col = i % 8;
        final file = _humanIsWhite ? col : 7 - col;
        final rank = _humanIsWhite ? 7 - row : row;
        final square = '${'abcdefgh'[file]}${rank + 1}';
        final light = (file + rank) % 2 == 1;
        final piece = _game.get(square);
        final last = _lastMove;
        final highlighted = square == _selected || (last != null && (last.$1 == square || last.$2 == square));
        // The king of the side to move, when it is in check.
        final checked = _game.in_check && piece != null && piece.type == ch.PieceType.KING && piece.color == _game.turn;
        return GestureDetector(
          onTap: () => _tap(square),
          child: Container(
            color: checked
                ? const Color(0xFFE5484D)
                : highlighted
                ? (light ? const Color(0xFFCDD26A) : const Color(0xFFAAA23A))
                : (light ? const Color(0xFFEEEED2) : const Color(0xFF769656)),
            child: Stack(
              alignment: Alignment.center,
              children: [
                if (piece != null) _pieceGlyph(piece),
                if (targets.contains(square))
                  FractionallySizedBox(
                    widthFactor: 0.3,
                    heightFactor: 0.3,
                    child: Container(
                      decoration: const BoxDecoration(color: Color(0x55000000), shape: BoxShape.circle),
                    ),
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  static const _glyphs = {'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'};

  Widget _pieceGlyph(ch.Piece piece) {
    final white = piece.color == ch.Color.WHITE;
    return FittedBox(
      child: Padding(
        padding: const EdgeInsets.all(4),
        child: Text(
          _glyphs[piece.type.name]!,
          style: TextStyle(
            fontFamily: 'NotoSansSymbols2',
            fontSize: 48,
            color: white ? Colors.white : Colors.black,
            shadows: [Shadow(color: white ? Colors.black : Colors.white24, blurRadius: 2)],
          ),
        ),
      ),
    );
  }

  Widget _panel(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final turn = _lastTurn;
    final top = turn == null
        ? const <MapEntry<String, double>>[]
        : (turn.probabilities.entries.toList()..sort((a, b) => b.value.compareTo(a.value))).take(5).toList();
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                ClipOval(child: Image.asset('assets/images/dinah_face.png', width: 56, height: 56, fit: BoxFit.cover)),
                const SizedBox(width: 12),
                Text('Dinah', style: text.titleLarge),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                if ((_dinah == null && _loadError == null) || _thinking)
                  const Padding(
                    padding: EdgeInsets.only(right: 12),
                    child: SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)),
                  ),
                Expanded(child: Text(_status, style: text.titleMedium)),
              ],
            ),
            if (turn != null) ...[
              const SizedBox(height: 16),
              Text(
                "Dinah's last choice (${turn.questions} question${turn.questions == 1 ? '' : 's'}, "
                '${_lastThinkTime!.inMilliseconds} ms)',
                style: text.labelLarge,
              ),
              const SizedBox(height: 8),
              for (final e in top)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          DinahPlayer.toEnglish(e.key),
                          style: text.bodySmall,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      SizedBox(
                        width: 56,
                        child: Text('${(e.value * 100).toStringAsFixed(1)}%', textAlign: TextAlign.end),
                      ),
                    ],
                  ),
                ),
            ],
            const SizedBox(height: 16),
            Text('Moves', style: text.labelLarge),
            const SizedBox(height: 4),
            Text(
              _history.isEmpty
                  ? '-'
                  : [
                      for (var i = 0; i < _history.length; i += 2)
                        '${i ~/ 2 + 1}. ${_history[i]}${i + 1 < _history.length ? ' ${_history[i + 1]}' : ''}',
                    ].join('  '),
              style: text.bodySmall,
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton(onPressed: () => _newGame(humanIsWhite: true), child: const Text('New game as white')),
                OutlinedButton(onPressed: () => _newGame(humanIsWhite: false), child: const Text('New game as black')),
              ],
            ),
            const SizedBox(height: 16),
            Text(
              'Dinah learned chess from Lichess puzzles, not full games. It picks among up to 12 candidate moves at '
              'a time, described only by facts a fair harness can compute (captures, checks, hanging pieces), and '
              'solves about 6 in 10 puzzles of its test set. Expect a beginner (she\'s just a kitten, after all. You\'re not going to lose, are you!?). '
              'Dinah drawn by Shunnk.',
              style: text.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
