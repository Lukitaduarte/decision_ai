import 'dart:convert';

import 'package:unorm_dart/unorm_dart.dart' as unorm;

/// Text to token ids. The library ships [BpeTokenizer]; implement this to plug another tokenizer.
abstract interface class Tokenizer {
  List<int> encode(String text);
  int? tokenId(String token);
}

/// Byte-level BPE from a Hugging Face `tokenizer.json` (GPT-2, ModernBERT, Qwen, SmolLM and similar families).
///
/// Supported pieces: normalizer `NFC`, `NFKC`, `Sequence` of those, or none; pre-tokenizers `ByteLevel`, `Split`
/// (regex, `Isolated`), `Digits`, and `Sequence` of those; added tokens matched first (leftmost-longest). Anything
/// else (SentencePiece-style `Metaspace`, for example) raises [UnsupportedError]: plug your own [Tokenizer] instead.
class BpeTokenizer implements Tokenizer {
  BpeTokenizer._(this._vocab, this._ranks, this._addedFirst, this._normalize, this._steps, this._ignoreMerges);

  final Map<String, int> _vocab;
  final Map<String, int> _ranks;
  final Map<int, List<_AddedToken>> _addedFirst;
  final String Function(String) _normalize;
  final List<List<String> Function(List<String>)> _steps;
  final bool _ignoreMerges;
  final Map<String, List<int>> _cache = {};

  static const _gpt2Pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
  static final List<String> _byteToChar = _bytesToUnicode();

  /// Builds the tokenizer from the contents of a `tokenizer.json` file.
  factory BpeTokenizer.fromJson(String source) {
    final json = jsonDecode(source) as Map<String, dynamic>;
    final model = json['model'] as Map<String, dynamic>;
    if (model['type'] != 'BPE') {
      throw UnsupportedError('only BPE tokenizers are supported, got ${model['type']}');
    }
    final vocab = (model['vocab'] as Map<String, dynamic>).map((k, v) => MapEntry(k, v as int));
    final ranks = <String, int>{};
    final merges = model['merges'] as List<dynamic>;
    for (var i = 0; i < merges.length; i++) {
      final m = merges[i];
      final pair = m is List ? '${m[0]}\u0000${m[1]}' : (m as String).replaceFirst(' ', '\u0000');
      ranks[pair] = i;
    }
    final added = <_AddedToken>[];
    for (final a in (json['added_tokens'] as List<dynamic>? ?? const []).cast<Map<String, dynamic>>()) {
      final content = a['content'] as String;
      added.add(_AddedToken(content, a['id'] as int, a['lstrip'] == true, a['rstrip'] == true));
      vocab.putIfAbsent(content, () => a['id'] as int);
    }
    final first = <int, List<_AddedToken>>{};
    for (final t in added) {
      if (t.content.isEmpty) continue;
      first.putIfAbsent(t.content.codeUnitAt(0), () => []).add(t);
    }
    for (final list in first.values) {
      list.sort((x, y) => y.content.length.compareTo(x.content.length));
    }
    return BpeTokenizer._(
      vocab,
      ranks,
      first,
      _normalizer(json['normalizer']),
      _preTokenizer(json['pre_tokenizer']),
      model['ignore_merges'] == true,
    );
  }

  /// Id of a token string (special tokens included), or null.
  @override
  int? tokenId(String token) => _vocab[token];

  /// Token ids for [text], without special tokens.
  @override
  List<int> encode(String text) {
    if (text.isEmpty) return const [];
    final normalized = _normalize(text);
    final out = <int>[];
    for (final piece in _splitAdded(normalized)) {
      if (piece.added != null) {
        out.add(piece.added!.id);
        continue;
      }
      var pieces = [piece.text];
      for (final step in _steps) {
        pieces = step(pieces);
      }
      for (final p in pieces) {
        out.addAll(_bpeIds(p));
      }
    }
    return out;
  }

  static String Function(String) _normalizer(Object? cfg) {
    if (cfg == null) return (s) => s;
    final m = cfg as Map<String, dynamic>;
    switch (m['type']) {
      case 'NFC':
        return unorm.nfc;
      case 'NFKC':
        return unorm.nfkc;
      case 'Sequence':
        final parts = (m['normalizers'] as List).map(_normalizer).toList();
        return (s) => parts.fold(s, (acc, f) => f(acc));
      default:
        throw UnsupportedError('normalizer ${m['type']} is not supported; plug your own Tokenizer');
    }
  }

  static List<List<String> Function(List<String>)> _preTokenizer(Object? cfg) {
    if (cfg == null) return const [];
    final m = cfg as Map<String, dynamic>;
    switch (m['type']) {
      case 'Sequence':
        return [for (final p in m['pretokenizers'] as List) ..._preTokenizer(p)];
      case 'ByteLevel':
        if (m['add_prefix_space'] == true) {
          throw UnsupportedError('ByteLevel add_prefix_space is not supported; plug your own Tokenizer');
        }
        return m['use_regex'] == false ? const [] : [_isolate(RegExp(_gpt2Pattern, unicode: true))];
      case 'Split':
        if (m['behavior'] != 'Isolated' || m['invert'] == true) {
          throw UnsupportedError(
            'Split ${m['behavior']} (invert ${m['invert']}) is not supported; plug your own Tokenizer',
          );
        }
        final pattern = m['pattern'] as Map<String, dynamic>;
        final re = pattern.containsKey('Regex')
            ? RegExp(_dartRegex(pattern['Regex'] as String), unicode: true)
            : RegExp(RegExp.escape(pattern['String'] as String));
        return [_isolate(re)];
      case 'Digits':
        final digit = RegExp(m['individual_digits'] == true ? r'\p{N}' : r'\p{N}+', unicode: true);
        return [_isolate(digit)];
      default:
        throw UnsupportedError('pre-tokenizer ${m['type']} is not supported; plug your own Tokenizer');
    }
  }

  /// Splits each piece around the matches of [re], keeping matches and the text between them.
  static List<String> Function(List<String>) _isolate(RegExp re) => (pieces) {
    final out = <String>[];
    for (final s in pieces) {
      var last = 0;
      for (final m in re.allMatches(s)) {
        if (m.end == m.start) continue;
        if (m.start > last) out.add(s.substring(last, m.start));
        out.add(m[0]!);
        last = m.end;
      }
      if (last < s.length) out.add(s.substring(last));
    }
    return out;
  };

  /// Rewrites the inline case-insensitive groups of Hugging Face regexes, `(?i:'s|'t)`, which Dart does not support.
  static String _dartRegex(String p) {
    final out = StringBuffer();
    var i = 0;
    while (i < p.length) {
      if (p.startsWith('(?i:', i)) {
        var depth = 1, j = i + 4;
        final inner = StringBuffer();
        while (j < p.length && depth > 0) {
          final c = p[j];
          if (c == '(') depth++;
          if (c == ')') depth--;
          if (depth > 0) {
            final lo = c.toLowerCase(), up = c.toUpperCase();
            inner.write(lo != up ? '[$lo$up]' : c);
          }
          j++;
        }
        out.write('(?:$inner)');
        i = j;
      } else {
        out.write(p[i]);
        i++;
      }
    }
    return out.toString();
  }

  /// Splits on added tokens, leftmost-longest, honoring `lstrip`/`rstrip`.
  List<_Piece> _splitAdded(String text) {
    final pieces = <_Piece>[];
    var start = 0;
    var i = 0;
    while (i < text.length) {
      final candidates = _addedFirst[text.codeUnitAt(i)];
      _AddedToken? hit;
      if (candidates != null) {
        for (final t in candidates) {
          if (text.startsWith(t.content, i)) {
            hit = t;
            break;
          }
        }
      }
      if (hit == null) {
        i++;
        continue;
      }
      var before = text.substring(start, i);
      if (hit.lstrip) before = before.trimRight();
      if (before.isNotEmpty) pieces.add(_Piece(before, null));
      pieces.add(_Piece(hit.content, hit));
      i += hit.content.length;
      if (hit.rstrip) {
        while (i < text.length && _isSpace(text.codeUnitAt(i))) {
          i++;
        }
      }
      start = i;
    }
    if (start < text.length) pieces.add(_Piece(text.substring(start), null));
    return pieces;
  }

  List<int> _bpeIds(String word) {
    final cached = _cache[word];
    if (cached != null) return cached;
    final symbols = utf8.encode(word).map((b) => _byteToChar[b]).toList();
    if (_ignoreMerges) {
      final whole = _vocab[symbols.join()];
      if (whole != null) return [whole];
    }
    while (symbols.length > 1) {
      var best = -1;
      var bestRank = 0;
      for (var j = 0; j < symbols.length - 1; j++) {
        final r = _ranks['${symbols[j]}\u0000${symbols[j + 1]}'];
        // No sentinel like `1 << 62`: compiled to JavaScript, shifts are 32-bit and that is 0.
        if (r != null && (best < 0 || r < bestRank)) {
          bestRank = r;
          best = j;
        }
      }
      if (best < 0) break;
      final left = symbols[best], right = symbols[best + 1];
      final merged = <String>[];
      var j = 0;
      while (j < symbols.length) {
        if (j < symbols.length - 1 && symbols[j] == left && symbols[j + 1] == right) {
          merged.add(left + right);
          j += 2;
        } else {
          merged.add(symbols[j]);
          j++;
        }
      }
      symbols
        ..clear()
        ..addAll(merged);
    }
    final ids = <int>[];
    for (final s in symbols) {
      final id = _vocab[s];
      if (id == null) throw StateError('token not in vocabulary: $s');
      ids.add(id);
    }
    if (_cache.length < 50000) _cache[word] = ids;
    return ids;
  }

  static bool _isSpace(int c) => c == 0x20 || (c >= 0x09 && c <= 0x0d);

  static List<String> _bytesToUnicode() {
    final bs = <int>[
      for (var b = 0x21; b <= 0x7e; b++) b,
      for (var b = 0xa1; b <= 0xac; b++) b,
      for (var b = 0xae; b <= 0xff; b++) b,
    ];
    final cs = List<int>.of(bs);
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n++;
      }
    }
    final table = List<String>.filled(256, '');
    for (var i = 0; i < bs.length; i++) {
      table[bs[i]] = String.fromCharCode(cs[i]);
    }
    return table;
  }
}

class _AddedToken {
  _AddedToken(this.content, this.id, this.lstrip, this.rstrip);
  final String content;
  final int id;
  final bool lstrip;
  final bool rstrip;
}

class _Piece {
  _Piece(this.text, this.added);
  final String text;
  final _AddedToken? added;
}
