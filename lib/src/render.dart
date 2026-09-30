/// Renders state, instructions and options exactly as Dinah saw them in training (Python reference):
/// strings are stripped; objects and arrays become compact JSON with sorted keys
/// (`json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`).
String render(Object? content) {
  if (content == null) return '';
  if (content is String) return _pyStrip(content);
  return pythonJson(content, sortKeys: true, itemSeparator: ',', keySeparator: ':');
}

/// Python's `json.dumps(value, ensure_ascii=False, ...)`: defaults to its default separators (`", "`, `": "`) and keys
/// in insertion order, as models that were trained on `json.dumps(state)` saw their input.
String pythonJson(Object? value, {bool sortKeys = false, String itemSeparator = ', ', String keySeparator = ': '}) {
  final out = StringBuffer();
  _writeJson(value, out, sortKeys, itemSeparator, keySeparator);
  return out.toString();
}

void _writeJson(Object? v, StringBuffer out, bool sortKeys, String itemSep, String keySep) {
  if (v == null) {
    out.write('null');
  } else if (v is bool) {
    out.write(v ? 'true' : 'false');
  } else if (v is int) {
    out.write(v.toString());
  } else if (v is double) {
    out.write(_pyFloat(v));
  } else if (v is String) {
    _writeString(v, out);
  } else if (v is Map) {
    final keys = v.keys.toList();
    if (sortKeys) keys.sort((a, b) => _compareCodePoints(a.toString(), b.toString()));
    out.write('{');
    for (var i = 0; i < keys.length; i++) {
      if (i > 0) out.write(itemSep);
      _writeString(keys[i].toString(), out);
      out.write(keySep);
      _writeJson(v[keys[i]], out, sortKeys, itemSep, keySep);
    }
    out.write('}');
  } else if (v is Iterable) {
    out.write('[');
    var first = true;
    for (final e in v) {
      if (!first) out.write(itemSep);
      first = false;
      _writeJson(e, out, sortKeys, itemSep, keySep);
    }
    out.write(']');
  } else {
    throw ArgumentError('not JSON content: ${v.runtimeType}');
  }
}

/// Python sorts dict keys by code point; Dart's String.compareTo compares UTF-16 code units.
int _compareCodePoints(String a, String b) {
  final ia = a.runes.iterator, ib = b.runes.iterator;
  while (true) {
    final ha = ia.moveNext(), hb = ib.moveNext();
    if (!ha || !hb) return ha == hb ? 0 : (ha ? 1 : -1);
    final c = ia.current.compareTo(ib.current);
    if (c != 0) return c;
  }
}

void _writeString(String s, StringBuffer out) {
  out.write('"');
  for (final c in s.codeUnits) {
    switch (c) {
      case 0x22:
        out.write(r'\"');
      case 0x5c:
        out.write(r'\\');
      case 0x0a:
        out.write(r'\n');
      case 0x0d:
        out.write(r'\r');
      case 0x09:
        out.write(r'\t');
      case 0x08:
        out.write(r'\b');
      case 0x0c:
        out.write(r'\f');
      default:
        if (c < 0x20) {
          out.write('\\u${c.toRadixString(16).padLeft(4, '0')}');
        } else {
          out.writeCharCode(c);
        }
    }
  }
  out.write('"');
}

/// Python's float repr: shortest round-trip digits; fixed notation when -4 <= exponent < 16, else `d.ddde+XX`.
String _pyFloat(double v) {
  if (v.isNaN) return 'NaN';
  if (v.isInfinite) return v > 0 ? 'Infinity' : '-Infinity';
  if (v == 0) return (1 / v).isNegative ? '-0.0' : '0.0';
  final sign = v < 0 ? '-' : '';
  final s = v.abs().toString();
  var mantissa = s;
  var exp = 0;
  final e = s.indexOf('e');
  if (e >= 0) {
    mantissa = s.substring(0, e);
    exp = int.parse(s.substring(e + 1));
  }
  final dot = mantissa.indexOf('.');
  final intPart = dot >= 0 ? mantissa.substring(0, dot) : mantissa;
  final frac = dot >= 0 ? mantissa.substring(dot + 1) : '';
  final all = intPart + frac;
  final stripped = all.replaceFirst(RegExp(r'^0+'), '');
  final leadingZeros = all.length - stripped.length;
  // Decimal exponent of the first significant digit.
  final point = intPart.length + exp - leadingZeros - 1;
  var digits = stripped.replaceFirst(RegExp(r'0+$'), '');
  if (digits.isEmpty) digits = '0';
  if (point >= -4 && point < 16) {
    if (point >= 0) {
      final whole = digits.length > point + 1 ? digits.substring(0, point + 1) : digits.padRight(point + 1, '0');
      final rest = digits.length > point + 1 ? digits.substring(point + 1) : '';
      return '$sign$whole.${rest.isEmpty ? '0' : rest}';
    }
    return '${sign}0.${'0' * (-point - 1)}$digits';
  }
  final m = digits.length > 1 ? '${digits[0]}.${digits.substring(1)}' : digits;
  final es = point < 0 ? '-' : '+';
  return '$sign${m}e$es${point.abs().toString().padLeft(2, '0')}';
}

/// Python's str.strip(): the characters for which str.isspace() is true.
String _pyStrip(String s) {
  var a = 0, b = s.length;
  while (a < b && _pySpace(s.codeUnitAt(a))) {
    a++;
  }
  while (b > a && _pySpace(s.codeUnitAt(b - 1))) {
    b--;
  }
  return s.substring(a, b);
}

bool _pySpace(int c) =>
    (c >= 0x09 && c <= 0x0d) ||
    (c >= 0x1c && c <= 0x20) ||
    c == 0x85 ||
    c == 0xa0 ||
    c == 0x1680 ||
    (c >= 0x2000 && c <= 0x200a) ||
    c == 0x2028 ||
    c == 0x2029 ||
    c == 0x202f ||
    c == 0x205f ||
    c == 0x3000;
