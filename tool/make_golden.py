"""Build golden files for the Dart port: tokenizer ids and canonical JSON rendering, straight from the Python reference.

    python tool/make_golden.py --model <dir with tokenizer.json> --out test/golden
"""
from __future__ import annotations

import argparse, json, random, unicodedata
from pathlib import Path


def render(content) -> str:
    """Same as dinah.py: strings are stripped; objects and arrays become compact JSON with sorted keys."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


EDGE_TEXTS = [
    "", " ", "a", "Hello world", "Hello  world", "Hello   world", "tabs\there\tand\nnewlines\r\n",
    "trailing spaces   ", "   leading spaces", "multiple\n\n\nblank lines", " " * 30 + "x",
    "Olá, você está bem? Ação, coração, pão, açúcar.", "Café (decomposed) vs Café (composed)",
    "I'll go, you're here, it's done, we've seen, they'd say, I'm ok", "DON'T SHOUT", "número 1.234,56 e 1,234.56",
    "emoji 🙂 and 👍🏽 and 🇧🇷", "中文 日本語 한국어", "العربية", "math: x² + y² = z², ∑, ∫, √2",
    "[OPT] literal special token and [CLS] and [MASK] here", "email |||EMAIL_ADDRESS||| phone |||PHONE_NUMBER|||",
    "<|endoftext|> and <|padding|>", "code: def f(x):\n    return x ** 2  # comment", "URL https://lukita.me/posts/dinah-0/?a=1&b=2",
    "a non-breaking thin​zero-width", "ends with punctuation!!! ??? ...", "CamelCaseWords and snake_case_words",
]

EDGE_VALUES = [
    {"b": 1, "a": 2}, {"z": {"y": [1, 2, {"x": None}], "a": True}, "m": False}, [3, "tres", 3.5, -0.0001, 1e-05, 1e21, 1.0, 100.0],
    {"text": "quote \" backslash \\ tab \t newline \n control \u001b"}, {"ção": "coração", "Zebra": 1, "apple": 2, "_": 3},
    {"emoji": "🙂", "astral_key_\U0001F600": 1, "bmp_key_￿": 2}, {"nested": [[], {}, [{}]]},
    {"float": 0.1 + 0.2, "neg": -3.25, "exp": 6.02e23, "small": 1.5e-7}, "  string with spaces  ", None,
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--samples", nargs="*", default=[]); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--name", default="", help="suffix for the golden file, e.g. qwen")
    a = ap.parse_args()
    # The Rust `tokenizers` library is the reference for tokenizer.json (transformers may load a slow Python tokenizer).
    from tokenizers import Tokenizer
    tok_rs = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    texts = list(EDGE_TEXTS)
    texts += [unicodedata.normalize("NFD", t) for t in EDGE_TEXTS[11:13]]
    rng = random.Random(7)
    for path in a.samples:                      # jsonl files with state/instructions/criteria
        rows = [json.loads(l) for l in open(path)]
        rng.shuffle(rows)
        for d in rows[: a.n]:
            for v in (d.get("state"), d.get("instructions"), d.get("question"), *(d.get("criteria") or d.get("options") or [])):
                if v is not None:
                    texts.append(render(v))
    tok_cases = [{"text": t, "ids": tok_rs.encode(t, add_special_tokens=False).ids} for t in texts]
    values = list(EDGE_VALUES)
    for path in a.samples:
        rows = [json.loads(l) for l in open(path)][: 50]
        values += [d.get("state") for d in rows if isinstance(d.get("state"), (dict, list))]
    render_cases = [{"value": v, "rendered": render(v)} for v in values]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{a.name}" if a.name else ""
    (out / f"tokenizer{suffix}.json").write_text(json.dumps(tok_cases, ensure_ascii=False))
    if not a.name:
        (out / "render.json").write_text(json.dumps(render_cases, ensure_ascii=False))
    print(f"tokenizer cases: {len(tok_cases)} ({sum(len(c['ids']) for c in tok_cases)} ids) | render cases: {len(render_cases)}")


if __name__ == "__main__":
    main()
