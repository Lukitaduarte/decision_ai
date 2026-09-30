"""Reference decisions for the example's GLiClass reader, using Verdict (heman10x/rlcd-modernbert-151m, Apache-2.0).

The prompts come from Verdict's own `core/formatting.py`, so the Dart reader is checked against the authors' contract,
not against a second copy of it.

Step 1 (copies the fp32 model; --quantize writes an 8-bit one instead, which moved 10 of 64 argmaxes here: the
calibration temperatures, up to 5.0, leave near-ties that small logit shifts flip):
    python example/tool/gliclass_golden.py prepare --src <snapshot of heman10x/rlcd-modernbert-151m> --out <dir>
Step 2 (onnxruntime==1.23.0, tokenizers, numpy, pydantic; the Verdict repository checked out at --verdict-repo):
    python example/tool/gliclass_golden.py golden --verdict-repo <openJev-verdict-2.0> --dir <dir> \
        --requests example/assets/golden_decisions.json --out <dir>/golden_gliclass.json

Score levels are given the value of their index (0, 1, ...), the scale the library reports scores on.
"""
from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path


def prepare(a):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("tokenizer.json", "calibrator.json"):
        shutil.copy(src / f, out / f)
    if not a.quantize:
        shutil.copy(src / "model.onnx", out / "model.onnx")
        return print("copied", out / "model.onnx")
    q = MatMulNBitsQuantizer(onnx.load(str(src / "model.onnx")), bits=8, block_size=32, is_symmetric=True, accuracy_level=a.accuracy_level)
    q.process()
    tmp = out / "model_nbits8.onnx"
    q.model.save_model_to_file(str(tmp), use_external_data_format=False)
    quantize_dynamic(str(tmp), str(out / "model_8bit.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    tmp.unlink()
    print("wrote", out / "model_8bit.onnx", f"{(out / 'model_8bit.onnx').stat().st_size / 1e6:.1f} MB")


def text(v):
    if v is None:
        return ""
    return v.strip() if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def golden(a):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    sys.path.insert(0, a.verdict_repo)
    from core.formatting import build_model_input, format_query
    from core.primitives import Choice, Level, Noul, Option, Score

    d = Path(a.dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    cal = json.loads((d / "calibrator.json").read_text())
    sess = ort.InferenceSession(str(d / a.model), providers=["CPUExecutionProvider"])
    cases = []
    for r in json.loads(Path(a.requests).read_text()):
        w, context = r["question"], text(r.get("state"))
        question = text(w.get("instructions"))
        if w["type"] == "choice":
            q = Choice(id="q", question=question, options=[Option(id=k, description=text(v if v is not None else k)) for k, v in w["criteria"].items()])
        elif w["type"] == "score":
            q = Score(id="q", question=question, levels=[Level(id=str(i), description=text(v), value=i) for i, v in enumerate(w["criteria"])])
        else:
            q = Noul(id="q", proposition=question, semantics="conditional_on_sufficient_evidence_v2")
        formatted, labels, ids = format_query(context, q)
        prompt = build_model_input(q.question, context, labels) if q.kind in ("choice", "score") else build_model_input("", formatted, labels)
        enc = tok.encode(prompt, add_special_tokens=True).ids
        assert len(enc) <= 512, len(enc)
        x = np.array([enc], dtype=np.int64)
        logits = sess.run(None, {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0][: len(ids)].astype(np.float64)
        t = float(cal["per_k"].get(str(len(ids)), cal["temperature"]))
        z = logits / t
        p = np.exp(z - z.max()); p /= p.sum()
        cases.append({"state": r.get("state"), "question": w, "expected": {"ids": ids, "probabilities": p.tolist(), "tokens": len(enc)}})
    Path(a.out).write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases, onnxruntime {ort.__version__}, max tokens {max(c['expected']['tokens'] for c in cases)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--accuracy-level", type=int, help="see tool/export_causal_lm.py")
    g = sub.add_parser("golden")
    g.add_argument("--verdict-repo", required=True); g.add_argument("--dir", required=True); g.add_argument("--model", default="model.onnx")
    g.add_argument("--requests", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else golden(a)


if __name__ == "__main__":
    main()
