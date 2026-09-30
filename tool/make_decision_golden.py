"""Golden decisions for the example app's parity test: questions in the wire format + the probabilities the Python
reference gets with the same model on ONNX Runtime 1.23 (the plugin's version), one question per forward pass.

Step 1 (any Python with transformers):  python tool/make_decision_golden.py encode --model-dir <hf dir> --td <td test jsonl> --out <dir>
Step 2 (onnxruntime==1.23.0):           python tool/make_decision_golden.py run --onnx <model.onnx> --out <dir>
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

CARD = [
    {"state": "The package arrived broken and I need it for tomorrow.",
     "question": {"type": "choice", "instructions": "What does the customer want?",
                  "criteria": {"refund": "The customer wants their money back", "replacement": "The customer wants a new unit sent",
                               "complaint": "The customer only wants to complain"}}},
    {"state": "Hi, please cancel my subscription starting today.",
     "question": {"type": "noul", "instructions": "Does the customer want to cancel?",
                  "criteria": {"true": "The customer wants to cancel.", "false": "The customer does not want to cancel."}}},
    {"state": "Great product, but shipping took three weeks.",
     "question": {"type": "score", "instructions": "Rate the review.",
                  "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}},
    {"state": {"pedido": {"total": 1200, "limite": 1000}, "cliente": "Olá, quero cancelar minha assinatura a partir de hoje."},
     "question": {"type": "noul", "instructions": "O cliente quer cancelar?"}},
]


def wire(d):
    if d["kind"] == "choice":
        labels = d.get("labels") or [str(i) for i in range(len(d["criteria"]))]
        return {"type": "choice", "instructions": d.get("instructions"), "criteria": dict(zip(labels, d["criteria"]))}
    if d["kind"] == "noul":
        return {"type": "noul", "instructions": d.get("instructions"), "criteria": {"false": d["criteria"][0], "true": d["criteria"][1]}}
    return {"type": "score", "instructions": d.get("instructions"), "criteria": d["criteria"]}


def encode(a):
    import numpy as np
    sys.path.insert(0, a.model_dir)
    from dinah import DinahONNX, QUESTION_TYPES
    m = DinahONNX.__new__(DinahONNX)
    from dinah import _Base
    _Base.__init__(m, Path(a.model_dir))
    rows = [json.loads(l) for l in open(a.td)]
    picked = {"choice": 0, "noul": 0, "score": 0}; cases = list(CARD)
    for d in rows:
        if picked[d["kind"]] < 20:
            picked[d["kind"]] += 1; cases.append({"state": d["state"], "question": wire(d)})
    arrays = {}
    for i, c in enumerate(cases):
        q = c["question"]
        if q["type"] == "choice":
            opts = [v if v is not None else k for k, v in q["criteria"].items()]
        elif q["type"] == "noul":
            cr = q.get("criteria") or {}
            opts = [cr.get("false") or "The statement above is false.", cr.get("true") or q.get("instructions")]
        else:
            opts = q["criteria"]
        ids, pos = m._encode(q.get("instructions"), c["state"], opts)
        b = m._batch([(ids, pos, QUESTION_TYPES[q["type"]])])
        for k, v in b.items():
            arrays[f"{k}_{i}"] = v
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "inputs.npz", **arrays, n=np.array(len(cases)))
    json.dump(cases, open(out / "cases.json", "w"), ensure_ascii=False)
    print("cases", len(cases))


def run(a):
    import numpy as np, onnxruntime as ort
    out = Path(a.out); z = np.load(out / "inputs.npz"); cases = json.load(open(out / "cases.json"))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    for i, c in enumerate(cases):
        feed = {k: z[f"{k}_{i}"] for k in ("ids", "pad_mask", "opt_positions", "opt_mask", "qtype")}
        lg, cf = s.run(["logits", "conf_logit"], feed)
        k = int(feed["opt_mask"][0].sum()); zz = lg[0, :k].astype(np.float64); p = np.exp(zz - zz.max()); p /= p.sum()
        c["expected"] = {"probabilities": p.tolist(), "confidence": float(1 / (1 + np.exp(-float(cf[0])))), "tokens": int(feed["pad_mask"][0].sum())}
    json.dump(cases, open(out / "decisions.json", "w"), ensure_ascii=False)
    print("ort", ort.__version__, "| golden decisions", len(cases))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["encode", "run"])
    ap.add_argument("--model-dir"); ap.add_argument("--td"); ap.add_argument("--onnx"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); encode(a) if a.step == "encode" else run(a)
