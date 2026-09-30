"""Exports a Hugging Face causal LM for the `label-logits` reader and makes golden decisions for the parity test.

The ONNX graph returns only the logits of the last position, `[1, vocab]`, so the app never copies a
`[length, vocab]` tensor through the platform channel. Weights are 8-bit (MatMulNBits, block 32) and the embedding
table int8, the recipe that kept Dinah-0 closest to fp32 on ONNX Runtime 1.23. The math stays in fp32 by default, so
Android and iOS give the same numbers; `--accuracy-level 4` (int8 activations) did not (61 of 64 argmaxes on Android).

Step 1 (torch + transformers + onnxruntime):
    python tool/export_causal_lm.py export --model HuggingFaceTB/SmolLM2-135M-Instruct --out <dir>
Step 2 (onnxruntime==1.23.0, the plugin's version; tokenizers; numpy):
    python tool/export_causal_lm.py golden --dir <dir> --requests example/assets/golden_decisions.json \
        --out <dir>/golden_llm.json

Step 2 re-implements the reader from the manifest (prompt, keys, render) so the app can be checked against it.
"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path

SMOLLM_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
    "<|im_start|>user\n{instructions}\n\nContext:\n{state}\n\nOptions:\n{options}\n\n"
    "Answer with the letter of the best option only.<|im_end|>\n<|im_start|>assistant\n"
)


def template(a):
    if a.template_file:
        return Path(a.template_file).read_text()
    return a.template or SMOLLM_TEMPLATE


def export(a):
    import torch
    from transformers import AutoModelForCausalLM

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").eval()

    class LastLogits(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]

    ids = torch.ones(1, 16, dtype=torch.long)
    # Models above 2 GB are written with their weights in separate files: keep them all in one folder.
    (out / "fp32").mkdir(exist_ok=True)
    fp32 = out / "fp32" / "model.onnx"
    torch.onnx.export(LastLogits(model), (ids, torch.ones_like(ids)), str(fp32), input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], dynamic_axes={"input_ids": {1: "length"}, "attention_mask": {1: "length"}},
                      opset_version=17, dynamo=False)

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=8, block_size=32, is_symmetric=True, accuracy_level=a.accuracy_level)
    q.process()
    nbits = out / "model_nbits8.onnx"
    q.model.save_model_to_file(str(nbits), use_external_data_format=False)
    (out / "onnx").mkdir(exist_ok=True)
    quantize_dynamic(str(nbits), str(out / "onnx" / "model_8bit.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    nbits.unlink()
    if not a.keep_fp32:
        shutil.rmtree(out / "fp32")

    from huggingface_hub import hf_hub_download
    src = Path(a.model) / "tokenizer.json" if Path(a.model).is_dir() else Path(hf_hub_download(a.model, "tokenizer.json"))
    shutil.copy(src, out / "tokenizer.json")
    manifest = {
        "format": "decision-ai/model@1",
        "name": a.name or a.model.split("/")[-1],
        "reader": "label-logits",
        "files": {"model": "onnx/model_8bit.onnx", "tokenizer": "tokenizer.json"},
        "max_length": a.max_length,
        "reader_config": {"template": template(a), "option_line": "{key}. {text}", "key_prefix": ""},
    }
    (out / "decision_ai.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("wrote", out / "onnx" / "model_8bit.onnx", f"{(out / 'onnx' / 'model_8bit.onnx').stat().st_size / 1e6:.1f} MB")


def render(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def options(q, cfg):
    c = q.get("criteria")
    if q["type"] == "choice":
        return list(c.keys()), [v if v is not None else k for k, v in c.items()]
    if q["type"] == "noul":
        c = c or {}
        return ["false", "true"], [c.get("false", cfg.get("noul_false_default", "The statement is false.")), c.get("true", q.get("instructions"))]
    return [str(i) for i in range(len(c))], list(c)


def golden(a):
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer

    d = Path(a.dir)
    m = json.loads((d / "decision_ai.json").read_text())
    cfg = m["reader_config"]
    tok = Tokenizer.from_file(str(d / m["files"]["tokenizer"]))
    sess = ort.InferenceSession(str(d / m["files"]["model"]), providers=["CPUExecutionProvider"])
    keys = [chr(c) for c in range(65, 91)]
    cases = []
    for r in json.loads(Path(a.requests).read_text()):
        q = r["question"]
        labels, texts = options(q, cfg)
        listed = "\n".join(cfg["option_line"].replace("{key}", keys[i]).replace("{text}", render(t)) for i, t in enumerate(texts))
        prompt = cfg["template"].replace("{instructions}", render(q.get("instructions"))).replace("{state}", render(r.get("state"))) \
            .replace("{options}", listed)
        ids = tok.encode(prompt, add_special_tokens=False).ids
        key_ids = []
        for k in keys[:len(texts)]:
            enc = tok.encode(cfg.get("key_prefix", "") + k, add_special_tokens=False).ids
            assert len(enc) == 1, k
            key_ids.append(enc[0])
        x = np.array([ids], dtype=np.int64)
        logits = sess.run(["logits"], {"input_ids": x, "attention_mask": np.ones_like(x)})[0][0]
        z = logits[key_ids].astype(np.float64)
        p = np.exp(z - z.max()); p /= p.sum()
        cases.append({"state": r.get("state"), "question": q, "expected": {"probabilities": p.tolist(), "tokens": len(ids)}})
    Path(a.out).write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases, onnxruntime {ort.__version__}, max tokens {max(c['expected']['tokens'] for c in cases)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", required=True); e.add_argument("--out", required=True); e.add_argument("--name")
    e.add_argument("--template", help="prompt with {instructions}, {state} and {options}; default: SmolLM2's chat format")
    e.add_argument("--template-file", help="the same, read from a file (easier for chat markup)")
    e.add_argument("--max-length", type=int, default=2048)
    e.add_argument("--keep-fp32", action="store_true", help="keep the fp32 export in <out>/fp32/")
    e.add_argument("--accuracy-level", type=int, help="MatMulNBits compute precision; unset keeps fp32 math, the same on every platform. 4 (int8 activations) is faster but its kernels differ between ONNX Runtime builds (Android vs iOS).")
    g = sub.add_parser("golden")
    g.add_argument("--dir", required=True); g.add_argument("--requests", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else golden(a)


if __name__ == "__main__":
    main()
