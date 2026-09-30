"""Ports a Laya checkpoint (Convai Innovations, Apache-2.0) to decision_ai's `laya` reader.

Laya ships PyTorch weights and its own Python runtime (github.com/NandhaKishorM/laya). This script exports the
checkpoint with Laya's own model code, stores the weights in 8 bits (MatMulNBits, fp32 math, so iOS, Android and
the web give the same numbers) and the embedding table in int8, and writes `decision_ai.json` from the checkpoint's
`rl_agent_config.json`.

Step 1 (torch, onnx, onnxscript, onnxruntime and the laya package):
    pip install "laya @ git+https://github.com/NandhaKishorM/laya" torch onnx onnxscript onnxruntime
    python tool/port_laya.py export --model convaiinnovations/laya --out <dir>
Step 2 (onnxruntime==1.23.0, the plugin's version, plus the laya package):
    python tool/port_laya.py golden --dir <dir> --requests example/assets/golden_decisions.json \
        --out <dir>/golden_laya.json

The reference in step 2 is Laya's own `build_sequence` and temperatures, run on the exported model.
"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path


def export(a):
    import torch
    from huggingface_hub import hf_hub_download
    from laya.agent import Agent

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    agent = Agent(a.model, compile=False, device="cpu", revision=a.revision)
    # Batch, sequence and marker counts all > 1 and different, or torch.export bakes them in (Laya's exporter does
    # the same).
    batch, seq, k = 2, 17, 3
    inputs = (torch.randint(0, 100, (batch, seq)), torch.ones((batch, seq), dtype=torch.long),
              torch.tensor([[1, 5, 9]] * batch), torch.ones((batch, k), dtype=torch.bool), torch.zeros(batch, dtype=torch.long))
    b, s, m = torch.export.Dim("batch"), torch.export.Dim("seq"), torch.export.Dim("markers")
    (out / "fp32").mkdir(exist_ok=True)
    fp32 = out / "fp32" / "model.onnx"
    torch.onnx.export(agent.model.float().eval(), inputs, str(fp32), opset_version=18, do_constant_folding=True,
                      input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
                      output_names=["logits", "act_logits"],
                      dynamic_shapes=({0: b, 1: s}, {0: b, 1: s}, {0: b, 1: m}, {0: b, 1: m}, {0: b}))

    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=8, block_size=32, is_symmetric=True, accuracy_level=None)
    q.process()
    # The torch exporter leaves intermediate shapes that the quantizer's shape inference re-derives differently;
    # they are informational only (Laya's own quantization drops them too).
    del q.model.model.graph.value_info[:]
    nbits = out / "model_nbits8.onnx"
    q.model.save_model_to_file(str(nbits), use_external_data_format=False)
    (out / "onnx").mkdir(exist_ok=True)
    quantize_dynamic(str(nbits), str(out / "onnx" / "model_8bit.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
    nbits.unlink()
    if not a.keep_fp32:
        shutil.rmtree(out / "fp32")

    def fetch(name):
        local = Path(a.model) / name
        return local if local.exists() else Path(hf_hub_download(a.model, name, revision=a.revision))

    shutil.copy(fetch("tokenizer/tokenizer.json"), out / "tokenizer.json")
    cfg = json.loads(fetch("rl_agent_config.json").read_text())
    manifest = {
        "format": "decision-ai/model@1",
        "name": a.name or a.model.rstrip("/").split("/")[-1],
        "license": "apache-2.0",
        "reader": "laya",
        "files": {"model": "onnx/model_8bit.onnx", "tokenizer": "tokenizer.json"},
        "max_length": cfg["max_len"],
        "reader_config": {
            "head_max_len": cfg["head_max_len"],
            "option_max_tokens": 48,
            "temperature": cfg.get("temperature", [1.0, 1.0, 1.0]),
            "temperature_by_options": cfg.get("temperature_by_options", {}),
            "temperature_range": [0.5, 5.0],
        },
        "source": {"model": a.model, "revision": a.revision, "runtime": "https://github.com/NandhaKishorM/laya"},
    }
    (out / "decision_ai.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print("wrote", out / "onnx" / "model_8bit.onnx", f"{(out / 'onnx' / 'model_8bit.onnx').stat().st_size / 1e6:.1f} MB")


def golden(a):
    import numpy as np
    import onnxruntime as ort
    from laya.common import QTYPES, build_sequence, clamp_temperature, temp_bucket
    from laya.onnx_agent import ONNXAgent
    from transformers import PreTrainedTokenizerFast

    d = Path(a.dir)
    m = json.loads((d / "decision_ai.json").read_text())
    cfg = m["reader_config"]
    tok = PreTrainedTokenizerFast(tokenizer_file=str(d / m["files"]["tokenizer"]), cls_token="[CLS]", sep_token="[SEP]",
                                  mask_token="[MASK]", pad_token="[PAD]", unk_token="[UNK]")
    sess = ort.InferenceSession(str(d / m["files"]["model"]), providers=["CPUExecutionProvider"])
    temps = [clamp_temperature(t) for t in cfg["temperature"]]
    by_options = {k: clamp_temperature(v) for k, v in cfg["temperature_by_options"].items()}
    cases = []
    for r in json.loads(Path(a.requests).read_text()):
        q = ONNXAgent._to_internal(r["question"])
        ids, markers, stats = build_sequence(tok, r.get("state"), q, m["max_length"], cfg["head_max_len"],
                                             return_truncation_stats=True)
        assert not stats["truncated"], "the golden requests must fit without truncation"
        k, qt = len(markers), QTYPES[q["t"]]
        feed = {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
                "marker_pos": np.array([markers], dtype=np.int64), "marker_mask": np.ones((1, k), dtype=bool),
                "qtype": np.array([qt], dtype=np.int64)}
        z = sess.run(["logits"], feed)[0][0][:k].astype(np.float64) / by_options.get(temp_bucket(qt, k), temps[qt])
        p = np.exp(z - z.max()); p /= p.sum()
        cases.append({"state": r.get("state"), "question": r["question"], "expected": {"probabilities": p.tolist(), "tokens": len(ids)},
                      "ids": ids, "markers": markers})
    Path(a.out).write_text(json.dumps(cases, ensure_ascii=False) + "\n")
    print(f"{len(cases)} cases, onnxruntime {ort.__version__}, max tokens {max(c['expected']['tokens'] for c in cases)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", default="convaiinnovations/laya", help="Hub id or local checkpoint folder")
    e.add_argument("--revision"); e.add_argument("--out", required=True); e.add_argument("--name")
    e.add_argument("--keep-fp32", action="store_true", help="keep the fp32 export in <out>/fp32/")
    g = sub.add_parser("golden")
    g.add_argument("--dir", required=True); g.add_argument("--requests", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else golden(a)


if __name__ == "__main__":
    main()
