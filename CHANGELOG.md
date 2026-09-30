## 1.0.0

* Typed decisions (choice, noul, score) with one `DecisionEngine` interface for on-device models, the browser and
  Decision API providers (`api`, `openRouter`).
* Pluggable on-device engine: Runtime (ONNX Runtime; ONNX Runtime Web in a worker on the web), Tokenizer (byte-level
  BPE), Reader and Calibrator.
* Built-in readers: `option-reader` (Dinah-0), `laya` (Laya) and `label-logits` (causal LMs).
* Models bundled as assets, or downloaded from Hugging Face or any URL, SHA-256 checked, cached and usable offline.
* Tools to port models: `tool/export_causal_lm.py` (Hugging Face LLMs) and `tool/port_laya.py` (Laya).
