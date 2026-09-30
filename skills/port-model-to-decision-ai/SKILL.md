---
name: port-model-to-decision-ai
description: "Port a Hugging Face model to the decision_ai Flutter package (typed decisions, jev-style: choice, noul, score) and configure an app to use it. Use when someone wants to run a Hugging Face model (a causal LM, Laya, a GLiClass/encoder classifier, or any other decision model) on device or in the browser with decision_ai, export it to ONNX, write its decision_ai.json manifest, write a custom Reader, or check that the port matches the original model."
---

# Porting a Hugging Face model to decision_ai

decision_ai answers typed questions (`Choice`, `Noul`, `Score`) with one forward pass that scores the options. An
on-device model is four pieces: a `Runtime` (ONNX Runtime; ONNX Runtime Web on the web), a `Tokenizer`
(byte-level BPE from `tokenizer.json`), a `Reader` (how the model family reads a question and turns outputs into one
logit per option) and a `Calibrator`. A port is correct only when the app's probabilities match the model's own
Python code on the same requests. Never skip that check: every bug found while building this package (a tokenizer
difference, a platform-dependent kernel, a browser integer quirk) looked fine until the numbers were compared.

## 1. Identify the model family

| the model is | reader | how to port |
|---|---|---|
| an instruction-tuned causal LM (SmolLM, Qwen, Llama 3 style) | `label-logits` (built in) | `tool/export_causal_lm.py` |
| Laya (`convaiinnovations/laya*`) | `laya` (built in) | `tool/port_laya.py` |
| Dinah-0 or a model trained on its protocol | `option-reader` (built in) | already published: `Lukitaduarte/dinah-0` |
| any other classifier or encoder (GLiClass, a custom head) | your own `Reader` | section 4 |

Read the model's own inference code before anything else (its repository, `modeling_*.py`, the authors' API
wrapper). The input format (order of instructions, options and state; special tokens; how options are written;
truncation; option order for noul), the output (which logits, in which order, an abstention slot) and the calibration
(temperatures, per option count or per question type) all come from there. Prefer the maintained upstream repository
over a copy on the Hub: they drift (Laya's GitHub renders structured criteria differently from the copy on its model
card).

## 2. Check the tokenizer

`BpeTokenizer` reads byte-level BPE `tokenizer.json` files: normalizers NFC/NFKC, pre-tokenizers ByteLevel, Split
(regex, including inline `(?i:...)` groups), Digits, and added tokens. SentencePiece (`Metaspace`) and WordPiece are
not built in: implement `Tokenizer` and pass it with `tokenizer:`.

Make references with the Rust `tokenizers` library (`Tokenizer.from_file(...).encode(text,
add_special_tokens=False).ids`), not `transformers.AutoTokenizer`, which can load a slow Python tokenizer that splits
runs of spaces differently (it did for SmolLM2).

## 3. Use a published ONNX, or export one

First look for an ONNX file that already exists: the model repository's `onnx/` folder, the authors' own export
(Verdict ships `model.onnx`), or a community port. For a causal LM, `python tool/export_causal_lm.py manifest --model
<repo> --onnx onnx/<file>.onnx --template-file <template> --out <dir>` writes the manifest from `config.json` with no
export (the reader feeds the empty attention cache and reads the last position). Before settling on a published file,
compare it with fp32 on the same requests: SmolLM2's own `model_q4.onnx` agreed with its fp32 on 45 of 64 answers,
while an 8-bit export agreed on 64 of 64, and full-sequence logits made each question about 2.5 times slower. Avoid
published dynamic int8 files (`model_int8`, `model_quantized`): their kernels differ between Android and iOS.

Export yourself when no suitable file exists or when size, speed or fidelity matter:

- Export only the logits you need. For a causal LM, wrap the model so the graph returns the last position,
  `[1, vocab]`; `[1, length, vocab]` is copied through the platform channel on every question.
- Declare dynamic sequence (and option-count) axes. If the model has a hand-written attention reshape, export with
  `torch.onnx.export(..., dynamo=True, dynamic_shapes=...)` and dummy inputs whose batch, length and option count are
  all different and greater than 1, or the traced sizes get baked in.
- Quantize weights to 8 bits with `MatMulNBitsQuantizer(model, bits=8, block_size=32, is_symmetric=True,
  accuracy_level=None)` and the embedding table with `quantize_dynamic(..., op_types_to_quantize=["Gather"])`.
  **Leave `accuracy_level` unset**: level 4 quantizes activations at run time, and its int8 kernels differ between
  the Android and iOS builds of ONNX Runtime (61 of 64 answers on Android against 64 of 64 on iOS for SmolLM2).
  Avoid dynamic int8 (`MatMulInteger`) for the same reason.
- Before quantizing a dynamo export, `del model.graph.value_info[:]` (its intermediate shapes disagree with the
  quantizer's shape inference).
- Models above 2 GB are written with external weight files: export into a separate folder and delete the whole
  folder afterwards, so no stray weight files end up next to the published model.
- Measure the quantized model against fp32 (argmax agreement and max probability difference on the same requests).
  Heavy calibration temperatures leave near-ties that quantization flips; if it moves more than a couple of answers,
  ship fp32 (Verdict stays fp32 for this reason).

## 4. Write a Reader (only for a new family)

```dart
class MyReader implements Reader {
  MyReader(this.manifest, this.tokenizer);
  final ModelManifest manifest;
  final Tokenizer tokenizer;

  @override
  ReaderInput encode(Object? state, Question question) { /* the model's input format, exactly */ }

  @override
  Readout decode(ReaderInput input, Map<String, Tensor> outputs) { /* one logit per option */ }
}

DecisionAI.registerReader('my-reader', MyReader.new);
```

- Return logits in the library's option order: `Choice` in the order of its map, `Noul` as `[false, true]`,
  `Score` from the lowest level. If the model lists options differently (Verdict puts true first), reorder in
  `decode`.
- An "insufficient evidence" option goes last, with `abstentionIndex` set; the library splits it off and renormalizes.
- Render text exactly as the model saw it in training: `render()` is Python's `json.dumps(sort_keys=True,
  separators=(",", ":"))` with stripped strings; `pythonJson()` is `json.dumps` with default separators and keys in
  insertion order. Check whether the model strips strings, sorts keys, and how it writes floats.
- If a request does not fit, throw `RequestTooLong`. Never truncate silently, even if the original runtime does.
- Temperatures that depend on more than the option count (Laya's depend on the question type) are applied in
  `decode`; otherwise use `TemperatureCalibrator` through the manifest's `calibration`.
- Use `Tensor.int64` for ids, never `Int64List` directly, and no bit shifts beyond 31 bits: the web build compiles to
  JavaScript, where `Int64List` does not exist and `1 << 62` is 0 (that one line broke every BPE merge in the
  browser).
- Complete examples: `lib/src/readers/laya_reader.dart` (built in) and `example/lib/gliclass_reader.dart` (an app-side
  reader for Verdict).

## 5. Write decision_ai.json

```json
{"format": "decision-ai/model@1", "name": "my-model", "license": "apache-2.0", "reader": "label-logits",
 "files": {"model": "onnx/model_8bit.onnx", "tokenizer": "tokenizer.json"},
 "sha256": {"onnx/model_8bit.onnx": "..."},
 "max_length": 2048, "calibration": {"temperature": 1.0}, "reader_config": {}}
```

`reader_config` belongs to the reader (see the reader's doc comment for its keys). A repository without this file can
still be used with a `ModelManifest.custom(...)` in code.

## 6. Prove the port

1. **Reference**: run the model's own formatting code on the exported ONNX with `onnxruntime==1.23.0` (the version the
   app runs) over `example/assets/golden_decisions.json`, and save the probabilities (see the `golden` commands of
   the two tools). Save the token ids too when a pure-Dart test can compare them.
2. **Dart, no device**: a unit test that the reader builds the same token ids and markers as the reference
   (`test/laya_test.dart` does this for all 64 requests).
3. **Device**: `example/integration_test/parity_test.dart` on the iOS simulator and an Android emulator, with the
   model folder served locally (instructions at the top of the file). Expect 64 of 64 and max |Δp| below 2e-3.
4. **Browser**: build a page like `demo/chess/tool/web_parity.dart` and read its console. One known difference: a
   whole-number float (`532.0`) is `532` in JavaScript.

## 7. Configure the app

```yaml
dependencies:
  decision_ai: ^1.0.0
```

```dart
final ai = await DecisionAI.huggingFace('org/model', revision: '<commit>'); // or local(), remote(), custom()
final answers = await ai.decide(state: state, questions: {'intent': Choice({...}, instructions: '...')});
```

- iOS: `platform :ios, '16.0'` and `use_frameworks! :linkage => :static` in the Podfile.
- Android: `android.permission.INTERNET` in the main manifest (downloads, APIs) and
  `-keep class ai.onnxruntime.** { *; }` in `android/app/proguard-rules.pro` (release builds).
- Web: serve `ort.wasm.min.js`, `ort-wasm-simd-threaded.mjs` and `ort-wasm-simd-threaded.wasm` from
  `onnxruntime-web@1.23.0` under `web/ort/`, load `ort/ort.wasm.min.js` before `flutter_bootstrap.js`, and build with
  `--no-web-resources-cdn`. Inference runs in ONNX Runtime Web's worker, so the page does not freeze.
