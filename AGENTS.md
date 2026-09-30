# AGENTS.md

Guidance for coding agents working in this repository.

## What this is

`decision_ai` is a Flutter package for typed decisions (jev-style: `Choice`, `Noul`, `Score`). The same
`DecisionEngine` interface runs a model on the device (ONNX Runtime), in the browser (ONNX Runtime Web in a worker) or
behind a Decision API provider. On-device models are a `Runtime`, a `Tokenizer`, a `Reader` and a `Calibrator`,
chosen by the model's `decision_ai.json`.

## Layout

- `lib/src/decision_ai.dart`: the `DecisionAI` entry points and the reader registry.
- `lib/src/readers/`: built-in readers (`option_reader.dart` for Dinah-0, `laya_reader.dart`,
  `label_logits_reader.dart` for causal LMs).
- `lib/src/tokenizer.dart`, `lib/src/render.dart`: byte-level BPE and Python-identical JSON rendering.
- `lib/src/runtime.dart` (native) and `lib/src/runtime_web.dart` (web, selected by conditional import).
- `lib/src/model_source.dart`: downloads, SHA-256 checks, cache, and the web loading path.
- `tool/`: Python tools that export models and make reference outputs.
- `example/`: the parity app (integration tests on simulators and devices) and a custom reader example.
- `demo/chess/`: the browser demo (play chess against Dinah-0).
- `skills/port-model-to-decision-ai/`: how to port a new model; read it before touching a reader or an export.

## Commands

```bash
flutter pub get
dart format --output=none --set-exit-if-changed lib test example/lib example/integration_test   # 120 columns
flutter analyze && (cd example && flutter analyze) && (cd demo/chess && flutter analyze)
flutter test                                  # package: tokenizer parity, readers, downloads, API
(cd demo/chess && flutter test)               # the demo speaks Dinah's training format; 200 full games
```

Device parity (needs the model folders served locally; see the top of the file):
`cd example && flutter test integration_test/parity_test.dart -d <device>`.

## Rules that keep the package correct

- **Parity is the definition of correct.** A change to a tokenizer, a reader, rendering or calibration is done only
  when the numbers still match the Python references (unit tests, then device parity, then the browser page in
  `demo/chess/tool/web_parity.dart`).
- **Never truncate.** A request that does not fit raises `RequestTooLong`.
- **The web compiles to JavaScript.** No `Int64List` outside `Tensor.int64`, no bit shifts past 31 bits, and
  remember that `1.0 == 1` there.
- **Exports stay portable.** 8-bit `MatMulNBits` with `accuracy_level` unset; no dynamic int8. Kernels that differ
  between ONNX Runtime builds make Android and iOS disagree.
- Code, comments and docs are in English. Match the surrounding style: short doc comments that say why, 120-column
  lines, no new dependencies without need.
