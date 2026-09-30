# decision_ai

[![pub package](https://img.shields.io/pub/v/decision_ai.svg)](https://pub.dev/packages/decision_ai)
[![pub points](https://img.shields.io/pub/points/decision_ai)](https://pub.dev/packages/decision_ai/score)
[![likes](https://img.shields.io/pub/likes/decision_ai)](https://pub.dev/packages/decision_ai/score)
[![CI](https://github.com/Lukitaduarte/decision_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Lukitaduarte/decision_ai/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Lukitaduarte/decision_ai/actions/workflows/codeql.yml/badge.svg)](https://github.com/Lukitaduarte/decision_ai/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/Lukitaduarte/decision_ai/graph/badge.svg)](https://codecov.io/gh/Lukitaduarte/decision_ai)
[![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Lukitaduarte/decision_ai?labelColor=171717&color=FF570A&label=CodeRabbit+Reviews)](https://coderabbit.ai)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<br>
[![Flutter](https://img.shields.io/badge/Flutter-%E2%89%A53.38-02569B?logo=flutter&logoColor=white)](https://flutter.dev)
[![Dart](https://img.shields.io/badge/Dart-%E2%89%A53.10-0175C2?logo=dart&logoColor=white)](https://dart.dev)
[![Platforms](https://img.shields.io/badge/platforms-iOS%20%7C%20Android%20%7C%20Web-lightgrey)](#platform-setup)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![style: flutter_lints](https://img.shields.io/badge/style-flutter__lints-40c4ff.svg)](https://pub.dev/packages/flutter_lints)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Dinah--0-yellow)](https://huggingface.co/Lukitaduarte/dinah-0)
[![Live demos](https://img.shields.io/badge/demos-try%20it%20in%20your%20browser-7C5CFF)](https://lukitaduarte.github.io/decision_ai/)

<p align="center">
  <img src="https://raw.githubusercontent.com/Lukitaduarte/decision_ai/main/.github/images/iphone-demo.gif" width="300" alt="decision_ai on an iPhone XR: Dinah-0 and SmolLM2 answering typed questions on the device">
</p>

Ask a model **typed questions (jev-style)** about some data and get **typed answers with probabilities**, in Flutter.
The model can run **on the device** (bundled with the app, or downloaded once from Hugging Face), **in the browser**,
or **behind an API**; your code is the same either way.

```dart
final ai = await DecisionAI.huggingFace('Lukitaduarte/dinah-0');

final answers = await ai.decide(
  state: 'The package arrived broken and I need it for tomorrow.',
  questions: {
    'intent': const Choice({
      'refund': 'The customer wants their money back',
      'replacement': 'The customer wants a new unit sent',
      'complaint': 'The customer only wants to complain',
    }, instructions: 'What does the customer want?'),
    'urgent': const Noul('The customer needs it soon.'),
    'mood': const Score(['angry', 'upset', 'neutral', 'happy'], instructions: 'How does the customer feel?'),
  },
);

answers['intent']!.choice;        // 'replacement'
answers['intent']!.probabilities; // {refund: 0.002, replacement: 0.971, complaint: 0.027}
answers['urgent']!.noul;          // probability that the statement is true
answers['mood']!.score;           // expected level, 0 (angry) to 3 (happy)
```

No text is generated and nothing is parsed: every answer comes from one forward pass that scores the options.

**Try it in your browser:** [play chess against Dinah](https://lukitaduarte.github.io/decision_ai/chess/), a
150M-parameter decision model ([source](demo/chess)), or [ask open models your own questions](https://lukitaduarte.github.io/decision_ai/playground/)
([source](example)).

**Read more:** [decision_ai, the guide and the story behind it](https://lukita.me/posts/decision-ai-en/), on my blog
(also [in Portuguese](https://lukita.me/posts/decision-ai/)).

## Contents

- [Install](#install)
- [The three question types](#the-three-question-types)
- [Where the model runs](#where-the-model-runs)
- [Supported models](#supported-models)
- [Use any Hugging Face LLM](#use-any-hugging-face-llm)
- [Laya](#laya)
- [Bring your own model family](#bring-your-own-model-family)
- [Platform setup](#platform-setup)
- [For coding assistants](#for-coding-assistants)
- [How correctness is checked](#how-correctness-is-checked)

## Install

```yaml
dependencies:
  decision_ai: ^1.0.0
```

Then follow [Platform setup](#platform-setup) for iOS, Android and the web.

## The three question types

A request has a **state** (text or any JSON value: the data the questions are about) and named **questions**.

| type | you give | you get |
|---|---|---|
| `Choice` | labels, each with an optional description | `choice` (the most likely label), `probabilities` per label, `confidence` |
| `Noul` | a statement (optionally what "true" and "false" look like) | `noul`: the probability that the statement is true |
| `Score` | ordered levels, lowest first | `score`: the expected level (0 to n-1), `probabilities` per level |

Models with an "insufficient evidence" option also return `abstention`, the probability that the state does not say
enough; the other probabilities are then conditional on it being enough. A request that does not fit the model's
context throws `RequestTooLong`: nothing is ever cut.

## Where the model runs

| call | the model is | good for |
|---|---|---|
| `DecisionAI.local()` | bundled in the app as assets | offline from the first launch |
| `DecisionAI.huggingFace('org/model')` | downloaded on first use, SHA-256 checked, cached | small app downloads; pin `revision:` for reproducible builds |
| `DecisionAI.remote(ModelSource.url('https://...'))` | the same, from any HTTP folder (your CDN) | private hosting |
| the same calls, on the web | opened by URL and cached by the browser | demos, no install |
| `DecisionAI.api(endpoint:, apiKey:, model:)` | behind any Decision API provider | big models |
| `DecisionAI.openRouter(apiKey:, model:)` | behind OpenRouter's Decisions API | big models, one key |

After the first download, `huggingFace` and `remote` work offline. `onProgress` reports download progress. Every
engine has `close()`.

## Supported models

Each row is checked end to end: the app's probabilities match the Python reference on 64 requests, on iOS and on
Android. In the browser, Dinah-0 matches on 63 of 64 (see [web](#platform-setup) for the one difference). Dinah-0
ships its `decision_ai.json`, so it loads in one line; for the others you write a manifest (or let the tools write
it) for the original model's files or for your own export.

| model | kind | reader | on device | license | how to load |
|---|---|---|---|---|---|
| [Dinah-0](https://huggingface.co/Lukitaduarte/dinah-0) | encoder, 150M | `option-reader` | 165 MB (8-bit) | CC BY-NC 4.0 | `DecisionAI.huggingFace('Lukitaduarte/dinah-0')` |
| [Laya](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large encoder, 421M | `laya` | 490 MB (8-bit) | Apache-2.0 | port with [`tool/port_laya.py`](#laya) |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | causal LM, 135M | `label-logits` | 182 MB (the repo's own 4-bit) or 181 MB (8-bit export) | Apache-2.0 | a [manifest](#use-any-hugging-face-llm) for its `onnx/` files, or an export |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | causal LM, 494M | `label-logits` | 694 MB (8-bit) | Apache-2.0 | export with the [script](#use-any-hugging-face-llm) (the guide uses it) |
| [Verdict](https://huggingface.co/heman10x/rlcd-modernbert-151m) | GLiClass encoder, 151M | custom (`example/lib/gliclass_reader.dart`) | 606 MB (fp32) | Apache-2.0 | [Bring your own model family](#bring-your-own-model-family) |

Median time per question on the iOS simulator and Android emulator (a Mac): Verdict about 0.12 s, Dinah-0 and
SmolLM2 about 0.2 to 0.26 s, Laya about 0.7 s, Qwen2.5-0.5B about 1 s. On a real iPhone XR (2018), Dinah-0 takes
0.93 s and matches Python on 64 of 64. In the browser (WebAssembly, one thread) Dinah-0 takes about 0.9 s.

Beyond this list:

- **causal LMs** (instruction-tuned, byte-level BPE tokenizer: SmolLM, Qwen, GPT-2 style, Llama 3 style) work through
  `label-logits`, from the ONNX files their repository ships or from an export (see
  [Use any Hugging Face LLM](#use-any-hugging-face-llm)). How well they answer depends on the model; small ones are
  weak deciders.
- **other encoders** need a reader that matches how they were trained (see
  [Bring your own model family](#bring-your-own-model-family)).
- **SentencePiece tokenizers** (T5, Gemma, XLM-R, Llama 2) are not built in: pass your own `Tokenizer`.

## Use any Hugging Face LLM

Many model repositories already ship ONNX files (an `onnx/` folder, as SmolLM2 does). Then nothing needs exporting:
write a manifest for the file you want and load it.

```bash
python tool/export_causal_lm.py manifest --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --onnx onnx/model_q4.onnx --template-file template.txt --out smollm2
```

The command reads the model's `config.json` and writes `decision_ai.json` (the reader feeds the graph's empty
attention cache and reads the last position of its logits). Upload the folder, or pass the same manifest in code with
`ModelManifest.custom(...)` next to `DecisionAI.huggingFace('HuggingFaceTB/SmolLM2-135M-Instruct')`.

What you give up, measured on SmolLM2-135M-Instruct over the same 64 requests:

| | size | answers equal to fp32 | median per question (iOS simulator / Android emulator) |
|---|---|---|---|
| the repository's `onnx/model_q4.onnx`, no export | 182 MB | 45 of 64 | 611 / 1,101 ms |
| exported with the script below | 181 MB | 64 of 64 | 237 / 259 ms |

Both run exactly as the Python reference does (64 of 64 on the device). The export is faster because it returns only
the last position's logits, and closer to the original because its 8-bit weights keep more precision than a 4-bit
file. Published int8 files (`model_int8`, `model_quantized`) use dynamic quantization, whose kernels differ between
Android and iOS; prefer fp32, 4-bit or your own export.

### Export it yourself

[`tool/export_causal_lm.py`](tool/export_causal_lm.py) `export`:

1. exports the model to ONNX, returning only the logits of the last position (small output, fast on a phone);
2. stores the weights in 8 bits and the embedding table in int8 (about a quarter of the fp32 size);
3. writes `decision_ai.json` with your prompt template and copies `tokenizer.json`.

**1. Install the Python tools**

```bash
pip install torch transformers onnx onnxruntime huggingface_hub
```

**2. Write the prompt template** in the model's chat format. Three placeholders are filled per question:
`{instructions}` (the question), `{state}` (the data) and `{options}` (one line per option: `A. ...`, `B. ...`).
End it where the model's answer starts, so the next token is the option letter. For Qwen2.5 (`template.txt`):

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
{instructions}

Context:
{state}

Options:
{options}

Answer with the letter of the best option only.<|im_end|>
<|im_start|>assistant
```

**3. Export**

```bash
python tool/export_causal_lm.py export \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --template-file template.txt \
  --out my_model
```

`my_model/` now holds `decision_ai.json`, `tokenizer.json` and `onnx/model_8bit.onnx`.

**4. (Optional) Check the export** against ONNX Runtime 1.23, the version the app runs:

```bash
pip install onnxruntime==1.23.0 tokenizers numpy
python tool/export_causal_lm.py golden --dir my_model \
  --requests example/assets/golden_decisions.json --out my_model/golden_llm.json
```

**5. Use it**, either uploaded to Hugging Face:

```bash
hf upload your-name/my-model my_model --exclude "golden_llm.json"
```

```dart
final ai = await DecisionAI.huggingFace('your-name/my-model');
```

or bundled in the app: copy `my_model/` to `assets/my_model/`, list its files under `flutter: assets:` in
`pubspec.yaml`, and:

```dart
final ai = await DecisionAI.local(manifestPath: 'assets/my_model/decision_ai.json');
```

The `reader_config` in `decision_ai.json` can be edited by hand: `keys` (default `A` to `Z`), `option_line`
(default `{key}. {text}`), `key_prefix` (`" "` for tokenizers that glue the space to the letter), and `by_type` to
change any of them for one question type (for example `"noul": {"keys": ["No", "Yes"]}`).

## Laya

[Laya](https://github.com/NandhaKishorM/laya) (Convai Innovations, Apache-2.0) ships PyTorch weights and its own
Python runtime. The built-in `laya` reader follows Laya's code (`laya.common.build_sequence` and its temperatures per
question type and option count): on the 64 test requests it builds the same token ids as Laya itself.
[`tool/port_laya.py`](tool/port_laya.py) exports a checkpoint with Laya's own model code and writes `decision_ai.json`:

```bash
pip install "laya @ git+https://github.com/NandhaKishorM/laya" torch onnx onnxscript onnxruntime
python tool/port_laya.py export --model convaiinnovations/laya --out laya
```

```dart
final ai = await DecisionAI.local(manifestPath: 'assets/laya/decision_ai.json'); // or huggingFace / remote
```

Laya cuts a state that does not fit its context; this package raises `RequestTooLong` instead. The 8-bit export
matches Laya's PyTorch model on 62 of 64 answers; the two that differ are noul questions at about 50%.

## Bring your own model family

An on-device engine is four pieces, and each can be replaced:

| piece | built in | replace it when |
|---|---|---|
| `Runtime` | `OnnxRuntimeBackend` (ONNX Runtime Web on the web) | you run models with another engine (TFLite, Core ML, llama.cpp) |
| `Tokenizer` | `BpeTokenizer` (byte-level BPE from `tokenizer.json`) | the model uses SentencePiece or WordPiece |
| `Reader` | `option-reader`, `laya`, `label-logits` | the model reads questions in its own format |
| `Calibrator` | `TemperatureCalibrator` | you calibrate probabilities differently |

The **reader** is what differs between model families. It has two jobs: turn a question into input tensors
(`encode`) and turn output tensors into one logit per option (`decode`). Softmax, calibration, abstention and the
answer are handled by the library.

```dart
class MyReader implements Reader {
  MyReader(this.manifest, this.tokenizer);
  final ModelManifest manifest;
  final Tokenizer tokenizer;

  @override
  ReaderInput encode(Object? state, Question question) {
    final ids = tokenizer.encode('${render(question.instructions)}\n${render(state)}');
    return ReaderInput(optionCount: 2, tensors: {'input_ids': Tensor.int64(ids, [1, ids.length])});
  }

  @override
  Readout decode(ReaderInput input, Map<String, Tensor> outputs) =>
      Readout(outputs['logits']!.doubles.take(input.optionCount).toList());
}

DecisionAI.registerReader('my-reader', MyReader.new);

// A repo without decision_ai.json: describe it in code.
final ai = await DecisionAI.huggingFace(
  'org/model',
  manifest: ModelManifest.custom(
    reader: 'my-reader',
    files: {'model': 'model.onnx', 'tokenizer': 'tokenizer.json'},
    maxLength: 512,
  ),
);
```

A complete example is [`example/lib/gliclass_reader.dart`](example/lib/gliclass_reader.dart), which runs Verdict
with its abstention option and per-option-count temperatures. To skip manifests entirely, build the pieces yourself:
`DecisionAI.custom(runtime: ..., reader: ..., calibrator: ...)`.

## Platform setup

**iOS** (16 or newer): in `ios/Podfile`, use static frameworks:

```ruby
platform :ios, '16.0'
use_frameworks! :linkage => :static
```

**Android**: add the internet permission to `android/app/src/main/AndroidManifest.xml` if the app downloads models or
calls an API (Flutter only adds it to debug builds):

```xml
<uses-permission android:name="android.permission.INTERNET"/>
```

and keep ONNX Runtime's classes in release builds, in `android/app/proguard-rules.pro`:

```text
-keep class ai.onnxruntime.** { *; }
```

**Web**: the package runs models with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.23 in a
worker, so the page never freezes. Copy `ort.wasm.min.js`, `ort-wasm-simd-threaded.mjs` and
`ort-wasm-simd-threaded.wasm` from the [`onnxruntime-web@1.23.0`](https://www.npmjs.com/package/onnxruntime-web)
package into `web/ort/` and load it before Flutter in `web/index.html`:

```html
<script src="ort/ort.wasm.min.js"></script>
<script src="flutter_bootstrap.js" async></script>
```

On the web, `huggingFace` and `remote` open the model by URL and the browser caches it; pin `revision:` to a commit,
since SHA-256 checks and private repositories are not available there. One difference comes from JavaScript itself:
a whole-number float such as `532.0` is the same value as `532` in the browser, so it reaches the model as `532`
(Python writes `532.0`). Send such values as strings if the model was trained on them.

Serve everything from your own site (build with `--no-web-resources-cdn`, as [the demo](demo/chess) does): content
blockers such as Brave Shields can remove scripts and fonts loaded from CDNs and leave a blank page.

Models run on ONNX Runtime 1.23 through [`flutter_onnxruntime`](https://pub.dev/packages/flutter_onnxruntime).

## For coding assistants

[`skills/port-model-to-decision-ai`](skills/port-model-to-decision-ai/SKILL.md) is an
[Agent Skill](https://agentskills.io) that teaches coding assistants to port a Hugging Face model and set up an app:
choosing the reader, exporting and quantizing, writing `decision_ai.json` and proving the port against the model's own
code. Claude Code picks it up inside this repository; elsewhere, copy the folder to `~/.claude/skills/` (or your
assistant's skills folder). [`AGENTS.md`](AGENTS.md) describes the repository for agents that work on the package
itself.

## How correctness is checked

- `flutter test`: the tokenizer matches Hugging Face `tokenizers` on 8,612 texts across three tokenizer families,
  JSON rendering matches Python, and readers, calibration, downloads and the API client are covered with fakes.
- `example/integration_test/parity_test.dart`: on a simulator or phone, each supported model answers the same 64
  requests as the Python reference on ONNX Runtime 1.23. Instructions for serving the models are at the top of the
  file.
- [`example/tool/web_parity.dart`](example/tool/web_parity.dart): the same in a browser, for the playground's models
  loaded from Hugging Face: Dinah-0 and SmolLM2 63 of 64, Laya's community 4-bit port 62 of 64 (a whole-number float,
  see [web](#platform-setup), and 4-bit kernels account for the differences).

Tools: [`tool/export_causal_lm.py`](tool/export_causal_lm.py) (LLM export and references),
[`tool/port_laya.py`](tool/port_laya.py) (Laya export and references),
[`tool/make_golden.py`](tool/make_golden.py) (tokenizer and rendering references),
[`tool/make_decision_golden.py`](tool/make_decision_golden.py) (Dinah references),
[`tool/mock_decision_server.py`](tool/mock_decision_server.py) (a local Decision API provider),
[`example/tool/gliclass_golden.py`](example/tool/gliclass_golden.py) (Verdict references).

## License

[MIT](LICENSE). Models keep their own licenses (see [Supported models](#supported-models)); the data, fonts and
runtime files the repository redistributes are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
