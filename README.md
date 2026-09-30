# decision_ai

[![CI](https://github.com/Lukitaduarte/decision_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Lukitaduarte/decision_ai/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Lukitaduarte/decision_ai/actions/workflows/codeql.yml/badge.svg)](https://github.com/Lukitaduarte/decision_ai/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/Lukitaduarte/decision_ai/graph/badge.svg)](https://codecov.io/gh/Lukitaduarte/decision_ai)
[![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Lukitaduarte/decision_ai?labelColor=171717&color=FF570A&label=CodeRabbit+Reviews)](https://coderabbit.ai)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<br>
[![Flutter](https://img.shields.io/badge/Flutter-%E2%89%A53.38-02569B?logo=flutter&logoColor=white)](https://flutter.dev)
[![Dart](https://img.shields.io/badge/Dart-%E2%89%A53.10-0175C2?logo=dart&logoColor=white)](https://dart.dev)
[![Platforms](https://img.shields.io/badge/platforms-iOS%20%7C%20Android-lightgrey)](#platform-setup)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![style: flutter_lints](https://img.shields.io/badge/style-flutter__lints-40c4ff.svg)](https://pub.dev/packages/flutter_lints)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Dinah--0-yellow)](https://huggingface.co/Lukitaduarte/dinah-0)

Ask a model **typed questions** about some data and get **typed answers with probabilities**, in Flutter. The model
can run **on the device** (bundled with the app or downloaded once from Hugging Face) or **behind an API**; your
code is the same either way.

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

## Contents

- [Install](#install)
- [The three question types](#the-three-question-types)
- [Where the model runs](#where-the-model-runs)
- [Supported models](#supported-models)
- [Use any Hugging Face LLM](#use-any-hugging-face-llm)
- [Bring your own model family](#bring-your-own-model-family)
- [Platform setup](#platform-setup)
- [How correctness is checked](#how-correctness-is-checked)

## Install

The package is not on pub.dev yet; depend on the repository:

```yaml
dependencies:
  decision_ai:
    git: https://github.com/Lukitaduarte/decision_ai
```

Then follow [Platform setup](#platform-setup) for iOS and Android.

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
| `DecisionAI.api(endpoint:, apiKey:, model:)` | behind any Decision API provider | big models |
| `DecisionAI.openRouter(apiKey:, model:)` | behind OpenRouter's Decisions API | big models, one key |

After the first download, `huggingFace` and `remote` work offline. `onProgress` reports download progress. Every
engine has `close()`.

## Supported models

Each row is checked end to end: the app's probabilities match the Python reference on 64 requests, on iOS and on
Android.

| model | kind | reader | on device | license | how to load |
|---|---|---|---|---|---|
| [Dinah-0](https://huggingface.co/Lukitaduarte/dinah-0) | encoder, 150M | `option-reader` | 165 MB (8-bit) | CC BY-NC 4.0 | `DecisionAI.huggingFace('Lukitaduarte/dinah-0')` |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | causal LM, 135M | `label-logits` | 181 MB (8-bit) | Apache-2.0 | export with the [script](#use-any-hugging-face-llm) |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | causal LM, 494M | `label-logits` | 694 MB (8-bit) | Apache-2.0 | export with the [script](#use-any-hugging-face-llm) (the guide uses it) |
| [Verdict](https://huggingface.co/heman10x/rlcd-modernbert-151m) | GLiClass encoder, 151M | custom (`example/lib/gliclass_reader.dart`) | 606 MB (fp32) | Apache-2.0 | [Bring your own model family](#bring-your-own-model-family) |

Median time per question on the iOS simulator and Android emulator (a Mac, not a phone): Verdict about 0.12 s,
Dinah-0 and SmolLM2 about 0.2 to 0.26 s, Qwen2.5-0.5B about 1 s.

Beyond this list:

- **causal LMs** (instruction-tuned, byte-level BPE tokenizer: SmolLM, Qwen, GPT-2 style, Llama 3 style) work through
  `label-logits` after the export below. How well they answer depends on the model; small ones are weak deciders.
- **other encoders** need a reader that matches how they were trained (see
  [Bring your own model family](#bring-your-own-model-family)).
- **SentencePiece tokenizers** (T5, Gemma, XLM-R, Llama 2) are not built in: pass your own `Tokenizer`.

## Use any Hugging Face LLM

Causal language models are exported with [`tool/export_causal_lm.py`](tool/export_causal_lm.py). The script:

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

## Bring your own model family

An on-device engine is four pieces, and each can be replaced:

| piece | built in | replace it when |
|---|---|---|
| `Runtime` | `OnnxRuntimeBackend` | you run models with another engine (TFLite, Core ML, llama.cpp) |
| `Tokenizer` | `BpeTokenizer` (byte-level BPE from `tokenizer.json`) | the model uses SentencePiece or WordPiece |
| `Reader` | `option-reader`, `label-logits` | the model reads questions in its own format |
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

Models run on ONNX Runtime 1.23 through [`flutter_onnxruntime`](https://pub.dev/packages/flutter_onnxruntime).

## How correctness is checked

- `flutter test`: the tokenizer matches Hugging Face `tokenizers` on 8,612 texts across three tokenizer families,
  JSON rendering matches Python, and readers, calibration, downloads and the API client are covered with fakes.
- `example/integration_test/parity_test.dart`: on a simulator or phone, each supported model answers the same 64
  requests as the Python reference on ONNX Runtime 1.23. Instructions for serving the models are at the top of the
  file.

Tools: [`tool/export_causal_lm.py`](tool/export_causal_lm.py) (LLM export and references),
[`tool/make_golden.py`](tool/make_golden.py) (tokenizer and rendering references),
[`tool/make_decision_golden.py`](tool/make_decision_golden.py) (Dinah references),
[`tool/mock_decision_server.py`](tool/mock_decision_server.py) (a local Decision API provider),
[`example/tool/gliclass_golden.py`](example/tool/gliclass_golden.py) (Verdict references).

## License

[MIT](LICENSE). Models keep their own licenses (see [Supported models](#supported-models)).
