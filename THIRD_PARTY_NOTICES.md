# Third-party notices

decision_ai is MIT-licensed ([LICENSE](LICENSE)). The repository also redistributes the following third-party
material, under its own terms.

## Data used in tests and parity references

Texts sampled from these datasets appear in `test/golden/`, `test/fixtures/` and
`example/assets/golden_decisions.json`. They are used unchanged (or, for typed-decisions-pt, translated) as inputs for tokenizer and model
parity tests.

| source | license | used for |
|---|---|---|
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 | tokenizer and decision references |
| typed-decisions-pt (a Portuguese translation of typed-decisions, made for Dinah-0) | Apache-2.0 | tokenizer references |
| [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Apache-2.0 | tokenizer references |
| [Lichess puzzle database](https://database.lichess.org/#puzzles) | CC0 | `demo/chess/test/dinah_positions.json` |

## Tokenizers used in tests

The `tokenizer.json` files in `test/fixtures/` and `example/assets/model/` belong to their models:

| model | license |
|---|---|
| [Dinah-0](https://huggingface.co/Lukitaduarte/dinah-0) (moBERTo / ModernBERT tokenizer) | CC BY-NC 4.0 (the model); Apache-2.0 (ModernBERT) |
| [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache-2.0 |
| [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 |
| [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) | Apache-2.0 |

## Web runtime and fonts

| component | license | where |
|---|---|---|
| [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) 1.23.0, Microsoft | MIT | `demo/chess/web/ort/`, `example/web/ort/` |
| [Roboto](https://fonts.google.com/specimen/Roboto) | SIL Open Font License 1.1 | `demo/chess/assets/fonts/`, `example/assets/fonts/` |
| [Noto Sans Symbols 2](https://fonts.google.com/noto/specimen/Noto+Sans+Symbols+2) | SIL Open Font License 1.1 | `demo/chess/assets/fonts/` |

## Artwork

The drawing of Dinah (the demos' avatar and icons) is by Shunnk. It is not covered by the MIT license.

## Models run by the examples

The examples download models at run time; they are not part of this repository. Each keeps its own license: Dinah-0
(CC BY-NC 4.0), SmolLM2 and Qwen2.5 (Apache-2.0), Laya and its community ONNX port (Apache-2.0), Verdict (Apache-2.0).
