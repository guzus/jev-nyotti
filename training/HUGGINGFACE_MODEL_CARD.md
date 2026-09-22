---
license: apache-2.0
library_name: peft
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
language:
- en
tags:
- lora
- unsloth
- qwen3_5
- finance
- research
model-index:
- name: jev-nyotti
  results: []
---
# jev-nyotti (jev뇨띠)

A **rank-16 LoRA adapter for Qwen3.5-4B**, trained on 4,096 sanitized examples derived from user-supplied historical BTC execution records attributed to 워뇨띠 (AOA). Trader identity and authenticity were not independently verified; this project is not affiliated with or endorsed by that trader.

This is an experimental **next-hour position-side imitation model**, not a full standalone checkpoint or a proven trading strategy. It predicts `long`, `short` or `flat` exposure from historical closed candles and the prior position side. It does not generate the trader's reasoning or execute orders.

[Source and training code](https://github.com/guzus/jev-nyotti) · [Website](https://jt.memtherscan.xyz/) · [Detailed evaluation](https://github.com/guzus/jev-nyotti/blob/main/training/REAL_DATA_RESULTS.md)

**For educational purposes only.** The website serves this adapter as an experimental next-hour position-side classifier, assuming a hypothetical prior flat position. Its live inputs use Kraken spot markets and multiple candle intervals; transfer beyond the hourly BitMEX BTC training distribution is unvalidated. This deployment does not establish trading value.

## Evaluation

| Accuracy | Validation (128) | Test (128) |
|---|---:|---:|
| Original Qwen3.5-4B | 94.53% | 96.88% |
| This adapter | 98.44% | 99.22% |
| Keep the previous position | 98.44% | 99.22% |
| Correct position transitions, adapter | 0 / 2 | 0 / 1 |

**It does not outperform keeping the previous position.** Most hourly targets preserve exposure, so high accuracy is not evidence of useful trading decisions. The validation/test splits have only 2/1 transitions and no flat targets. No profitability, transaction-cost or live trading evaluation was performed. The full action-space metrics and confusion matrices are in `evaluation.json`; macro-F1 includes the absent flat class with zero score.

## Training data and method

- BTC XBTUSD executions from 2018–2021 were sorted and accumulated into signed positions. All 3,961 BTC funding anchors matched reconstructed absolute exposure.
- Official BitMEX hourly candles supply market context. Inputs contain 24 closed candles, features using up to 96 hours, and the position side immediately before the cutoff. Targets use fills strictly before the next-hour boundary.
- Train: 4,096 examples from 2018–2020. Validation/test: 128 each from the first/second halves of 2021, with 96-hour purge gaps.
- Account/order/execution identifiers, wallet fields and arbitrary source text were excluded. Raw exports and individual training examples are not included in this release.
- Naive execution timestamps were interpreted as UTC, corroborated by three official historical funding time/rate matches. Original export timezone metadata is unavailable.
- Chronological isolation applies to fine-tuning data; unknown base-model pretraining may include historical market information.
- Frozen BF16 base; FP32 LoRA rank/alpha 16, dropout 0; 32,464,896 trainable parameters; 1,024 steps; batch 2 × accumulation 2; AdamW learning rate 1e-4. One H100, 14.71 minutes of training including compilation.
- Only the single classifier answer token is supervised. Option order is shuffled and token labels are mapped back to action names.

## Loading and classifier format

Required base: `Qwen/Qwen3.5-4B`, pinned revision **`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`**. These files contain the adapter and tokenizer, not the base weights.

The validated CUDA runtime uses Unsloth 2026.9.7, Unsloth Zoo 2026.9.6, Transformers 5.5.0, PEFT 0.18.1 and Torch 2.8.0. Use the exact dependencies in [`training/modal_real.py`](https://github.com/guzus/jev-nyotti/blob/main/training/modal_real.py), including the causal-conv1d wheel. The deployed native Transformers runtime and strict PEFT loader are documented in `inference/` in the source repository. Startup verifies the pinned checkpoint, every exported FP32 adapter tensor, and active LoRA layers. Compatibility with arbitrary runtimes is not implied.

From a clone of the source repository, in that CUDA environment:

```sh
git clone https://github.com/guzus/jev-nyotti.git
cd jev-nyotti
PYTHONPATH=inference:training python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from validation import load_exported_adapter

snapshot_download(
    'Qwen/Qwen3.5-4B',
    revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'],
)
adapter_path = snapshot_download('guzus/jev-nyotti')
model = load_exported_adapter(Path(adapter_path), max_length=4096)
print('Loaded next-hour exposure imitation adapter')
PY
```

Use [`training/real_data.py`](https://github.com/guzus/jev-nyotti/blob/main/training/real_data.py) for the exact feature schema and rubric, and [`inference/jev_inference/prompt.py`](https://github.com/guzus/jev-nyotti/blob/main/inference/jev_inference/prompt.py) for the classifier prompt with thinking disabled. Score only the verified single-token candidate labels as implemented in [`training/validation.py`](https://github.com/guzus/jev-nyotti/blob/main/training/validation.py). This is not a general chat model; `flat` means zero exposure. The website's trained decisions preserve this `flat` label; older shared base-model results retain their original `hold` label.

The adapter was reloaded onto a fresh pinned base with byte-identical FP32 tensors. All 12 verification choices matched, with maximum logit difference 0.0. `checksums.json` records the released file hashes.

## License and use

Released under Apache-2.0, matching the base model license included as `LICENSE`. This derivative adds LoRA fine-tuning weights; the original Qwen weights remain a separate dependency. This does not grant redistribution rights to the original trade export, which is not included.

Research use only. DYOR NFA. The released results do not establish predictive trading value, safety for autonomous trading, or profitability.
