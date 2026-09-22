# Real-data pilot results — 2026-09-22

**Training and adapter export succeeded. The held-out results do not beat keeping the previous position, so this pilot does not justify production promotion.**

User-supplied BTC execution history was converted into 4,096 historical next-hour position-side examples, with 128 chronological validation and 128 test examples. This is exposure imitation, not a return forecast or a profitable trading policy. See the [data construction and limitations](REAL_DATA.md), [dataset manifest](REAL_DATA_MANIFEST.json), and [aggregate measured report](REAL_DATA_RESULTS.json).

| Accuracy | Validation (128) | Test (128) |
|---|---:|---:|
| Original Qwen3.5-4B | 94.53% | 96.88% |
| Trained LoRA | 98.44% | 99.22% |
| Previous-position persistence | 98.44% | 99.22% |
| Correct position transitions, trained LoRA | 0 / 2 | 0 / 1 |

Most hourly labels preserve the previous position. The tuned model's held-out performance equals this trivial baseline. There are too few transitions to establish transition-prediction quality, and no flat targets in either held-out split. Macro-F1 includes all three predefined classes, assigning zero to the absent flat class; full per-class support and confusion matrices are in the JSON report. Historical separation applies to this fine-tuning dataset, not unknown information in the base model's pretraining. No transaction-cost, execution or prospective profitability evaluation was performed.

## Measured run

- Model: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Dataset: `e4555b73ab5e4194`; run: `cd8911d3b400405a894c93ea437855cc`.
- One H100 80 GB; BF16 base with rank-16 FP32 LoRA; 32,464,896 trainable parameters.
- 1,024 optimizer steps, batch 2 × accumulation 2: one pass over 4,096 training examples.
- Training: **882.60 seconds (14.71 minutes)**, including first-step compilation. Full GPU child process including evaluation/reload: **1,063.29 seconds (17.72 minutes)**. CPU preparation: 55.19 seconds.
- Steady median step: 0.762 seconds. Training peak allocated VRAM: 10.09 GiB (PyTorch allocations, not total GPU memory).
- Finite losses/gradients; 496 adapter tensors changed. Mean loss over first/last 10 steps: 0.4433 / 0.1340. These training losses do not establish generalization.
- Adapter file: 129,934,448 bytes; SHA256 `918fdcd054e3d77116ddb7b708cc7c2a24aa443696f4639bd103408777051831`.
- Adapter tensors were byte-identical in memory, on disk and after loading a fresh pinned base. All 12 reload choices matched; maximum logit difference **0.0**.

Export checks apply to the training runtime, not the production inference image. The live service continues to report `trainingStatus: base`; this adapter has not been deployed.

## Cost and shutdown

Using [Modal's published rates](https://modal.com/pricing), GPU-only child runtime is about **$1.17**; including configured CPU/RAM and preparation gives about **$1.30**. These are estimates, not invoices, and exclude startup/shutdown, the initial failed CPU startup, image/storage overhead, and any usage above requested resources. The successful run's configured worst-case compute estimate was $2.23 within the approved **$5 pilot budget**; there were no automatic training retries.

The first attempt stopped during CPU preparation because the remote module lacked the local `training` package. The import/image wiring was corrected before GPU allocation. Modal confirmed both attempt apps stopped with zero tasks:

- [Completed pilot](https://modal.com/apps/storminggalaxys4/main/ap-qqG8lOosYZv5QiX0RaR9Hy): stopped at 18:36:43 KST.
- Initial CPU attempt `ap-9Thbqc4Tq95qWV2JismeF3`: stopped at 18:17:22 KST; no GPU dispatched.

The private adapter is stored in Modal Volume `jev-qwen-real-pilot-artifacts`, under `cd8911d3b400405a894c93ea437855cc/adapter/`, and downloaded locally under the ignored `.runtime/training/` directory. Raw CSVs, example rows and model weights are not committed to git.

Before spending on a larger run, redesign the task and evaluation around exposure transitions with enough held-out transition cases, while retaining a naturally sampled evaluation to measure false signals. A controlled next experiment can test prior-position ablation and event-focused training; do not repeatedly tune against this test set.
