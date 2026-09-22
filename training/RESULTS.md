# Synthetic rehearsal — 2026-09-22

Qwen3.5-4B was actually fine-tuned on a Modal H100 with Unsloth BF16 LoRA, exported and reloaded. All 496 exported adapter tensors matched byte-for-byte, and all 12 evaluation cases produced identical logits after the loader was corrected. This used synthetic rule-following examples, **not 워뇨띠 records**. Production still serves the base checkpoint.

## Measurements

| Measurement | Observed value |
|---|---:|
| GPU | NVIDIA H100 80GB HBM3 |
| LoRA | Rank 16; 32,464,896 trainable parameters |
| Training examples / separate evaluation examples | 192 / 12 |
| Average input length | 1,871 tokens |
| Optimizer steps / effective batch size | 100 / 4 |
| Training time, including first-step compilation | 192.94 s (3m13s) |
| First optimizer step | 116.44 s |
| Median optimizer step after five warm-up steps | 0.763 s |
| Input throughput after five warm-up steps | 9,711 tokens/s |
| Peak PyTorch allocated GPU memory during training | 10.03 GiB |
| Training process through save and initial reload check | 315.66 s (5m16s) |
| Separate corrected export/reload verification | 117.28 s (1m57s) |
| Exported adapter weights | 129,934,448 bytes (~124 MiB) |
| Reloaded tensor bytes / choices | Exact match / 12 of 12 identical |
| Maximum post-reload logit difference | 0.0 |

Synthetic-rule accuracy changed from 7/12 to 12/12. The rule is explicitly present in the prompt; this is a plumbing check, not evidence of trading quality. Throughput counts processed input tokens; only one answer token per example contributes to the supervised loss. Memory excludes reserved allocator memory and other non-PyTorch GPU allocations.

For similarly sized inputs, 10,000 examples for one epoch would take roughly 32 minutes of steady training plus loading, compilation and evaluation. A planning range of **35–45 minutes and roughly $2.3–$3.0 in H100 charges** is an extrapolation, not another measured run. Different context lengths, batch sizes and training settings can change this substantially.

## Cost and corrected failures

Both training attempts completed 100 steps. The first reused an unloaded model and failed output parity. The second saved exact weights, but its reload disabled PEFT's adapter autocasting. On this BF16 base, that caused FP32 adapter values to round through BF16 when loaded; later promotion back to FP32 could not recover them. The final loader uses a fresh revision-pinned base, PEFT's pre-load FP32 promotion and Unsloth post-patching. The saved second adapter then passed tensor and output parity in a separate verification-only run. The original 0.15 logit tolerance was not relaxed; the measured difference was zero.

Estimated GPU charges across **both training attempts and the verification** were approximately **$0.90** at $0.001097/second. This excludes container startup, CPU, RAM, storage and CPU preparation; it is not an invoice or an audited remaining credit balance. All rehearsal apps were confirmed stopped with zero tasks after completion. Total observed GPU process time was about 13.5 minutes, within the original 30-minute allowance.

An independent read-only review identified infrastructure rescheduling as a hole in a per-attempt timeout; the immutable caller deadline and explicit cancellation address it. The reviewer also checked the pinned PEFT/Unsloth load order and the final precision-preserving loader. Four local data/encoding tests passed; the GPU runs verified the real tokenizer, finite gradients, changed LoRA weights, export and reload.

## Evidence and artifacts

- [Recorded measurements and verification output](results/2026-09-22.json)
- [Measured training attempt](https://modal.com/apps/storminggalaxys4/main/ap-g2JBwDRVrHkc1lu0ZTkFDF): `db8832ef9ba14411841a3537ee801c38`
- [Successful verification-only run](https://modal.com/apps/storminggalaxys4/main/ap-44G9LRDDVr6BvNVG3OR2iK): `c22eab5c05d14037915c42db396ef25d`
- [First attempt](https://modal.com/apps/storminggalaxys4/main/ap-Oy2fPOr7FOVTL5iEZv9EOu): `4eaff443064745bea8d6bdaae1b7dfb3`
- Private Modal Volume: `jev-qwen-training-rehearsals`, adapter under `db8832ef9ba14411841a3537ee801c38/adapter/`.
- Local export: `.runtime/training/db8832ef9ba14411841a3537ee801c38/adapter/` (ignored by Git).
- Adapter SHA-256: `4c29fe9c9e7f9d3e7ed62144b0396389ff349c3dae4e7d68e7715506ff62ab1a`.

The two-step training/verification sequence passed. A further full training run after the loader correction was unnecessary and was not performed. Live inference compatibility with the separate production Transformers runtime remains untested; this synthetic adapter was not promoted. Actual trader training still needs verified historical records, decision-time market context and chronological evaluation.

Rate source: [Modal pricing](https://modal.com/pricing). Reproduction: [training entrypoint and limits](README.md).
