# Serving and training decision

Checked 2026-09-22. USD throughout. These are public-rate calculations, not vendor quotes or measured benchmarks.

## Recommended route

For the newly specified low-traffic **public website**, see the [expanded provider comparison and product plan](provider-landscape.md). Prefer precomputed, held-out historical examples and shared cached results. Shortlist Together for managed training and DeepInfra/Novita for managed scale-to-zero hosting; Modal provides more runtime control. Exact checkpoint compatibility remains untested.

For **internal research/paper evaluation**, Tinker training plus checkpoint sampling remains an option. Its beta API's stated internal/testing scope is why it is not the first choice for the public app.

Runpod remains a programmable GPU alternative. Exported checkpoints need verification of architecture, adapter conversion, chat template, numerical behavior, context capacity, and latency wherever they are hosted.

## Tinker

[Official model pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/) lists `Qwen/Qwen3.8-27B` at default 64K context:

| Operation | USD / million tokens |
| --- | ---: |
| Uncached input | 1.86 |
| Cached input | 0.372 |
| Generated output | 5.595 |
| Training | 4.103 |

The 256K `:peft:262144` variant has different prices. Budget calculations here use 64K, no cache discount, and **all generated tokens including thinking**.

Saved sampler checkpoints can be called through the native sampler or [OpenAI-compatible API](https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/). That API is beta, intended for testing and low internal traffic, with variable latency/throughput. The separately advertised serverless beta lists Inkling models only; it does not establish production Qwen serving support.

[Exporting a PEFT adapter](https://tinker-docs.thinkingmachines.ai/tutorials/deployment/lora-adapter/) is documented. **Importing a Together-trained adapter into Tinker is not verified.** Do not assume that training on Together followed by Tinker sampling works. Export availability also does not prove prediction parity in a different runtime.

## Per-request budget

Assumptions: one call per interval, one decision stream, 30 continuous days, 2,000 input tokens and 200 generated tokens per call, no retries or cache savings.

`cost_per_call = (2000 × 1.86 + 200 × 5.595) / 1,000,000 = $0.004839`

| Frequency | Calls / 30 days | Tinker sampling |
| --- | ---: | ---: |
| Hourly | 720 | $3.48 |
| Every five minutes | 8,640 | $41.81 |
| Every minute | 43,200 | $209.04 |

Separate per-instrument calls multiply cost. A combined multi-instrument prompt changes token counts instead. Longer thinking, retries, and validation calls add cost. These estimates do not establish that any frequency yields useful trading decisions.

## Dedicated and serverless GPU comparison

One GPU is assumed below only for cost illustration. Exact fit, context capacity, supported deployment profile, concurrency, and latency are unverified.

| Provider / illustrative GPU | USD / hour | 720-hour equivalent |
| --- | ---: | ---: |
| Together H100, regular rate | 5.49 | 3,952.80 |
| Fireworks H100 or H200 | 8.00 | 5,760.00 |
| Runpod A100 80GB Pod, listed Secure Cloud rate | 1.59 | 1,144.80 |
| Runpod A100 serverless flex, worker-running equivalent | 2.72 | 1,958.40 |

[Together pricing](https://www.together.ai/pricing) also lists an H100 promotion at $3.99/hour through 2026-09-30; do not use it as a lasting monthly rate. [Fine-tuned endpoint billing](https://support.together.ai/articles/7922061943-fine-tuning-pricing) continues while the endpoint runs. The [current dedicated autoscaling description](https://www.together.ai/blog/autoscaling-endpoints-for-llm-inference) distinguishes explicit stop (`min=max=0`) from automatic wake-on-request: stopped endpoints require an explicit restart.

[Fireworks prices](https://fireworks.ai/pricing) are per GPU; its [LoRA deployment documentation](https://docs.fireworks.ai/fine-tuning/deploying-loras) requires on-demand dedicated deployment, not shared serverless LoRA hosting. However, [dedicated deployments DO support scale-to-zero](https://docs.fireworks.ai/deployments/autoscaling). Therefore $5,760 is a 720-hour illustration, **not a mandatory monthly minimum**. The default idle timeout is one hour and can be reduced to five minutes. Cold-start requests return 503 rather than being queued; the client must retry.

[Runpod listed rates](https://www.runpod.io/articles/guides/ai-server-cost) vary by cloud tier and availability. [Flex billing](https://www.runpod.io/product/serverless) covers worker start through full stop, not just token generation. At 30 **total billable worker hours**, the A100 flex example is $81.60, plus storage and other charges. Frequent requests, model loading, and idle timeout can keep a worker billable; scale-to-zero does not imply negligible latency or cost.

27B bf16 weights alone are roughly 54GB before runtime overhead. An 80GB GPU is a sizing candidate for bounded context/concurrency, not a guarantee. A 48GB deployment would require a compatible reduced-precision approach or offload; validate output changes after conversion.

## Training comparison

10,000 single-turn examples × 1,000 input-plus-output tokens × three epochs = 30 million processed training tokens.

| Provider / path | USD / million tokens | Illustrative training bill |
| --- | ---: | ---: |
| Together Qwen3.8-27B LoRA SFT | 1.05 | 31.50 |
| Fireworks managed LoRA SFT, 16.1–80B tier | 3.00 | 90.00 |
| Tinker Qwen3.8-27B, 64K training | 4.103 | 123.09 |

Rates are from the provider pages linked above. Fireworks' tier price is not confirmation of exact model/account availability. Validation, repeated forward passes, format-dependent token expansion, and any storage/minimum charges are additional. Together's job minimum for this model is $4. Tokenize actual examples and inspect a dry-run estimate before spending.
