# Jev trading research

Private research starter for learning trading actions from historical executions using **Qwen/Qwen3.8-27B**. Current status: planning and cost analysis only; no dataset imported, model trained, endpoint deployed, or trading integration implemented.

## Current recommendation: a small public website

Start with a historical-chart decision game and shareable results, with model outputs computed in advance on held-out cases. This keeps page views independent of inference calls. The public-site requirement supersedes the earlier internal-only Tinker recommendation.

For **Qwen3.8-27B**, shortlist **Together managed SFT → exported checkpoint → DeepInfra or Novita managed hosting**, with **Modal** as the more programmable alternative. All require an exact-model/export compatibility check before purchase. DeepInfra lists A100 80GB at $0.89/hour; Novita lists H100 80GB at $1.99/hour; both document scale-to-zero. Tinker remains an internal training/evaluation option, not the proposed public website backend.

If changing the model is acceptable, evaluate **W&B Training's supported Qwen variants and token-priced trained-model inference**. Its published Qwen3-14B rate is inexpensive, but that does not establish Qwen3.8-27B fine-tuning support.

This is a recommendation, not a purchased service or measured performance result. A larger model is a candidate to test, not evidence of a trading advantage.

- [Serving options and budget](docs/serving.md)
- [전체 업체 비교·추천·바이럴 사이트 운영안 (한국어)](docs/provider-landscape.md)
- [Dataset and evaluation plan](docs/experiment.md)
- [Dated price assumptions](config/cost-assumptions.json)

## Pilot budget

As of 2026-09-22, 10,000 examples × 1,000 tokens × three epochs gives an illustrative Together LoRA training bill of **$31.50**. Fifty billable GPU-hours would add **$44.50 on DeepInfra A100 80GB**, **$99.50 on Novita H100**, or **$124.92 on Modal A100 80GB** before extras. These are equal-hour scenarios, not equivalent performance or a traffic forecast. A first-pilot reserve of **$100–250** is a planning choice, not an authorization or guaranteed bill.

Prices and limitations are sourced in the serving document. Actual token counts, latency, and account availability remain unmeasured.

## Next work

1. Obtain a small lawful-to-use trade export and its column definitions; confirm venue, timezone, instruments, and contract units.
2. Reconstruct positions and pre-decision market observations before constructing labels.
3. Freeze time-based evaluation periods and compare the untuned model, tuned model, and a numerical baseline.
4. Benchmark sampling latency, complete-token cost, and export compatibility before choosing production hosting.

Raw trade records, account identifiers, credentials, model weights, and generated runtime artifacts stay out of Git. The source screenshot's dataset size and performance claims have not been independently verified. “Jev” is the user's intended agent name; its implementation and tool interface are not yet specified.
