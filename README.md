# Jev trading research

Private research starter for learning trading actions from historical executions using **Qwen/Qwen3.8-27B**. Current status: planning and cost analysis only; no dataset imported, model trained, endpoint deployed, or trading integration implemented.

## Proposed first experiment

Use **Thinking Machines Lab Tinker for LoRA training and low-volume checkpoint sampling** if its beta access and latency fit the experiment. This avoids an always-running GPU during paper evaluation. Together remains the cheaper managed supervised-training option, especially if the checkpoint will be exported for separate hosting.

This is a recommendation, not a purchased service or measured performance result. A larger model is a candidate to test, not evidence of a trading advantage.

- [Serving options and budget](docs/serving.md)
- [Dataset and evaluation plan](docs/experiment.md)
- [Dated price assumptions](config/cost-assumptions.json)

## Pilot budget

As of 2026-09-22, an illustrative Tinker run with 10,000 examples, 1,000 tokens each, and three epochs costs **$123.09** for training. Sampling every five minutes for 30 days, with 2,000 uncached input tokens and 200 total generated tokens per call, costs **$41.81**. Combined: **$164.90**, excluding additional validation, retries, storage, taxes, and data preparation. Reserve **$200–300** for a first pilot; this does not authorize spend.

Prices and limitations are sourced in the serving document. Actual token counts, latency, and account availability remain unmeasured.

## Next work

1. Obtain a small lawful-to-use trade export and its column definitions; confirm venue, timezone, instruments, and contract units.
2. Reconstruct positions and pre-decision market observations before constructing labels.
3. Freeze time-based evaluation periods and compare the untuned model, tuned model, and a numerical baseline.
4. Benchmark sampling latency, complete-token cost, and export compatibility before choosing production hosting.

Raw trade records, account identifiers, credentials, model weights, and generated runtime artifacts stay out of Git. The source screenshot's dataset size and performance claims have not been independently verified. “Jev” is the user's intended agent name; its implementation and tool interface are not yet specified.
