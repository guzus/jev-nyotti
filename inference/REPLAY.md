# Bounded historical reconstruction

This is a separate offline Modal app, never a live serving invocation. The pinned
adapter predicts next-hour position. The explicit strategy holds that prediction
for four hours; it is not a four-hour-trained model. Source must accurately identify
spot/USDT transfers. This is educational reconstructed behavior, not actual orders.

Input JSON: `{source,from,to,series:[{symbol,candles:[{time,open,high,low,close,volume}]}]}`.
Times are candle-open UTC seconds. Range `[from,to)` uses aligned UTC four-hour
cutoffs and requires 96 contiguous preceding hourly bars and all four execution
bars per cutoff, for every symbol. Any gap fails rather than filling invented bars.
Initial position is flat, then each symbol carries its prior prediction forward.
Start a new run ID for a changed start/end/model/input. Never reverse chronology.

```
modal run inference/modal_replay.py --input-file /absolute/input.json --run-id benchmark-001 --budget-usd 0.75 --max-decisions 20
```

The conservative reservation rate is $0.0013/sec (H100, 4 CPUs, 32GiB), with 180s
reserved for startup/teardown. A $0.75 invocation hard-stops its Python process
after at most 396.9 function seconds; GPU calls/downloads cannot bypass that timer.
Function/platform timeouts also bound execution; retries are disabled. Each call
reserves its FULL stated budget, including resumed calls. The operator must deduct
all reserved attempts from the approved total; this is not a provider billing cap.
The shared model-cache volume may incur small storage costs independently.
No tensor batching is enabled: existing sequential verified engine is retained.

Completed full portfolio cutoffs are committed to `jev-nyotti-replay` after each
step. Partial interrupted cutoffs are not published and will be recomputed.
Use `--resume` with byte-equivalent parsed input to continue, explicitly reserving
a NEW budget. The local CLI fetches output, status, and checkpoint JSON alongside
the input when the function finishes normally. On watchdog interruption retrieve:
`modal volume get jev-nyotti-replay RUN_ID/output.json ./output.json`.
Output can be imported by the four-hour-aware PnL importer. Auxiliary checkpoint
contains logits/timings; published input contains no model-generated price data.

The offline runner now attempts cross-symbol tensor batches after the first actual
cutoff matches sequential winners and probabilities within 0.01. First-cutoff
outputs remain sequential; parity/error failures fall back to the serial engine.
The live inference service is unchanged.

One-shot completion publisher (does **not** start or resume GPU work):
`python3 scripts/publish_replay_when_ready.py --run-id RUN --input INPUT.json --expected-decisions 16200`
It reads the existing Modal Volume, validates the exact requested range/model and
completed count, passes results through the PnL importer, commits only the two
public report artifacts on main, pushes, and verifies the deployed API. Concurrent
report edits or incomplete runs stop publication. It writes a local publication
receipt alongside the input. The normal Modal local entrypoint also saves results
there, so the job is recoverable without further inference spend.

## ACTION_V1 stateful 15-minute replay

`modal_action_replay.py` (app/volume `jev-action-replay`) replays the frozen
[ACTION_V1](../training/ACTION_V1.md) contract; pure logic is `jev_inference/action_replay.py`.
Build input from official Binance data (fails on any gap, records archive sha256s):

```
python3 scripts/fetch_binance_replay_market.py --interval 15m --symbols "BTC ETH" \
  --start 2026-09-01T00:00:00Z --end 2026-09-08T00:00:00Z --output-dir /abs/dir
modal run inference/modal_action_replay.py --input-file /abs/dir/action-replay-input.json \
  --run-id action-001 --budget-usd 0.75 --max-decisions 200 --adapter-id OWNER/REPO \
  --adapter-revision SHA40 --adapter-sha256 SHA64 --hold-margin M
```

Input `{source, from, to, series:[{symbol, market?, candles}], provenance?}`; `[from,to)`
are 15-minute cutoffs and `time` is candle OPEN. Each cutoff needs the 96 candles
opening at `cutoff-96*15m .. cutoff-15m` plus the execution candle opening at `cutoff`,
for every symbol; coverage is validated locally before a GPU is allocated.
Every prompt comes from `action_task.build_job`; the forward candle is never model input.
Each symbol starts flat and carries the model's OWN paper position via
`action_task.apply_action`, executed at the close of the candle ending at the cutoff
(1 unit open, +1 add up to 3, halve on reduce, flat on close). Omitting the adapter
flags replays the base model. Adapter id/revision/sha256, hold margin, base revision,
contract hash and input form the run identity; a checkpoint from a different identity
is refused, and on resume every stored decision is re-derived from its logits and
margin and the position chain is re-verified. Budget, watchdog, `retries=0`,
`max_containers=1`, whole-cutoff checkpoints and the batch parity gate match the
hourly runner above.

Output: `{model, revision, task: "ACTION_V1", intervalMinutes: 15, holdMargin, from, to,
source, generatedAt, series:[{symbol, decisions:[{marketAsOf, action, sideAfter,
unitsAfter, price, probabilities}], candles}]}` where `to` is the completed end and
`candles[i]` is the execution candle opening at `decisions[i].marketAsOf`. The 7.5 bps
fee and paper PnL are for the downstream importer; they are not computed here.
