# ACTION_V2 — pre-registration (one attempt)

Written and committed **before** the ACTION_V2 baseline or model was fit or scored (2026-09-23).
Successor to [ACTION_V1](ACTION_V1.md), which failed its gate. It keeps the ACTION_V1
interface (the `/action` API, the replay format, the paper rule, cadence, labels, splits and the
teacher episode rule). It changes only what the evidence below implicates.

## Why (evidence gathered on April validation only, CPU)

Numeric-baseline ablation on the ACTION_V1 representation. Each row: validation trade F1 and a
closed-loop rollout from flat over **April 2018**.

| Position fields shown | Val trade F1 | April opens / closes | Time in position |
|---|---:|---:|---:|
| side, unrealized, age, minutes since last execution (V1) | 0.563 | 13 / 12 | 99.6 % |
| same, time fields capped | 0.586 | 5 / 4 | 99.9 % |
| without minutes since last execution | 0.513 | 7 / 6 | 99.3 % |
| **side + unrealized return only** | **0.434** | **148 / 148** | **54 %** |

The self-referential time fields cause the absorbing state. Much of V1's teacher-forced timing
skill came from the trader's own activity bursts. The market-only timing signal is weaker, but
still far above chance (about 0.1 at these trade rates).

## Changes (prompt revision 2 in `inference/jev_inference/action_task.py`)

1. The position view shows only `side` and `unrealized_return_pct`.
2. Market string `BTC/USD` (`market_label`) is used identically in training, replay and live
   Kraken serving, which removes the train/serve string mismatch found in review.
3. Training: no base-model scoring pass (its time goes to optimization), about 3 epochs
   (915 steps), LR 2e-4. Train loss is reported against the 1.04-nat class-prior entropy.

## Gate (all four must pass; margin tuned on April validation only)

Criteria 1–3 are unchanged and use **test = May 2018**, with thresholds from the V2 numeric
baseline (same V2 features, all 24 candles, L2 and margin tuned on April).
**Disclosure:** May was already viewed in V1, so these three are no longer untouched-holdout
evidence.

Criterion 4 (no absorbing state) is judged on a **fresh confirmatory window no model has seen**:
closed loop from flat over Binance spot BTC/USDT 15m, 2026-08-23 → 2026-09-22 (2,880 decisions).
It needs ≥ 5 opens, ≥ 5 closes and time in position between 5 % and 95 %. This window is also
closest to live serving (spot, recent, same market string). No May rollout of the LoRA is run.

A failure blocks the live swap. **This is the last paid attempt in this budget.** If V2 fails, the
finding goes to the user; there will be no V3 without their decision.

## Budget

One H100 dispatch with the same caps as V1: at most $2.99 conservative (worker cap raised to 2,000 s), from the remaining $5.27
of the additional $10 approved.
