# ACTION_V1 — stateful 15-minute execution-action imitation

Frozen **before** any ACTION_V1 model was trained or scored (2026-09-23). Fixes the
failure recorded in [TRANSITION_DIAGNOSTIC.md](TRANSITION_DIAGNOSTIC.md). The earlier
adapter was trained to copy its prior position, and serving always called it flat, so it
stayed flat. ACTION_V1 predicts **what the trader executes next**, given a carried
position state. The live service and the replay feed each decision back into that state.

## Contract (single implementation: `inference/jev_inference/action_task.py`)

- **Cadence:** one decision per 15-minute UTC boundary (`cutoff`). Kraken publishes 15m
  OHLC, so live BTC matches the training cadence. Other coins are labelled untested transfers.
- **Market input:** 96 closed 15m candles ending exactly at `cutoff`. Training candles are
  aggregated from official BitMEX XBTUSD 1m buckets (bucket end ≤ cutoff). Everything is
  unit-free: returns, ranges, and volume relative to the 96-bar mean. No absolute price or
  volume unit reaches the model.
- **Position state:** `side` (flat/long/short), `unrealized_return_pct` from a harmonic entry,
  `position_age_minutes`, and `minutes_since_last_execution`. Size is **not** an input:
  teacher contract counts and paper units are not comparable.
- **Options depend on state:**
  - flat → `hold / open_long / open_short`
  - long or short → `hold / add / reduce / close` (a reversal counts as `close`)
- **Target (teacher):** the first same-direction execution episode in `[cutoff, cutoff+15m)`.
  An episode ends at an opposite-side fill or at a gap of more than 60 s. The class is found by
  applying the episode's signed quantity to the reconstructed position before `cutoff`.
  `hold` is used only when there is no fill. A window with same-timestamp fills on both sides
  is excluded as ambiguous.
- **Paper execution rule (replay and live only):** `open` = 1 unit, `add` = +1 unit (max 3,
  harmonic entry), `reduce` = halve, `close` = flat. Execution happens at the cutoff candle's
  close with a 7.5 bps taker fee per unit traded. **Teacher sizes are not imitated in V1.**
- **Decision rule:** `argmax(logits + b)`, where `b` subtracts a margin `m` from `hold` only.
  `m` is chosen on validation and then frozen before test is scored. It is served as a config
  value and is reported.

## Data and splits (chronological, no shuffling across time)

| Split | Cutoffs | Use |
|---|---|---|
| Train | 2018-03-02 → 2018-03-31 | LoRA + numeric baseline fit |
| Validation | 2018-04-02 → 2018-04-30 | margin `m`, early checks |
| Test | 2018-05-02 → 2018-05-31 | reported once, gate |

The first 24 h of each month is purged, so no lookback crosses a split. Pre-freeze 15m
counts are all-window, pre-purge. March: 1,623 flat-hold, 465 position-hold, 140 opens,
335 add/reduce/close. Train uses every action window plus a fixed-seed subsample of holds.
Validation and test are scored on the **natural** distribution: every window, no enrichment.

## Promotion gate (all must pass on **test**, with `m` frozen from validation)

1. **Trade detection:** trade-vs-hold F1 must exceed always-hold (0) **and** the numeric
   baseline. The baseline is logistic regression on the same unit-free features and state,
   with its threshold also tuned on validation.
2. **Trade rate sanity:** predicted trade rate within 0.5×–2× of the teacher's actual rate.
3. **Action quality:** macro-F1 over the executed classes on teacher-forced test must exceed
   the numeric baseline's.
4. **No absorbing state:** a closed-loop rollout from flat over all of May 2018 (the model's own
   paper position) must have ≥ 5 opens, ≥ 5 closes and time-in-position between 5 % and 95 %.

Fee-inclusive paper PnL, drawdown and exposure are **reported, not gated**. After a
divergence, teacher actions are not the correct labels for the model's own states. Any
failing criterion blocks the live swap. The pipeline and report still ship, and the
existing model stays live until the user decides.

## Honest limits

Only three months, one market regime and ~475 training actions. The data holds executed
fills only, not intentions, cancels, news or the order book. Matching the teacher's actions
is not the same as matching the teacher's profitability. The source attribution is
user-supplied and not independently verified.
