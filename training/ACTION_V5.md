# ACTION_V5 — pre-registration (cost-aware decision rules on V4)

Committed while the rule search runs on the **2023 tune window only**. This is before any rule
is scored on 2024 (selection) and before any 2025–2026 confirmation data exists on disk
(2026-09-23). The model weights and prompt are unchanged from [ACTION_V4](ACTION_V4.md). V5 adds
only a deterministic decision layer (`inference/jev_inference/decision_rules.py`): margins,
allowed actions, minimum holding, cooldown, open-confidence floor and trend alignment.
Production, the replay and the lab call the same function.

## Why

V4 imitates the trader's timing (it passed its gate) but loses money. Measured by the PnL lab
(`training/pnl_lab.py`, which reproduces V4's confirmatory rollout exactly), fee-inclusive,
in % of one unit notional:

| | 4-coin mean net | fees |
|---|---:|---:|
| 2023 | −418.7 % | 325.7 % |
| 2024 | −363.96 % | 384.6 % |

Fees dominate. The source trader mostly traded as a **maker with rebates** (504,271 negative
commission fills). The paper rule charges a 7.5 bps taker fee, so copying the trader's frequency
cannot pay.

## Protocol

1. **Search:** rule configs on 2023 (BTC, ETH, SOL, XRP), in four parallel families
   (turnover, confidence, trend, combo). Every evaluated config is logged in
   `benchmarks/runs/*.jsonl`. An activity floor applies: each symbol needs ≥ 24 trades per year.
2. **Selection (2024):** the critic's shortlist plus each family's best, deduplicated. The config
   with the highest 2024 4-coin mean net return that meets the activity floor wins. Selection is
   done once, and no config is edited after it sees 2024.
3. **Confirmation (never seen):** 2025-01-01 → 2026-06-23, same four coins. The data is downloaded
   only after selection. V5 is scored once, alongside the V4 rules for reference.

## Gate

1. The 4-coin mean net return on the confirmation window must be **above V4's** on that window.
2. The activity floor must hold on confirmation: each symbol ≥ 24 trades per year, pro-rated.
3. BTC must not get stuck on confirmation: ≥ 5 opens, ≥ 5 closes, time in position 5–95 %.

Reported, not gated:
- Absolute profit: whether the confirmation net is > 0.
- The same run with next-bar-open fills.
- Buy-and-hold comparison.
- Teacher-forced imitation metrics on May 2018.

If the gate passes, V5 replaces V4 live (a new paper book), and the benchmark chart and the
Hugging Face card are updated. If it fails, V4 stays live.
