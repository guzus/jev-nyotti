# ACTION_V7 — pre-registration (V6 model + direction-neutral cost rules; last iteration)

Committed before any V7 rule was searched, selected or confirmed (2026-09-23).

## Why

V6 imitates better than the live V4 on unseen 2021 (trade F1 0.205 vs 0.130). Its unseen 2022-03
closed loop was about +1.8 % gross and −4.6 % net, so fees were the whole difference. That is
the first gross-positive out-of-sample result in the series. V5 showed rules can cut fees, but
V5 won on long beta. V7 therefore combines the V6 model with rules that must stay
**direction-neutral**.

## Protocol

- **Model:** the V6 artifact, sha `c639c6c5…`, unchanged. Only `decision_rules` are searched.
- **Constraint:** `allowed` must contain both `open_long` and `open_short`. The activity floor
  is ≥ 24 trades and ≥ 12 opens per symbol per year.
- **Search:** 2023, 4 coins, parallel agents, logged in `benchmarks/runs/v7-*.jsonl`.
- **Selection (2024, once):** highest 4-coin mean net among the families' top configs that meet
  the constraint and the floor. The in-simulator 1-unit hold is reported alongside.
- **Confirmation (never seen):** Binance 15m BTC/ETH/SOL/XRP, **2022-04-02 → 2022-12-31**, a bear
  year. It is downloaded only after selection.

## Gate (confirmation)

1. 4-coin mean net above V6 without rules **and** above V4 on the same window.
2. 4-coin mean net above the 1-unit buy-and-hold control.
3. Activity floor holds, pro-rated.
4. BTC is non-absorbing: ≥ 5 opens, ≥ 5 closes, time in position 5–95 %.

Absolute profit (net > 0) is reported, not gated. Imitation on 2021 with the rules applied is a
disclosed check on viewed data.

**Stopping rule:** this is the last iteration on reused teacher data. If V7 fails, the
leaderboard and findings are the deliverable. No V8 on reused windows.

## Standing limits

- No unseen teacher period remains.
- The 2025-01 → 2026-06 leaderboard window is used for display only. No selection is ever made
  on it.
- The paper fee is a 7.5 bps taker fee, while the source trader mostly earned maker rebates.
  Maker fills are not simulated from 15m candles.
