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

## Measured outcome — 2026-09-23: gate PASSED, deployed

- **Search:** V6 model on 2023, three families, `benchmarks/runs/v7-*.jsonl`.
- **2024 selection:** all 15 candidates were negative. The best (−1.66 %, fees 12.8 %) was
  selected: `{"allowed": ["open_long", "open_short", "close"], "cooldown_bars": 8,
  "margins": {"flat": -1.5, "position": -0.9}, "min_hold_bars": 32,
  "trend_filter": "return_24h_pct"}`. V6 without rules made −141.5 % on 2024.

Confirmation, 2022-04-02 → 12-31 ([results](ACTION_V7_RESULTS.json)):

| | Net | Fees |
|---|---:|---:|
| V7 | **+19.0 %** | 15.0 % |
| V6 | −29.2 % | 162.7 % |
| V4 | −333.0 % | 354.0 % |
| 1-unit hold | −70.2 % | — |

- Per coin: BTC +27.1 / ETH +25.6 / SOL −14.2 / XRP +37.5. Sharpe 0.51. Next-open fills: +18.8 %.
- All four criteria pass. BTC: 45 opens, 44 closes, 41.8 % time in position.

Reported:

| Window | V7 | Buy & hold |
|---|---:|---:|
| 2023 | +44.7 % | +311 % |
| 2024 | −1.7 % | +114 % |
| Leaderboard 2025-01 → 2026-06 | −15.9 % | −48.9 % |
| 10 coins, 2026-08-23 → 09-23 | −0.6 % | — |

The last 30 days on 10 coins had BTC with 0 opens.

Honest reading: the cost controls are real, with fees falling from 163 % to 15 %. The positive
2022 result coincides with a bear year and a short-leaning model. The policy has not beaten
buy-and-hold in bull years.

Deployment:
- Artifact sha `95531fcd…`, HF `guzus/jev-nyotti-action@71fb088`.
- The serving replay matches the lab decisions exactly.
- The live paper book restarted.
