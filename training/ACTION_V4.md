# ACTION_V4 — pre-registration (rate-calibrated numeric policy)

Committed **before** V4 touched May or its confirmatory window (2026-09-23). Same model as
[ACTION_V3](ACTION_V3.md): logistic regression, fit on March+April, artifact sha256 `2e068dad…`.
No refit. **Only the decision rule changes.**

## Why

V3 traded at 4.1× the teacher's May rate. In live replay it lost −89.8 % over 30 days on 10
coins, with fees 81.9 %. On April, in-sample, the frozen +0.10 margin over-trades 3.7× when flat
and 1.3–1.7× in position. Because the margin is *subtracted from hold*, a positive margin adds
trades.

## Change

The hold margin is set per state family, each chosen so the predicted trade rate on April
natural rows equals the teacher's. The grid is 0.01 wide over [−3, 0].

| Family | Margin | April predicted / teacher rate | April trade F1 (in-sample) |
|---|---:|---:|---:|
| flat | **−0.68** | 1.00 | 0.270 |
| in position | **−0.25** | 1.00 | 0.592 |

## Gate (V3 criteria; thresholds from `ACTION_V2_BASELINE.json`)

1. May trade F1 > 0.298.
2. May predicted/teacher trade rate within 0.5–2×.
3. May action macro-F1 > 0.109.
4. **New unseen window:** closed loop from flat on Binance BTC/USDT 15m,
   **2026-06-24 → 07-24**. It needs ≥ 5 opens, ≥ 5 closes and time in position 5–95 %.

May has been viewed repeatedly, so criteria 1–3 are imitation checks, not holdout evidence.
Fee-inclusive PnL is reported, not gated: the V4 window, the V3 windows, and the 10-coin
08-23 → 09-23 replay.

**Serving:**
- If the gate passes, V4 replaces V3 live, and the live paper book restarts (new revision).
- If it fails, V3 stays live as it is today, and the user is told.

## Measured outcome — 2026-09-23: gate PASSED, deployed

[Aggregate results](ACTION_V4_RESULTS.json).

| Criterion | Value | Threshold | |
|---|---:|---:|---|
| May trade F1 | 0.370 | > 0.298 | pass |
| Predicted / teacher trade rate | 1.22× | 0.5–2× | pass |
| May action macro-F1 | 0.158 | > 0.109 | pass |
| Unseen closed loop, 2026-06-24 → 07-24 | 39 opens, 39 closes, 44 % in position | ≥ 5 / ≥ 5, 5–95 % | pass |

The gate measures imitation, not profit. Fee-inclusive paper returns are still negative:

| Window (BTC unless noted) | Return | V3 |
|---|---:|---:|
| 06-24 → 07-24 | −18.1 % | — |
| 07-24 → 08-23 | −8.2 % | −17.2 % |
| 08-23 → 09-22 | −19.3 % | −32.2 % |
| 10 coins, 08-23 → 09-23 | −67.0 % | −89.8 % |

Deployment:
- Artifact sha256 `9101f3f5…`, published at HF `guzus/jev-nyotti-action@11c710b`.
- The serving path reproduces the gate rollout's action counts exactly.
- The live paper book restarted under the new revision.
