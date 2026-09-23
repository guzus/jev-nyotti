# ACTION_V6 — pre-registration (full 2018–2021 teacher history)

Committed before the V6 dataset was built or any V6 model was fit (2026-09-23).

## Why

Every earlier model learned from **March 2018 only**: 1,220 rows, 140 opens. The codex 1m
downloader has now completed the official BitMEX XBTUSD 1m history from 2018-03 to 2022-01
(2,018,881 bars, sha256 `75d4c248…`, no gaps). The teacher's fills cover the same span
(3–64k fills per month). V5 showed that decision rules cannot create an edge the model lacks.
The next lever is more data for the model itself.

## Data and splits (chronological; unchanged ACTION_V1 labels, prompt revision 2)

| Split | Cutoffs | Use |
|---|---|---|
| Train | 2018-03-02 → 2020-07-01 | fit; every action window + 2 holds per action per side |
| Validation | 2020-07-02 → 2021-01-01 | model/feature/C choice and per-family rate-matched margins |
| Test | 2021-01-02 → 2022-01-01 | scored once, gate |

- The ACTION series has never scored 2021.
- An unrelated hourly exposure pilot evaluated 2021 in September 2026, on a different task and
  labels. This is disclosed.
- Validation and test use the natural distribution.

## Candidates, selected on validation only

- Logistic regression per family with the V2 features, C ∈ {0.003, 0.01, 0.03, 0.1}.
- The same with time-of-day features.
- HistGradientBoosting (it may work with 100× more data).

The winner is the candidate with the best validation trade F1 whose rate-matched margins give
validation predicted/teacher trade rate within 0.9–1.1×.

## Gate (test 2021, frozen margins)

1. Trade F1 above **V4 applied to the same 2021 test** (the live model's imitation on unseen
   data).
2. Action macro-F1 above V4's on 2021.
3. Predicted/teacher trade rate within 0.5–2×.
4. Unseen closed loop, Binance BTC 15m **2022-03-02 → 04-01**: ≥ 5 opens, ≥ 5 closes, time in
   position 5–95 %.

Fee-inclusive PnL is reported, not gated: the 2022 window, the V5 confirmation window and the
2023/2024 lab years. If the gate passes, V6 replaces V4 live.

## Measured outcome — 2026-09-23: gate FAILED (1 of 4), not deployed

- **Validation selection** (`.runtime/action-v6/v6_selection.json`): HistGradientBoosting won with
  val trade F1 0.252. Logistic regression scored 0.19–0.20; with time-of-day features, 0.191.
- **Frozen margins:** flat −1.58, position −1.09. [Results](ACTION_V6_RESULTS.json).

| Criterion (2021 test) | V6 | Threshold | |
|---|---:|---:|---|
| Trade F1 | 0.205 | > 0.130 (V4 on 2021) | pass |
| Action macro-F1 | 0.063 | > 0.035 (V4 on 2021) | pass |
| Predicted / teacher trade rate | 2.29× | 0.5–2× | **fail** |
| Unseen 2022-03 closed loop | 22 opens, 21 closes, 53 % in position | non-absorbing | pass |

Findings:
- More data improves imitation: trade F1 is +58 % over the live V4 on unseen 2021.
- The 2022 30-day rollout lost −4.6 % after fees (6.4 % fees), the best out-of-sample PnL so far.
  It is still negative.
- The rate-matched margins drifted. The trader's activity changed between 2020 H2 and 2021, so a
  static margin over-trades. 2021 has now been viewed, and no unseen teacher period remains.
