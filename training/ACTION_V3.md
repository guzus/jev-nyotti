# ACTION_V3 — pre-registration (numeric policy; one promotion candidate)

Committed **before** the V3 candidate was fit on March+April, before it touched May, and before
any policy was scored on the confirmatory window (2026-09-23). Interface, labels, cadence, paper
rule and prompt revision 2 are unchanged from [ACTION_V2](ACTION_V2.md).

## Evidence used (April validation only, CPU; no May or confirmatory data)

| Candidate on V2 features (fit on March) | Val trade F1 | Flat-open AUC | April closed loop |
|---|---:|---:|---|
| Logistic regression, C=0.01 (V2 baseline) | **0.434** | **0.648** | 148 opens, 54 % in position |
| + time-of-day (hour sin/cos) | 0.421 | 0.645 | 112 opens, 62 % |
| + time-of-day + weekend | 0.412 | 0.647 | 121 opens, 56 % |
| HistGradientBoosting (best of 4 configs) | 0.389 | 0.579 | 194 opens, 45 % |
| Qwen LoRA V2 (fit on March) | 0.454 | 0.540 | (fresh window: 0 opens) |

Regularized logistic regression carries the best entry signal. The LoRA does not read these
numeric inputs well enough to time entries. The binding constraint is **training size**: March
only has 1,220 rows and 140 opens.

## The single promotion candidate

**A numeric policy, not Qwen.** Logistic regression per state family on the V2 features
(`action_baseline.vector`), `C = 0.01`, `class_weight='balanced'`.

- Refit on **March train rows plus April rows**, subsampled by the train rule: every action
  window plus 2 fixed-seed holds per action per side, seed 3407.
- Hold margin is frozen at **+0.10**: the April-tuned margin of the identical March fit. May is
  not used to set it.

It is served on CPU and labelled honestly as a numeric policy. No LLM decision is shown under the
Qwen name.

## Gate (unchanged criteria; thresholds fixed from `ACTION_V2_BASELINE.json`)

1. Test (May 2018) trade F1 must be above **0.298**.
2. The predicted/teacher trade rate must be within 0.5–2×.
3. Test executed-action macro-F1 must be above **0.109**.
4. **Confirmatory, unseen:** closed loop from flat over Binance spot BTC/USDT 15m,
   **2026-07-24 → 2026-08-23**, needs ≥ 5 opens, ≥ 5 closes and time in position 5–95 %.
   Scored once, only with this candidate.

Disclosure: May has been scored by the V1 and V2 baselines and LoRAs. It is not a clean holdout,
and criteria 1–3 are imitation checks, not proof of skill. Criterion 4's window is new.
Fee-inclusive paper PnL is reported, not gated, over both the confirmatory window and the V2
window (2026-08-23 → 09-22).

If the gate passes: serve the policy through the CPU `/action` path, flip the site to action mode
after a real endpoint call, mobile check and review, and publish the artifact and model card.
If it fails: do not flip.

## Exploratory (not eligible for promotion in V3)

At most one Qwen LoRA dispatch (≤ $3). It is fit on March+April with flat-family rebalancing, and
the prompt carries the candidate's option probabilities as `numeric_prior`. It is scored on May and
on the confirmatory window **for reporting only**, to judge whether a future Qwen version is worth
running.

## Measured outcome — 2026-09-23: gate FAILED, not promoted

[Aggregate results](ACTION_V3_RESULTS.json). Fit once on 2,436 rows (March train + April subsample).

| Criterion | Value | Threshold | |
|---|---:|---:|---|
| May trade F1 | 0.236 | > 0.298 | **fail** |
| Predicted / teacher trade rate | 4.14× | 0.5–2× | **fail** |
| May action macro-F1 | 0.151 | > 0.109 | pass |
| Confirmatory closed loop, 2026-07-24 → 08-23 | 70 opens, 69 closes, 62 % in position | ≥ 5 / ≥ 5, 5–95 % | pass |

The frozen +0.10 margin came from the March-only fit and was miscalibrated after the refit, so
the policy over-trades. The confirmatory window shows it does not get stuck, but it is not
profitable: −17.2 % fee-inclusive paper return (22.7 % fees), 20.7 % max drawdown. The previously
viewed window (08-23 → 09-22) was −32.2 %. Every policy tested in V1–V3 that trades loses money
after 7.5 bps fees at this cadence. Production is unchanged.
