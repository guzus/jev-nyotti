# Corrective market-only diagnostic

The original adapter does not establish a useful trading policy. It matched position persistence on held-out accuracy and missed all three sampled validation/test transitions. An inspected replay snapshot contains **6,590 flat predictions and zero position changes**. The replay GPU and its automatic publisher were canceled; the public 20-decision pilot remains unchanged.

## Root cause evidence

The original 4,096-row training sample contained 173 changes (4.2%). Only 29 examples departed from a flat prior, versus 270 that stayed flat. Its validation and test periods had no flat priors or flat targets. Consequently, the reported 99.22% test accuracy never evaluated the site's flat starting condition.

The live prompt supplies a flat prior on every call. Historical replay starts flat and carries model predictions forward. Copying the prior therefore creates an absorbing flat state. The source task predicts actual hourly BitMEX BTC exposure; inference additionally changes exchange, volume units, asset, dates and holding cadence. None of those transfers was validated by the original accuracy result. Hourly exposure labels also discard intrahour round trips and size changes, so a trade execution count is not a position-label transition count.

## Fixed experiment

`transition_data.py` builds a new task, `NEXT_HOUR_POSITION_SIDE_MARKET_ONLY_V2`, from the same reconciled BTC source. It removes previous exposure from the model input and explicitly describes it as unavailable. Actual previous exposure remains outside the prompt solely for evaluation; it is never falsified or overwritten with a hypothetical position.

- Train: 2,048 distinct examples from 2018–2020, with 1,024 transitions and 1,024 unchanged examples.
- Validation: union of 413 distinct 2021 H1 examples. Report **natural sample (256)**, **all validation transitions (75)** and **first contiguous 96 hours** separately. These cohorts overlap; their metrics must not be added. The enriched union's accuracy is not natural-distribution accuracy.
- The 96-hour path starts simulated exposure flat. Because the new model has no position input, carried exposure affects transition counts, not later model inputs.
- 2021 H2 test data is not evaluated in this diagnostic. Prior experiments already inspected that calendar period; it must not later be advertised as completely untouched.
- Chronological splits retain the original 96-hour purge and causal closed-candle boundary.

The base model and new LoRA are both evaluated on exactly the same cohort inputs. Metrics include confusion matrices, per-class precision/recall, macro-F1, correct transition precision/recall, false changes on unchanged hours, predicted occupancy and rollout entry/exit/reversal counts. Baselines are actual-prior persistence, always flat and the training-majority short class. Persistence uses information deliberately withheld from the ablation and is disclosed as that stronger reference.

The diagnostic screen is fixed before the run: nonconstant natural output; natural macro-F1 above both the base and majority; natural accuracy within 2 percentage points of persistence; correct transition recall at least 10%; correct natural transition precision at least 20%; false-change rate at most 10%. Passing this is **not promotion approval**. Independent testing, entry/exit coverage, cost-aware PnL and matching deployment inputs remain required. No replacement adapter is automatically served or uploaded to the public model repository.

## Bounded run

`modal_transition.py` launches one rank-16 BF16 LoRA run from the pinned base, at most 512 optimizer steps, with no automatic retry. Its 1,200-second immutable dispatch deadline, subprocess timeout, watchdog and explicit cancellation bound compute. A conservative allocation of `(1200 + 120 + 180) × $0.0013 = $1.95` leaves the configured **$2 compute limit** intact; image/storage charges and the provider invoice are separate. This is not an account billing cap.

The first corrective attempt reserved $2 under the earlier $5 pilot allowance, alongside the old pilot's $2.23 configured allocation. That canceled attempt's full reservation remains accounted for until actual usage is reconciled. The durable retry reserves $2 from the subsequently approved **additional $10**, leaving $8 unreserved. The replay's $10 allowance is separate. Limits are per dispatch; there is no automatic retry or automatic spending of the balance.

```sh
PYTHONPATH=inference:training python -m unittest discover -s training/tests
python training/transition_data.py --source /path/to/export \
  --candles .runtime/real-training/bitmex-xbtusd-1h.jsonl \
  --output .runtime/real-training/market-only-v2
PYTHONPATH=inference:training python -m modal deploy training/modal_transition.py
PYTHONPATH=inference:training python -m modal run training/modal_transition.py \
  --dataset-dir .runtime/real-training/market-only-v2 --background
```

Use Python 3.11+ with the inference validation dependencies and Modal installed. Raw records, generated examples, per-cutoff predictions and weights remain ignored/private. Only aggregate results and code are published. Pipeline/export `status: passed` means execution succeeded; use `diagnostic_gate` for measured behavior and `promotion_gate` for serving eligibility.


## Interruption and durable recovery

The first corrective attempt (`v2-26a1d18dae1040f8864a946f5c1bb299`) completed the base-model evaluation at 144.84 child-process seconds but was stopped when the attached local client disconnected. No optimizer step or replacement adapter was confirmed. Its partial report is not a successful training result. The replay was separately stopped at 7,100 saved decisions; the inspected 6,590-decision snapshot is a subset, not the full stopped checkpoint.

A further **$10** was explicitly authorized for correction. The next attempt reserves **$2** of that allowance; there are no automatic additional experiments. Deploy the bounded function first, then use `--background` so dispatch targets the deployed function. The short local dispatcher uploads verified inputs, persists the call ID, and exits. It does not wait for or cancel the remote call. Retrieve results using that call ID or the private artifact volume. Completion does not publish or promote the adapter.
