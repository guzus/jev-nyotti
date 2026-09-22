# Experiment plan

This document describes proposed work, not an implemented pipeline. The first target is imitation of observed trading actions; profitability is a separate hypothesis.

## Data prerequisites

- Confirm source authenticity, permission to use the data, exchange, timezone, instrument identifiers, contract multipliers, and fee/funding conventions.
- Group partial fills into orders where identifiers allow it. Preserve raw provenance outside Git. Execution time is not necessarily decision time.
- Reconstruct cash/equity and position inventory, separating transfers from trading P&L. Missing initial balances or positions must remain explicit.
- Join market observations available before the decision: prices, volumes, volatility, funding, and order book where available. An execution's price cannot be an input to predicting that execution.
- Construct observation intervals with no executions, while acknowledging that inactivity does not prove a deliberate hold decision. No fabricated trader rationales.

## Proposed example contract

Input fields: decision timestamp in UTC, market-data cutoff, instrument and venue, causal market features, pre-decision signed position, equity/collateral, and recent actions.

Target fields: action (`hold`, `open`, `increase`, `reduce`, `close`), signed target exposure relative to equity, and an observation-quality flag. Define whether exposure means notional/equity or margin/equity before making labels; never mix units. A short exposure can be negative. This is a proposed contract awaiting an actual export.

Use explicit missing values and dataset version identifiers. Do not convert absolute historical BTC order sizes directly into present-day trade sizes. Funding, leverage, liquidation mechanics, and inverse contracts need venue-specific reconstruction.

## First trial

1. Audit a small sample and reconstruct a continuous period. Count independent decisions after grouping fills.
2. Reserve later contiguous validation/test periods; prevent overlapping feature/outcome windows crossing split boundaries. Fit preprocessing on training periods only.
3. Select about 10,000 usable training examples spanning actions and market conditions, keeping the natural class distribution visible. Tokenize to replace the illustrative price assumptions.
4. Compare an untuned Qwen3.8-27B, one LoRA-tuned checkpoint, and a numerical baseline on the same inputs. Match generation settings; never let test results drive repeated tuning.
5. Evaluate schema validity, action/size errors, turnover, fees/funding/slippage, drawdown, and net returns in a simulator. Historical fills cannot be assumed for a policy that chooses different orders. Report simulator limitations.
6. Run internal paper evaluation with timing/error logs. Record full generated-token counts, timeout behavior, and prediction parity if changing serving providers or quantization.

Model training, paid endpoints, exchange credentials, and live order execution are not part of this repository's initial delivery. Any future execution layer needs deterministic position limits, stale-data checks, and duplicate-order protection independent of model text.

## Open questions

- Where is the actual trade export, and what fields are present?
- Is the intended decision horizon minutes, hours, or longer?
- One instrument or many? Separate calls or one portfolio decision?
- What latency and monthly budget are acceptable after the pilot?
- What is Jev's actual integration interface?
