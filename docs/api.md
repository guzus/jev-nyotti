# TypeSafe-compatible API

The [TypeSafe API](https://docs.typesafe.ai/api) defines the transport contract. This implementation runs Qwen with different scoring/calibration. Configure the SDK base URL as the site origin **without `/v1`**, because it appends `/v1/systemone`. Set model `Qwen/Qwen3.5-4B`, not a Jev model identifier.

See [`api-example.json`](api-example.json) for a complete mixed-primitive request. Keep secrets out of frontend code and URLs.

Official SDK example (verified using `typesafe-sdk==0.7.1`):

```python
import os
from typesafe_sdk import TypeSafeClient, Choice, RetryPolicy

with TypeSafeClient(
    api_key=os.environ['JEV_API_KEY'],
    base_url='https://jt.memtherscan.xyz',
    model='Qwen/Qwen3.5-4B',
    retry=RetryPolicy(max_retries=0, timeout=180),
) as client:
    result = client.system_one(
        {'description': 'A red apple'},
        {'color': Choice(instructions='Choose the stated color.', criteria={'red': None, 'blue': None})},
    )
    print(result.choices['color'].choice)
```

```sh
curl "$JEV_BASE_URL/v1/systemone" -H "Authorization: Bearer $JEV_API_KEY" -H 'Content-Type: application/json' --data-binary @docs/api-example.json
curl "$JEV_BASE_URL/v1/trading/decisions" -H "Authorization: Bearer $JEV_API_KEY" -H 'Content-Type: application/json' --data '{"symbol":"BTCUSD","interval":15}'
```

## Semantics

1–8 independent questions share a state; question IDs never enter the prompt. `choice`: 1–255 criteria. `score`: 2–10 ordered criteria. `noul`: false/true classification. State and instructions accept string, array or object; choice descriptions also accept null. Limit: 64 KiB HTTP body, 8,192 tokens per expanded prompt, no truncation. Repeating large state across questions can hit the inference body limit sooner.

Choice returns argmax plus a normalized distribution over **all** requested options. Score returns the expected zero-based criterion index, distribution and legend. Noul returns the true-label score. They are conditional next-label likelihoods, not trading success estimates; candidate order and wording can affect results.

Choice/score confidence is `1 − H(p)/log(N)` (1 for one option), not TypeSafe's proprietary calibration. Usage counts each complete independent input prompt, with `output_tokens=0` because inference reads logits without decoding. Metadata records exact model revision and score/usage semantics. The Korean summary is an explicit deterministic result summary, not a generated trading rationale.

Errors: `{ "error": { "code": "...", "message": "..." } }`. 401 invalid API key; 413 body size; 422 schema/model/token limit; 429 rate/daily limit; 502 invalid upstream response; 503 model/market unavailable; 529 active capacity. Do not auto-retry daily-limit errors. Use bounded backoff for transient failures; each retry may reserve another quota unit.

## Public demo

Public analysis takes only a fixed symbol/interval and server-owned prompt. No API key or arbitrary prompt reaches the browser. The chart shows 96 closed candles; 24 plus numerical features enter the model. Change covers the graphed window; RSI uses simple mean gain/loss over 14 intervals; volatility is population standard deviation of close returns (%); volume ratio uses the preceding 20-candle mean.

`/?decision=<uuid>` resolves the original persisted output, market cutoff and model revision without re-running inference. Shares expire after 30 days. Cached outputs retain original timestamps. Long/short/hold are hypothetical directions inferred from spot candles, not executable orders.

`GET /api/market?symbol=BTCUSD&interval=15` includes `cachedDecision` (the latest persisted decision for the selected symbol/interval and configured model/revision/training status, with `cached: true`, or `null`) and:

```json
{"cache":{"refreshIntervalMs":600000,"checkedAt":null,"nextRefreshAt":null,"refreshing":false,"error":null}}
```

`checkedAt` is the most recent completed refresh attempt, including failed attempts. `nextRefreshAt` is the scheduled next check (or outstanding lease expiry when later), not a guarantee that inference will finish then; a busy serial queue can lag. It is `null` when scheduling is disabled. `refreshing` reflects the persisted in-progress lease. `error` contains the latest refresh or current market-fetch error. A failed refresh retains the last decision, whose `marketAsOf` and `generatedAt` remain unchanged. If the market provider fails, the endpoint serves the last persisted candle snapshot when available. First use with no market snapshot may still return an upstream error. All responses use `Cache-Control: no-store`.

With `SCHEDULED_ANALYSIS_ENABLED=true`, one durable background worker checks the ten displayed coins × 15/60/240-minute intervals every ten minutes. SQLite persists due times, cooldowns and worker leases across deploys; one inference runs at a time and crashed jobs wait for their lease/cooldown before recovery. Identical snapshots reuse the saved result. Cache keys include the full closed-candle snapshot, cutoff, model identity/revision, training status and prompt version. Preserve the Railway `/data` volume.

`POST /api/analyze` and authenticated `POST /v1/trading/decisions` are cache reads only. They return a stored Decision with HTTP 200, or HTTP 202 with `{"status":"pending","cachedDecision":null,"cache":{...}}`. Neither route wakes the GPU, advances the queue or bypasses a cooldown/quota. Historical shares remain immutable even when the current model changes. Older symbols retained for share compatibility are not added to the thirty-slot schedule.

Hits preserve `id`, `generatedAt`, `marketAsOf`, scores and `latencyMs`, and set `cached: true`. `latencyMs` means the **original inference duration**, not cache retrieval time. Reads consume no model budget; HTTP limits remain 120 analysis requests/IP/minute and 60 market reads/IP/minute. The worker reserves daily quota before a model call; failures count conservatively and are never saved as decisions. Authenticated `/v1/systemone` is unchanged, uncached, and shares the daily model budget.

`GET /api/status` additionally exposes `scheduledAnalysisEnabled` and optional `gaMeasurementId` (`null` when unset). The latter is validated as `G-` plus uppercase alphanumerics. CSP permits GA4's script and collection origins, without enabling advertising frames.

Supported USD markets: `BTCUSD`, `ETHUSD`, `SOLUSD`, `XRPUSD`, `DOGEUSD`, `ADAUSD`, `AVAXUSD`, `LINKUSD`, `DOTUSD`, `LTCUSD`, `BNBUSD`, `SUIUSD`, `NEARUSD`, `PEPEUSD`, `ZECUSD`. Intervals remain 15, 60 and 240 minutes. Dogecoin uses Kraken’s `XDGUSD` market internally; clients always use `DOGEUSD`. All markets share the existing cache, quota and closed-candle validation rules.

The homepage shows only the ten symbols in [volume-ranking.json](volume-ranking.json), ordered by CoinGecko-reported global 24h USD volume after excluding stablecoins. The UI labels the snapshot date; ranking is not auto-refreshed. The five previous symbols remain API-supported to preserve historical shares. Global aggregate volume selects the list only; charts and inference still use Kraken USD candles.

## Fine-tuned decision semantics

With `MODEL_TRAINING_STATUS=fine_tuned`, trading results use `action: long | short | flat` and the corresponding three `scores` keys, with `semantics: next_hour_position_side`. `flat` means zero exposure; it is not silently renamed to the old base model's `hold` stance. Old shared base results retain `hold` and their original revision. The prediction horizon is one hour regardless of the selected chart interval. Inputs assume a hypothetical prior flat position and use Kraken spot candles, while training used hourly BitMEX BTC exposure; this transfer is experimental and has no demonstrated profitability. `/v1/systemone` still accepts caller-defined questions.

## ACTION_V1 stateful mode

With `MODEL_TRAINING_STATUS=action_v1` (contract: [training/ACTION_V1.md](../training/ACTION_V1.md)), the worker decides once per closed 15-minute Kraken candle for the ten displayed coins. 1h/4h stay market views; decision reads for them return 422 `action_interval_unsupported`. BTCUSD is `transfer: in_distribution`; the other nine are `untested_transfer`.

The gateway never builds the action prompt. It sends `POST {INFERENCE_URL}/action` with the usual auth headers and `{market:"Kraken BTCUSD spot", cutoff, candles, position}`: `cutoff` as ISO UTC `YYYY-MM-DDTHH:MM:SSZ`, exactly 96 contiguous closed candles (`time` = candle open), and the carried paper position `{side, entry_price, opened_at, last_trade_at}`. The response is validated as `{model, revision, task:"ACTION_V1", action, options:[{name, probability}], holdMargin, inputTokens, elapsedMs}`. The revision must equal `MODEL_REVISION`, and the options must equal the state's option set: flat → `hold/open_long/open_short`; long or short → `hold/add/reduce/close`. `action` is `argmax(logits + hold margin)`, so it can differ from the top probability.

Paper inventory is kept in SQLite per (symbol, model revision). Each cutoff is applied at most once in one transaction. `server/paper.ts` ports `action_task.apply_action`: open = 1 unit, add = +1 (max 3, harmonic entry), reduce = halve, close = flat. Execution is at the cutoff candle's close, with a 7.5 bps fee per unit traded. PnL is a percent of one unit's notional. Missed cutoffs are not backfilled: the next decision resumes from the latest candle and logs a `gap` row with `missedCutoffs`. The action log is append-only.

In this mode the decision JSON (`/api/analyze`, `/v1/trading/decisions`, `cachedDecision`) adds `task`, `options`, `holdMargin`, `transfer`, `positionBefore`, `execution`, `paper` (side, units, entryPrice, markPrice, unitReturnPct, unrealizedPct, realizedPct, feesPct, trades), `missedCutoffs` and `actionLog` (latest 50, newest first). `GET /api/paper` summarizes all ten coins. `/api/status` reports `task: "ACTION_V1"` and `decisionIntervals: [15]`.

`pnl-import.js` also accepts an ACTION_V1 closed-loop replay: `{model, revision, task:"ACTION_V1", intervalMinutes:15, holdMargin, from, to, source, generatedAt, series:[{symbol, decisions:[{marketAsOf, action, sideAfter, unitsAfter, price, probabilities}], candles:[…15m OHLCV]}]}`. Every cutoff in `[from, to)` needs one candle and one decision. The importer replays the paper rule from flat and fails closed if the `sideAfter`/`unitsAfter` chain, the execution price (cutoff candle close) or the option probabilities disagree. It reports equal-weight unit PnL after fees, drawdown in percentage points, time in position, and 1-unit buy-and-hold.
