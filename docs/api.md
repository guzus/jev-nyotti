# TypeSafe-compatible API

The [TypeSafe API](https://docs.typesafe.ai/api) defines the transport contract. This implementation runs Qwen with different scoring/calibration. Configure the SDK base URL as the site origin **without `/v1`**, because it appends `/v1/systemone`. Set model `Qwen/Qwen3.5-4B`, not a Jev model identifier.

See [`api-example.json`](api-example.json) for a complete mixed-primitive request. Keep secrets out of frontend code and URLs.

Official SDK example (verified using `typesafe-sdk==0.7.1`):

```python
import os
from typesafe_sdk import TypeSafeClient, Choice, RetryPolicy

with TypeSafeClient(
    api_key=os.environ['JEV_API_KEY'],
    base_url='https://jev-trading-web-production.up.railway.app',
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
