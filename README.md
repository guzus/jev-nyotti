# jev뇨띠 · jev-nyotti

[Live website](https://jevtrade.up.railway.app) · [Hugging Face adapter](https://huggingface.co/guzus/jev-nyotti) · [GitHub](https://github.com/guzus/jev-nyotti) · [API schema](https://jevtrade.up.railway.app/openapi.json)

Qwen3.5-4B 실제 logits를 사용하는 **TypeSafe/Jev 호환 API**와 한국어 시장 분석 웹사이트. Railway는 웹·API를, Modal L4는 모델을 서빙합니다. 현재 공개 서비스는 추가 학습 전 **기본 모델**을 사용합니다. 사용자가 제공한 AOA 매매내역으로 별도의 [BTC 포지션 모방 파일럿](training/REAL_DATA.md)을 진행하며, 이 어댑터는 공개 서비스에 적용하지 않습니다. 원본의 거래자 신원은 독립적으로 검증하지 않았습니다. 주문 실행 기능은 없습니다.

![Runtime architecture](diagrams/runtime.png)

[Editable Excalidraw](diagrams/runtime.excalidraw) · [Mermaid](diagrams/runtime.mmd) · [Evidence](diagrams/runtime.evidence.json)

## Run locally

Node.js 24+ is required. Copy `.env.example` to ignored `.env`; set a random 32+ character `API_KEY`. Set the Modal URL, inference key and proxy token pair to enable the real model. No credentials enter the browser bundle.

```sh
npm ci
npm run build
node --env-file=.env dist/server/index.js
```

Without an inference endpoint, the site shows actual market data and a model-unavailable state. It never synthesizes an analysis. For frontend development, export the environment and run `npm run dev`.

## Interfaces

| Interface | Purpose |
|---|---|
| `POST /v1/systemone` | Bearer-authenticated choice, score and noul classification |
| `POST /v1/trading/decisions` | Bearer-authenticated market research stance |
| `GET /v1/models` | Actual model identity |
| `GET /openapi.json` | API discovery and request schemas |
| Website `/` | Global-volume top 10 snapshot × 15m/1h/4h charts and shareable decisions |
| `GET /healthz` | Gateway liveness without waking the GPU |

Use `Qwen/Qwen3.5-4B` as the client model. This preserves TypeSafe's wire format, not Jev's weights or calibration. Chat completions are not implemented. See [API semantics and examples](docs/api.md).

## Model and operations

- Official pinned revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Modal L4: 0 minimum / 1 maximum container, 60-second idle scale-down; model cache persists. Cold-start latency includes loading weights.
- Every candidate is scored through verified single-token labels. Conditional softmax scores are **not calibrated probabilities of returns**. Confidence is 1 minus normalized entropy.
- Actual Kraken closed candles only; stale, invalid and gapped data is rejected. No order execution, exchange credentials or trader history.
- SQLite on Railway retains immutable shares for 30 days, daily quotas and the refresh queue across deploys. With `SCHEDULED_ANALYSIS_ENABLED=true`, one worker checks all ten displayed coins × three intervals every ten minutes, reuses identical closed-candle snapshots, and retains the last result when refresh fails. Slow upstream calls can delay the serial queue. Page views and public analysis requests never start inference. Keep one Railway replica.
- Scheduled production cap: 5,000 classification questions/day, shared with authenticated `/v1/systemone` requests. Failed model calls also consume quota. Thirty scheduled checks every ten minutes yield at most 4,320 checks/day; unchanged candles skip inference (normally about 1,260 new snapshots/day across these intervals). HTTP reads have separate rate limits. This is **not a dollar spending cap**; GPU idle time, CPU/memory and storage also consume credit, and Railway billing is separate.

[Training: real-data pilot and rehearsal](training/README.md) · [Deployment and recovery](docs/deployment.md) · [Inference service](inference/README.md) · [Future dataset/evaluation plan](docs/experiment.md)

## Verify

```sh
npm run typecheck
npm run build
npm test
uv venv inference/.venv
uv pip install --python inference/.venv/bin/python -r inference/requirements-dev.txt
PYTHONPATH=inference inference/.venv/bin/pytest inference/tests
```

Tests inject explicit test engines; production has no mock-model setting. Deployment checks separately verify real model revision/logits, market cutoffs, share links and browser behavior.

Earlier [vendor comparisons](docs/provider-landscape.md) and [serving research](docs/serving.md) predate the selected **Qwen3.5-4B + Modal** configuration. They are historical research, not current deployment instructions or approval to purchase training.

The coin picker uses the [dated CoinGecko global 24-hour USD volume snapshot](docs/volume-ranking.json), excluding its stablecoin category: BTC, ETH, XRP, SOL, DOGE, BNB, SUI, NEAR, PEPE, ZEC. This is a fixed snapshot with its timestamp visible in the UI, not an automatically refreshed ranking. Chart and inference inputs remain Kraken USD spot candles. Previously shared ADA/AVAX/LINK/DOT/LTC results remain supported.
