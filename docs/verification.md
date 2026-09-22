# Verification — 2026-09-22

Initial Railway deployment `4790094b-ec86-445a-a93e-ba5560840fa4` reached SUCCESS from commit `4458620`. Site: https://jev-trading-web-production.up.railway.app. Modal app: `storminggalaxys4/jev-qwen-35-4b`; model SHA `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## Observed model behavior

All observations use the actual unmodified model. These are individual samples, not latency percentiles or trading-quality evaluations.

| Request | Input tokens | Model service elapsed | Caller wall time |
|---|---:|---:|---:|
| First successful two-option request | 151 | 4.262 s | 12.015 s |
| Three options after idle | 169 | 2.360 s | 30.987 s |
| 255 options, all logits returned | 4,497 | 3.729 s | 4.827 s |
| Three options while warm | 169 | 0.101 s | 1.123 s |

Through Railway, the official `typesafe-sdk==0.7.1` successfully decoded choice/noul/score together (506 total input tokens, 3.145 s caller time). Its model-list call exposed a separate schema mismatch; `/v1/models` was corrected to the SDK's `models: [{name, description, release_date}]` shape and covered in the HTTP regression test.

The public BTC 15-minute research analysis used real Kraken candles with cutoff `2026-09-22T06:00:00Z`; gateway inference time was 62.861 s after idle. Its persisted result ID is `7a8d3ed1-5078-4e12-855a-a21c7b30f6fe`. A repeated request returned the same stored result without inference. ETH and SOL fresh analyses subsequently took 3.063 s and 1.291 s within the gateway (4.128 s and 2.271 s caller wall time). Different symbols/timestamps are not an accuracy benchmark.

## Checks and limits

- TypeScript check, production Vite/Node build and five Node tests passed: authentication, actual model identity, full probability sets, expected score, quota persistence, concurrent cache reuse, shared results, rejection of unfinished/stale/gapped/invalid prices and unconfigured inference.
- 34 inference tests passed in the inference workstream; the official tokenizer verified 255 distinct labels at real chat-template boundaries. Production has no environment-selected mock engine.
- Real Modal requests proved CUDA inference, exact model revision and all 255 scores. Unauthenticated Modal access returned 401 before reaching the function. Standard PyTorch kernels are currently used; no full-context latency or peak-memory benchmark has been run.
- Production Docker image was exercised against a root-owned temporary volume. `/proc/1/status` showed real/effective/saved UID and GID all 1000 with empty supplementary groups; SQLite files belonged to node and gateway health returned 200. This resolved the initially rejected Railway bootstrap UID setting.
- Actual website tested on desktop and mobile: real market loading, analysis loading/result, TypeSafe-only API drawer, share creation and separate shared-result navigation. No browser console errors observed in the inspected flow.
- Generated credentials were compared against tracked files: no matches. Credentials/runtime outputs are ignored and stored locally with mode 0600. Railway variables and Modal secrets hold the deployed copies.
- Editable architecture was rendered with the official Mermaid-to-Excalidraw converter, validated and visually inspected.

## Independent review reconciliation

One read-only Claude review inspected the source snapshot, including gateway, inference, frontend and deployment. It identified volume ownership (confirmed; verified privilege-dropping bootstrap), proxy-hop default (deployment explicitly uses one trusted hop), missing Modal proxy credential validation (fixed at startup), and startup timeout/quota behavior. The quota intentionally reserves **before** any paid work and counts failed calls; pre-warming before reservation was not adopted because it would allow unmetered GPU work. Live cold starts completed within the 180-second gateway limit, but transient timeouts remain possible. The website now explains that cold model startup can exceed one minute.

During these initial deployment checks, no historical trader dataset was loaded, fine-tuning performed, profitability measured or order executed. A later isolated [synthetic training rehearsal](../training/RESULTS.md) completed actual LoRA training and export/reload verification; it did not update production or use trader records. The request cap does not implement a dollar cap, and the remaining Modal credit balance was not audited. For a credit-only account policy, Modal's Usage & Billing page provides a net spend limit after credits; account-wide settings were not changed. See [Modal budgets](https://modal.com/docs/guide/budgets).
