# Deployment and recovery

Topology: Railway Node 24 web/API + `/data` volume; Modal L4 classifier + model-cache volume. GitHub source is public at [`guzus/jev-nyotti`](https://github.com/guzus/jev-nyotti). Raw training data and runtime credentials are excluded from git. Exchange credentials are unnecessary.

## Modal

Authenticate with `modal token new`. Create secret `jev-qwen-inference` with a random `INFERENCE_API_KEY` (32+ non-whitespace ASCII characters). Create a proxy token with `modal workspace proxy-tokens create`. Store credentials securely, never in Git or chat.

```sh
modal secret create jev-qwen-inference --from-json /private/path/inference-secret.json
modal deploy inference/modal_adapter.py
```

The endpoint requires both `Modal-Key`/`Modal-Secret` and `Authorization: Bearer <INFERENCE_API_KEY>`. Proxy auth rejects unauthenticated traffic before GPU startup. Railway's `INFERENCE_URL` is the returned origin, without `/score`. The base public weights require no HF token. First use downloads/loads weights; later starts reuse the volume. Avoid periodic Modal health requests because they prevent scale-to-zero. Railway `/healthz` never calls the GPU.

## Railway configuration

Project `jev-trading-research`, service `jev-trading-web`, production environment. GitHub source: `guzus/jev-nyotti`, branch `main` (the Railway project name is unchanged). Build the root Dockerfile. Use `railway.json`, one replica and one `/data` volume.

```text
NODE_ENV=production
PORT=3000
DATA_DIR=/data
RAILWAY_RUN_UID=0
MODEL_ID=Qwen/Qwen3.5-4B
MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a+lora:guzus/jev-nyotti@73867def94f8b062700ad3f8d63128b4e1c9b1d4
MODEL_TRAINING_STATUS=fine_tuned
MAX_DAILY_EVALUATIONS=5000
SCHEDULED_ANALYSIS_ENABLED=true
GA_MEASUREMENT_ID=G-LQLNM6WNZK
PUBLIC_REQUESTS_PER_MINUTE=3
INFERENCE_TIMEOUT_MS=300000
TRUST_PROXY_HOPS=1
INFERENCE_URL=<Modal origin>
API_KEY=<private client key>
INFERENCE_API_KEY=<Modal secret key>
MODAL_PROXY_KEY=<proxy id>
MODAL_PROXY_SECRET=<proxy secret>
```

Railway volumes are root-owned. The image defaults to `USER node`; the Railway override above permits the initial bootstrap only. `server/bootstrap.ts` creates/chowns the data directory, clears supplementary groups and irreversibly drops GID/UID to 1000 before loading the server. A real Docker test confirmed all real/effective/saved UID/GID values in `/proc/1/status` were 1000, the SQLite files belonged to node, and health succeeded. Never replace the bootstrap with a root HTTP entrypoint.

Quotas count classification questions, reserve before calls, persist across deploys and reset at UTC midnight. The IP limiter trusts one Railway edge proxy and stores HMAC identifiers. The opt-in scheduled worker checks thirty symbol/interval slots every ten minutes and reuses identical closed candles. Public analysis routes and page views only read the latest same-model result; they cannot trigger GPU calls. General HTTP read limits remain in place. Failed refreshes retain the prior decision and market snapshot. Scheduling defaults to disabled for local development; production explicitly enables it. Due times, failure cooldowns and single-worker leases persist in SQLite, so redeploying does not reset the queue. The worker is serial and may run behind its target interval during slow model calls. These are usage controls, not a dollar cap; Modal account budgets are separate.

## Verify, rollback and stop

Verify Railway SUCCESS, gateway health, actual identity, an authenticated full distribution, a real market cutoff, a stored share link and mobile/desktop browser behavior. `/api/status` reports configuration, not GPU uptime.

If inference fails, inspect Modal logs, revision and both auth layers. Never substitute mock predictions. If Railway startup fails, inspect Node version and volume permissions. Preserve the volume on redeploy; replacing it resets quotas and shares. Volumes preclude multiple active replicas and cause brief redeploy downtime.

Rollback Railway to the prior verified Git commit and redeploy the prior inference code with `modal deploy`. Verify a new checkpoint before changing the gateway revision; mismatches fail closed. Cache keys include revision and prompt version. To pause automatic analysis, set `SCHEDULED_ANALYSIS_ENABLED=false` and redeploy Railway; cached results remain readable. Authenticated `/v1/systemone` still permits explicitly requested inference. To halt all GPU usage, stop the active inference app in Modal. Shutdown stops claiming new slots and waits for the active call; forced termination leaves a persisted lease/cooldown for recovery. Model-cache storage can remain billable.

Fine-tuning requires the real licensed dataset and time-split validation. Only after verified LoRA deployment should gateway revision/provenance and `MODEL_TRAINING_STATUS` change. See `inference/README.md`.

## Trained adapter rollout

The educational website uses `guzus/jev-nyotti` pinned to Hugging Face revision `73867def94f8b062700ad3f8d63128b4e1c9b1d4`, on the pinned Qwen base above. Its full provenance string must match the gateway revision. Validate the independent Modal adapter endpoint before switching `INFERENCE_URL` and `MODEL_TRAINING_STATUS`; the gateway fails closed on revision mismatch and creates a separate scheduled cache identity. Keep the old base endpoint available for rollback, with zero minimum containers. Do not label an old shared base result as a trained prediction.
