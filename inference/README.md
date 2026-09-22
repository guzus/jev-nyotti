# Private Qwen 3.5 4B classifier

This GPU service supplies the Railway gateway with actual **next-token logits**
for every candidate option. It does not ask the model to invent a numeric
confidence, truncate options to an API's top-k logprobs, or place trades.
The initial model is the unmodified `Qwen/Qwen3.5-4B` post-trained checkpoint;
**no 워뇨띠 data has been acquired or used to fine-tune it**.

Base weights and tokenizer are pinned to Hugging Face revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (checked 2026-09-22).
Python 3.12, Transformers 5.17.0 and PyTorch 2.10.0 are pinned in the deployment.
CUDA defaults to BF16 on one GPU. CPU mode is available explicitly for debugging
real weights (`INFERENCE_DEVICE=cpu`), but is not a low-latency serving target.

## Internal API contract

`POST /score` requires `Authorization: Bearer <INFERENCE_API_KEY>`.
The Modal deployment additionally requires `Modal-Key` and `Modal-Secret`
proxy headers. These reject unauthenticated traffic before a GPU is started.
The key must contain at least 32 non-whitespace ASCII characters. Startup fails
before downloading weights if it is missing or invalid. Keep this credential
server-side in the Railway gateway; never place it in a browser bundle.

```json
{
  "jobs": [{
    "state": {"position": "flat", "change_1h_pct": 0.7, "volume_ratio": 1.2},
    "instructions": "Classify the supplied market snapshot. When evidence is insufficient, choose HOLD.",
    "options": [
      {"name": "LONG", "description": "Open a hypothetical long position"},
      {"name": "SHORT", "description": "Open a hypothetical short position"},
      {"name": "HOLD", "description": "Make no change"}
    ]
  }]
}
```

Response shape (numbers below illustrate the schema, not measured predictions):

```json
{
  "model": "Qwen/Qwen3.5-4B",
  "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
  "scores": [{"logits": [2.5, 0.4, 3.2], "inputTokens": 210}],
  "elapsedMs": 123.4
}
```

Scores retain the input job and option order. Each job gets an independent prompt
and forward pass, with no other question or question ID in its context. `state`
and `instructions` accept a string, JSON object, or array; descriptions additionally
accept `null`. Extra properties and duplicate option names are rejected.
Limits: 1–8 jobs, 1–255 options/job, 64 KiB body, 8,192 formatted input tokens/job.
Oversized input is rejected, never truncated. JSON non-finite numbers are rejected.
One GPU operation runs at a time, with at most eight pending HTTP requests.

`GET /healthz` bypasses the application Bearer check (Modal proxy authentication
still applies), exposes only model provenance/limits, and
reports ready only after loading. HTTP 401 means bad authentication; 413 means
body too large; 422 means invalid schema or context too long; 429 means queue
full; 503 means the model is unavailable. Responses never reflect credentials or
submitted state. Cold starts may need minutes and must be handled by the gateway.

Every option is mapped to a unique ASCII label verified to be exactly one token
after the actual chat template. Thinking is disabled. The model computes the full
vocabulary at the last input position (`logits_to_keep=1`); the service gathers
**all** mapped logits from that vector. The gateway can apply temperature/softmax.
These relative label likelihoods are neither calibrated market probabilities nor
evidence of profitable trading. Option ordering/label bias must be evaluated.
Data is JSON encoded and chat-token delimiters escaped, but a language model can
still be influenced by malicious content in state. This service executes no tools.

## Modal: scale to zero on one L4

Prerequisites: a Modal account with GPU billing enabled; a secret named
`jev-qwen-inference` containing a strong `INFERENCE_API_KEY`; outbound access to
the public Hugging Face model. No Hugging Face token is needed for the base model.
Create a proxy token with `modal workspace proxy-tokens create` and store its
pair in the gateway's `MODAL_PROXY_KEY` / `MODAL_PROXY_SECRET` environment.
Create the secret through Modal's dashboard or its local secret workflow; do not
paste credentials into source control or shell command history.

From the repository root:

```sh
uv venv inference/.venv --python 3.12
uv pip install --python inference/.venv/bin/python modal==1.5.5
inference/.venv/bin/modal setup
inference/.venv/bin/modal deploy inference/modal_app.py
```

Deployment returns a Modal HTTPS URL. The gateway's scoring endpoint is
`<that-url>/score`. Save that URL and the same private inference key in the
gateway's server-side environment. Do not give the inference key to public users.
The app sets zero minimum containers, one maximum container, and a 60-second idle
window. A persistent Modal volume caches model downloads. Startup/downloads,
inference, and warm idle time can incur charges; idle health polling can prevent
scale-to-zero. The first request downloads roughly model-sized weights and loads
the GPU; no instant cold-start promise is made.

## Docker: Novita or another NVIDIA GPU host

Requires a GPU host with NVIDIA Container Toolkit and drivers compatible with
the PyTorch 2.10 CUDA 12.8 wheel. A 24 GB L4/4090 is the initial target. Memory and
latency at the full 8,192-token limit still require measurement on the actual host.
Build from the **inference/** directory so source/credentials outside it are not
included:

```sh
docker build -t jev-qwen-inference inference/
docker run --rm --gpus all -p 8000:8000 \
  --env-file /secure/path/jev-inference.env \
  -v jev-qwen-cache:/models \
  jev-qwen-inference
```

The external env file contains `INFERENCE_API_KEY`; optional `HF_TOKEN` is only
needed for private adapters. Use the provider's secret/environment UI in hosted
deployments. Never pass secrets as image build arguments. On Novita, push this
image to an authorized private registry, select a CUDA GPU, expose HTTP port 8000,
set `/healthz` as the readiness path with a long startup allowance, and mount a
writable persistent `/models` cache if the product supports it. Registry push,
provider-specific routing/autoscaling and billing need the user's account; the
Docker definition alone does not establish a live Novita endpoint.

Run **one** Uvicorn worker per GPU. More workers duplicate model weights. This
implementation uses native Transformers/SDPA rather than vLLM because the API
needs the complete selected-label distribution, including low-ranked options.
It does not silently fall back to generated prose or a smaller model.

## Optional future LoRA

Set both `LORA_MODEL_ID=owner/private-adapter` and `LORA_REVISION=<40-char SHA>` in
the inference deployment's secrets/configuration. `HF_TOKEN` may authorize that
private repository. Only deployment configuration can select an adapter; request
bodies cannot cause arbitrary checkpoint downloads. The adapter configuration
must name `Qwen/Qwen3.5-4B` as its base. Responses append adapter ID/revision to
`revision` and health metadata changes `fineTuned` to true. This records that an
adapter was loaded, not what data trained it; data provenance remains a separate
training requirement. The pinned base revision must also match the training run.

## Checks

```sh
uv venv inference/.venv --python 3.12
uv pip install --python inference/.venv/bin/python -r inference/requirements-dev.txt
cd inference
.venv/bin/python -m pytest -q
```

Tests inject an explicit in-memory engine; no production fake-model switch exists.
They cover all-label selection, independent prompts, startup/auth failures,
schema/body/context limits, response ordering, and sanitized inference failures.
For the real pinned tokenizer without model weights:

```sh
uv pip install --python .venv/bin/python transformers==5.17.0 jinja2==3.1.6
.venv/bin/python verify_tokenizer.py
```

Before declaring the service live: deploy with credentials, verify health has the
pinned revision, submit a three-option and a 255-option request, measure cold/warm
latency and peak GPU memory, check the gateway end to end, and inspect provider
billing/scale-to-zero. A successful unit test or tokenizer check does not verify
GPU execution or trading quality.

Sources: [official checkpoint](https://huggingface.co/Qwen/Qwen3.5-4B),
[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[Modal ASGI](https://modal.com/docs/guide/webhooks),
[Modal cold starts](https://modal.com/docs/guide/cold-start).
