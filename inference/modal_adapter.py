"""Pinned jev-nyotti canary. Deploy: modal deploy inference/modal_adapter.py."""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
requirements = [
    line.strip()
    for name in ("requirements-api.txt", "requirements-gpu.txt")
    for line in (HERE / name).read_text().splitlines()
    if line.strip() and not line.startswith(("#", "-r "))
] if modal.is_local() else []
app = modal.App("jev-nyotti")
cache = modal.Volume.from_name("jev-qwen-model-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*requirements)
    .env({
        "PYTHONPATH": "/opt/inference",
        "HF_HOME": "/models/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
        "INFERENCE_DEVICE": "cuda",
        "LORA_MODEL_ID": "guzus/jev-nyotti",
        "LORA_REVISION": "73867def94f8b062700ad3f8d63128b4e1c9b1d4",
        "LORA_SHA256": "918fdcd054e3d77116ddb7b708cc7c2a24aa443696f4639bd103408777051831",
    })
    .add_local_dir(str(HERE / "jev_inference"), remote_path="/opt/inference/jev_inference", ignore=["__pycache__"])
)


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=16384,
    min_containers=0,
    max_containers=1,
    scaledown_window=60,
    timeout=300,
    startup_timeout=1200,
    volumes={"/models": cache},
    secrets=[modal.Secret.from_name("jev-qwen-inference", required_keys=["INFERENCE_API_KEY"])],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from jev_inference.app import create_app

    return create_app()
