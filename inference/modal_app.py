"""Deploy from repository root: modal deploy inference/modal_app.py.

Requires an existing Modal secret named jev-qwen-inference containing
INFERENCE_API_KEY. No credentials or model weights are embedded in the image.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
# Modal copies only the named requirements file; flatten our two pinned files
# so its remote builder never sees an unresolved local `-r` include.
requirements = [
    line.strip()
    for name in ("requirements-api.txt", "requirements-gpu.txt")
    for line in (HERE / name).read_text().splitlines()
    if line.strip() and not line.startswith(("#", "-r "))
] if modal.is_local() else []
app = modal.App("jev-qwen-35-4b")
cache = modal.Volume.from_name("jev-qwen-model-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*requirements)
    .env({
        "PYTHONPATH": "/opt/inference",
        "HF_HOME": "/models/huggingface",
        "TOKENIZERS_PARALLELISM": "false",
        "INFERENCE_DEVICE": "cuda",
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
