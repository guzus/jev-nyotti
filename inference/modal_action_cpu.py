"""CPU-only ACTION_POLICY=numeric service: /action + /healthz, no GPU, never loads Qwen.

Pins (artifact SHA-256 and hold margin) live in jev_inference/deployment.py NUMERIC_SERVING.
Deploy (the file is baked into the image; a hash mismatch refuses the deploy):
  JEV_NUMERIC_MODEL_FILE=/abs/path/numeric-policy.json modal deploy inference/modal_action_cpu.py
Same auth as modal_adapter.py: Modal proxy auth + bearer INFERENCE_API_KEY from secret jev-qwen-inference.
"""
import hashlib
import os
from pathlib import Path

import modal

from jev_inference.deployment import NUMERIC_MODEL_REMOTE_PATH, NUMERIC_SERVING, numeric_image_env

APP_NAME = 'jev-nyotti-action-cpu'
HERE = Path(__file__).resolve().parent
app = modal.App(APP_NAME)

if modal.is_local():
    env = numeric_image_env(APP_NAME)  # raises until the pins are set
    source = os.environ.get('JEV_NUMERIC_MODEL_FILE', '')
    if not source or not Path(source).is_file():
        raise SystemExit('set JEV_NUMERIC_MODEL_FILE to the exported numeric policy JSON')
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != NUMERIC_SERVING[APP_NAME]['numeric_model_sha256']:
        raise SystemExit('JEV_NUMERIC_MODEL_FILE does not match the SHA-256 pinned in deployment.py')
    requirements = [line.strip() for line in (HERE / 'requirements-api.txt').read_text().splitlines()
                    if line.strip() and not line.startswith(('#', '-r '))]
else:
    env, source, requirements = {}, '', []

image = modal.Image.debian_slim(python_version='3.12').pip_install(*requirements).env({'PYTHONPATH': '/opt/inference', **env})
if source:
    image = image.add_local_file(source, remote_path=NUMERIC_MODEL_REMOTE_PATH, copy=True)
image = image.add_local_dir(str(HERE / 'jev_inference'), remote_path='/opt/inference/jev_inference', ignore=['__pycache__'])


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    min_containers=0,
    max_containers=2,
    scaledown_window=120,
    timeout=60,
    secrets=[modal.Secret.from_name('jev-qwen-inference', required_keys=['INFERENCE_API_KEY'])],
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from jev_inference.app import create_app

    return create_app()  # Settings.from_env: ACTION_POLICY=numeric, artifact re-verified by SHA-256
