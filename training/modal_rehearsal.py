"""One manual, bounded synthetic training run. Never deploys or updates inference."""
from pathlib import Path
import json
import time
import uuid

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
app = modal.App("jev-qwen-training-rehearsal")
cache = modal.Volume.from_name("jev-qwen-training-cache", create_if_missing=True)
artifacts = modal.Volume.from_name("jev-qwen-training-rehearsals", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(
        "torch==2.8.0", "torchvision==0.23.0", "xformers==0.0.32.post2",
        "unsloth==2026.9.7", "unsloth_zoo==2026.9.6", "transformers==5.5.0",
        "trl==0.24.0", "peft==0.18.1", "torchao==0.13.0", "datasets==4.3.0",
        "accelerate==1.15.0", "pydantic==2.12.5",
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.0/causal_conv1d-1.6.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
    )
    .env({"HF_HOME": "/training-cache/huggingface", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONPATH": "/opt/inference:/opt/training", "HF_HUB_DISABLE_TELEMETRY": "1",
          "WANDB_DISABLED": "true"})
    .add_local_dir(str(ROOT / "inference" / "jev_inference"), "/opt/inference/jev_inference", ignore=["__pycache__"])
    .add_local_file(str(ROOT / "training" / "data.py"), "/opt/training/data.py")
    .add_local_file(str(ROOT / "training" / "run.py"), "/opt/training/run.py")
    .add_local_file(str(ROOT / "training" / "validation.py"), "/opt/training/validation.py")
    .add_local_file(str(ROOT / "training" / "verify.py"), "/opt/training/verify.py")
)


@app.function(image=image, cpu=2, memory=4096, timeout=900, retries=0,
              max_containers=1, volumes={"/training-cache": cache})
def prepare():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from data import encode_example, synthetic_examples
    from jev_inference.labels import select_labels
    from jev_inference.prompt import SYSTEM_PROMPT, format_prompt
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                             allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    cache.commit()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    probe = format_prompt(tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "{}"}])
    labels = select_labels(tokenizer, probe, count=3)
    rows = [encode_example(tokenizer, labels, x, 4096) for x in synthetic_examples(192, 3407)]
    return {"model": MODEL_ID, "revision": MODEL_REVISION, "cache_path": path,
            "max_input_tokens": max(len(x["input_ids"]) for x in rows)}


@app.function(image=image, gpu="H100!", cpu=4, memory=32768,
              timeout=1320, startup_timeout=120, retries=0, max_containers=1,
              scaledown_window=2,
              volumes={"/training-cache": cache, "/artifacts": artifacts})
def rehearse(run_id: str, deadline_epoch: float, verify_run_id: str = ""):
    import subprocess
    import sys
    from pathlib import Path
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("invalid run id")
    if verify_run_id and (len(verify_run_id) != 32 or any(c not in "0123456789abcdef" for c in verify_run_id)):
        raise ValueError("invalid verification source run id")
    remaining = min(1260, deadline_epoch - time.time() - 30)
    if remaining <= 0:
        raise TimeoutError("rehearsal deadline elapsed before execution")
    # Child-process timeout includes all imports, model loading, compilation and reload.
    # The fixed deadline survives infrastructure rescheduling; the caller also cancels.
    try:
        command = ([sys.executable, "/opt/training/verify.py", run_id, verify_run_id] if verify_run_id
                   else [sys.executable, "/opt/training/run.py", run_id])
        subprocess.run(command, check=True, timeout=remaining)
    finally:
        artifacts.commit()
    return json.loads((Path("/artifacts") / run_id / "report.json").read_text())


@app.local_entrypoint()
def main(gpu_budget_seconds: int = 1500, verify_run_id: str = ""):
    if not 60 <= gpu_budget_seconds <= 1500:
        raise ValueError("gpu-budget-seconds must be between 60 and 1500")
    run_id = uuid.uuid4().hex
    print(json.dumps({"run_id": run_id, "synthetic_only": True, "gpu_budget_seconds": gpu_budget_seconds,
                      "verify_run_id": verify_run_id or None}), flush=True)
    print(json.dumps(prepare.remote()), flush=True)
    deadline = time.time() + gpu_budget_seconds
    call = rehearse.spawn(run_id, deadline, verify_run_id)
    print(json.dumps({"call_id": call.object_id, "deadline_epoch": deadline}), flush=True)
    try:
        report = call.get(timeout=max(0, deadline - time.time()))
    finally:
        call.cancel(terminate_containers=True)
    output = ROOT / ".runtime" / "training" / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Local report: {output / 'report.json'}", flush=True)
