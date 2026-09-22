"""One manually dispatched, sanitized real-data pilot, with a $5 planning ceiling.

No deployment, production promotion, automatic retry, or inference secrets.
"""
from pathlib import Path
import json
import subprocess
import sys
import time
import uuid

import modal

if modal.is_local():
    from training.run_real import INPUT_FILES, MAX_LENGTH, MODEL_ID, MODEL_REVISION, load_dataset, valid_id
else:
    from run_real import INPUT_FILES, MAX_LENGTH, MODEL_ID, MODEL_REVISION, load_dataset, valid_id

ROOT = Path(__file__).resolve().parents[1]
# Local input validation uses the same typed inference Job as CPU/GPU workers.
sys.path.insert(0, str(ROOT / "inference"))
app = modal.App("jev-qwen-real-data-pilot")
inputs = modal.Volume.from_name("jev-qwen-real-pilot-inputs", create_if_missing=True)
artifacts = modal.Volume.from_name("jev-qwen-real-pilot-artifacts", create_if_missing=True)
cache = modal.Volume.from_name("jev-qwen-training-cache", create_if_missing=True)
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
image = image.add_local_file(str(ROOT / "training" / "run_real.py"), "/opt/training/run_real.py")
GPU_DEADLINE_SECONDS = 1800
# Published rates are an estimate, not account-wide billing enforcement.
GPU_USD_PER_SECOND = .001097
CPU_USD_PER_CORE_SECOND = .0000131
MEMORY_USD_PER_GIB_SECOND = .00000222
PLANNED_COMPUTE_CEILING_USD = (
    GPU_DEADLINE_SECONDS * (GPU_USD_PER_SECOND + 4 * CPU_USD_PER_CORE_SECOND + 32 * MEMORY_USD_PER_GIB_SECOND)
    + 900 * (2 * CPU_USD_PER_CORE_SECOND + 4 * MEMORY_USD_PER_GIB_SECOND)
)


@app.function(image=image, cpu=2, memory=4096, timeout=900, retries=0,
              max_containers=1, scaledown_window=2,
              volumes={"/training-cache": cache, "/inputs": inputs})
def prepare(dataset_id: str):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from run_real import load_dataset, valid_id
    from data import encode_example
    from jev_inference.labels import select_labels
    from jev_inference.prompt import SYSTEM_PROMPT, format_prompt
    manifest, examples = load_dataset(Path("/inputs") / valid_id(dataset_id))
    if manifest["dataset_id"] != dataset_id:
        raise ValueError("dataset directory identity mismatch")
    snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    cache.commit()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    probe = format_prompt(tokenizer, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "{}"}])
    labels = select_labels(tokenizer, probe, count=3)
    lengths = [len(encode_example(tokenizer, labels, row, MAX_LENGTH)["input_ids"])
               for split in examples.values() for row in split]
    return {"dataset_id": dataset_id, "rows": len(lengths), "min_input_tokens": min(lengths),
            "max_input_tokens": max(lengths), "model": MODEL_ID, "revision": MODEL_REVISION}


@app.function(image=image, gpu="H100!", cpu=4, memory=32768,
              timeout=1800, startup_timeout=120, retries=0, max_containers=1,
              scaledown_window=2,
              volumes={"/training-cache": cache, "/inputs": inputs, "/artifacts": artifacts})
def pilot(run_id: str, dataset_id: str, deadline_epoch: float):
    from run_real import valid_id
    run_id, dataset_id = valid_id(run_id), valid_id(dataset_id)
    remaining = min(1770, deadline_epoch - time.time() - 30)
    if remaining <= 0:
        raise TimeoutError("immutable dispatch deadline elapsed before worker execution")
    report_path = Path("/artifacts") / run_id / "report.json"
    try:
        subprocess.run([sys.executable, "/opt/training/run_real.py", run_id, dataset_id, str(deadline_epoch)],
                       check=True, timeout=remaining)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        # Return failure report for local preservation; caller still exits unsuccessfully.
        report = json.loads(report_path.read_text()) if report_path.exists() else {"run_id": run_id, "dataset_id": dataset_id}
        report.update(status="failed", worker_error_type=type(error).__name__)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        artifacts.commit()
    return json.loads(report_path.read_text())


@app.local_entrypoint()
def main(dataset_dir: str):
    # Required explicit directory, exact four-file allowlist. Never add the source folder.
    directory = Path(dataset_dir).expanduser().resolve(strict=True)
    manifest, _ = load_dataset(directory)
    dataset_id = valid_id(manifest["dataset_id"])
    assert PLANNED_COMPUTE_CEILING_USD < 5
    run_id = uuid.uuid4().hex
    output = ROOT / ".runtime" / "training" / run_id
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"run_id": run_id, "dataset_id": dataset_id,
                      "gpu_deadline_seconds": GPU_DEADLINE_SECONDS,
                      "planned_compute_ceiling_usd": round(PLANNED_COMPUTE_CEILING_USD, 4),
                      "cost_note": "Rate estimate; storage/image-build overhead additional. No automatic retry."}), flush=True)
    # Content-addressed dataset directory; a manual retry uploads the same verified files.
    with inputs.batch_upload(force=True) as batch:
        for filename in INPUT_FILES:
            batch.put_file(str(directory / filename), f"/{dataset_id}/{filename}")
    # CPU preparation is independently bounded and is terminated even on Ctrl-C.
    prepare_started = time.time()
    preparation = prepare.spawn(dataset_id)
    try:
        prepared = preparation.get(timeout=900)
        (output / "preparation.json").write_text(json.dumps(prepared, indent=2) + "\n")
        print(json.dumps({"prepared": prepared}), flush=True)
    finally:
        preparation.cancel(terminate_containers=True)
    deadline = time.time() + GPU_DEADLINE_SECONDS
    # The same absolute deadline survives infrastructure rescheduling.
    call = pilot.spawn(run_id, dataset_id, deadline)
    dispatch = {"run_id": run_id, "dataset_id": dataset_id, "call_id": call.object_id,
                "deadline_epoch": deadline, "preparation_elapsed_seconds": time.time() - prepare_started}
    (output / "dispatch.json").write_text(json.dumps(dispatch, indent=2) + "\n")
    print(json.dumps(dispatch), flush=True)
    try:
        report = call.get(timeout=max(0, deadline - time.time()))
        report["planned_compute_ceiling_usd"] = PLANNED_COMPUTE_CEILING_USD
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        (output / "caller_failure.json").write_text(json.dumps({**dispatch, "error_type": type(error).__name__}, indent=2) + "\n")
        raise
    finally:
        call.cancel(terminate_containers=True)
    summary_keys = ("run_id", "status", "completed_steps", "child_process_elapsed_seconds",
                    "baseline", "after", "reload_status", "reload_max_logit_difference",
                    "gpu_cost_estimate_usd", "planned_compute_ceiling_usd", "production_promoted")
    print(json.dumps({key: report[key] for key in summary_keys if key in report}, indent=2), flush=True)
    print(f"Local report: {output / 'report.json'}", flush=True)
    if report.get("status") != "passed":
        raise RuntimeError("pilot failed; preserved report in local runtime directory and artifact volume")
