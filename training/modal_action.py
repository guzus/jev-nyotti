"""ACTION_V1 bounded LoRA train+eval on one H100; $3 planned ceiling, no retries, no promotion.

Durable dispatch (survives the local process):
    modal deploy training/modal_action.py
    modal run training/modal_action.py --dataset-dir .runtime/action-v1 --background
Retrieve later (exit 0 while still running):
    modal run training/modal_action.py --fetch-call-id fc-... [--run-id action-...]
Without --background the ephemeral app is used and the caller waits (and cancels on exit).
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import modal

if modal.is_local():
    from training.modal_real import image, cache, ROOT
    sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'inference')]
    image = (image.add_local_file(str(ROOT / 'training' / 'run_action.py'), '/opt/training/run_action.py')
             .add_local_file(str(ROOT / 'training' / 'action_eval.py'), '/opt/training/action_eval.py'))
else:
    ROOT = Path('/opt')
    image = modal.Image.debian_slim(python_version='3.12')
    cache = modal.Volume.from_name('jev-qwen-training-cache')
from run_action import (BUDGET_USD, INPUT_FILES, MAX_SECONDS, OVERHEAD_SECONDS, RATE_USD_PER_SECOND,
                        load_action_dataset, planned_cost_usd, sha256_file, valid_id)

APP_NAME = 'jev-qwen-action-v1'
app = modal.App(APP_NAME)
inputs = modal.Volume.from_name('jev-qwen-action-v1-inputs', create_if_missing=True)
artifacts = modal.Volume.from_name('jev-qwen-action-v1-artifacts', create_if_missing=True)
STARTUP_SECONDS = 120
# Worker timeout 1800 s + 300 s startup/teardown at a conservative $0.0013/s = $2.73 <= $3.00.
assert RATE_USD_PER_SECOND == 0.0013 and OVERHEAD_SECONDS == 300 and BUDGET_USD == 3.0
assert RATE_USD_PER_SECOND * (MAX_SECONDS + OVERHEAD_SECONDS) <= BUDGET_USD


@app.function(image=image, gpu='H100', cpu=4, memory=32768, timeout=MAX_SECONDS, startup_timeout=STARTUP_SECONDS,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={'/training-cache': cache, '/inputs': inputs, '/artifacts': artifacts})
def train(run_id: str, dataset_id: str, deadline_epoch: float):
    import os
    import threading
    run_id, dataset_id = valid_id(run_id), valid_id(dataset_id)
    # Immutable dispatch deadline: rescheduling or queueing never extends paid time.
    remaining = min(MAX_SECONDS - 30, deadline_epoch - time.time() - 30)
    if remaining <= 0:
        raise TimeoutError('dispatch deadline expired before worker start')
    watchdog = threading.Timer(remaining + 20, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    report_path = Path('/artifacts') / run_id / 'report.json'
    try:
        subprocess.run([sys.executable, '/opt/training/run_action.py', run_id, dataset_id, str(deadline_epoch)],
                       check=True, timeout=remaining)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        report = json.loads(report_path.read_text()) if report_path.exists() else dict(run_id=run_id, dataset_id=dataset_id)
        report.update(status='failed', worker_error_type=type(error).__name__, production_promoted=False)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    finally:
        artifacts.commit()
        watchdog.cancel()
    return json.loads(report_path.read_text())


def main_checkout() -> Path:
    """dispatch.json/report.json go to the MAIN checkout's .runtime even from a worktree."""
    common = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                            capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


def summary(report: dict) -> dict:
    keys = ('run_id', 'status', 'phase', 'completed_steps', 'reload_status', 'adapter_sha256',
            'gpu_cost_estimate_usd', 'child_process_elapsed_seconds', 'production_promoted')
    out = {k: report.get(k) for k in keys}
    out['gate_passed'] = (report.get('gate') or {}).get('passed')
    out['lora_margin'] = (report.get('lora') or {}).get('margin')
    return out


def fetch(call_id: str, run_id: str) -> None:
    runs = main_checkout() / '.runtime' / 'training'
    if not run_id:
        matches = [p.parent.name for p in runs.glob('*/dispatch.json') if json.loads(p.read_text()).get('call_id') == call_id]
        if len(matches) != 1:
            raise SystemExit('pass --run-id: no unique local dispatch.json for this call_id')
        run_id = matches[0]
    output = runs / valid_id(run_id)
    try:
        report = modal.FunctionCall.from_id(call_id).get(timeout=5)
    except TimeoutError:
        report = None
    except Exception as error:  # remote failure or expired output: fall back to the artifact volume
        print(json.dumps({'call_error_type': type(error).__name__}), flush=True)
        report = None
    if report is None:
        try:
            report = json.loads(b''.join(artifacts.read_file(f'{run_id}/report.json')))
        except Exception:
            report = None
        if report is None or report.get('status') not in ('passed', 'failed'):
            print(json.dumps({'run_id': run_id, 'status': 'running', 'phase': (report or {}).get('phase')}), flush=True)
            return
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary(report), indent=2), flush=True)
    print(f'Local report: {output / "report.json"}', flush=True)


@app.local_entrypoint()
def main(dataset_dir: str = '', background: bool = False, fetch_call_id: str = '', run_id: str = ''):
    if fetch_call_id:
        return fetch(fetch_call_id, run_id)
    if not dataset_dir:
        raise SystemExit('--dataset-dir or --fetch-call-id is required')
    directory = Path(dataset_dir).expanduser().resolve(strict=True)
    manifest, rows, _, _ = load_action_dataset(directory)  # full validation before any upload or GPU
    dataset_id = valid_id(manifest['dataset_id'])
    function = train
    if background:
        function = modal.Function.from_name(APP_NAME, 'train')
        try:
            function.hydrate()
        except modal.exception.NotFoundError:
            raise SystemExit(f'deploy first: modal deploy training/modal_action.py ({APP_NAME} not found)') from None
    run_id = 'action-' + uuid.uuid4().hex
    output = main_checkout() / '.runtime' / 'training' / run_id
    output.mkdir(parents=True, exist_ok=False)
    # Content-addressed private input directory; the exact validated allowlist only.
    with inputs.batch_upload(force=True) as batch:
        for name in INPUT_FILES:
            batch.put_file(str(directory / name), f'/{dataset_id}/{name}')
    deadline = time.time() + MAX_SECONDS
    call = function.spawn(run_id, dataset_id, deadline)
    dispatch = dict(run_id=run_id, dataset_id=dataset_id, call_id=call.object_id, app=APP_NAME,
                    deployed=background, deadline_epoch=deadline, max_seconds=MAX_SECONDS,
                    rate_usd_per_second=RATE_USD_PER_SECOND, overhead_seconds=OVERHEAD_SECONDS,
                    planned_cost_ceiling_usd=round(planned_cost_usd(), 4), budget_usd=BUDGET_USD,
                    input_sha256={name: sha256_file(directory / name) for name in INPUT_FILES},
                    rows={s: len(r) for s, r in rows.items()})
    (output / 'dispatch.json').write_text(json.dumps(dispatch, indent=2) + '\n')
    print(json.dumps(dispatch), flush=True)
    if background:
        print('Durable deployed call dispatched; retrieve with --fetch-call-id. No local wait or cancel.', flush=True)
        return
    try:
        report = call.get(timeout=MAX_SECONDS + STARTUP_SECONDS)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(summary(report), indent=2), flush=True)
    finally:
        call.cancel(terminate_containers=True)
