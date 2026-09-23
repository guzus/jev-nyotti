"""Explicit-budget, resumable ACTION_V1 stateful replay. Never invoked by the live service.

modal run inference/modal_action_replay.py --input-file IN.json --run-id ID --budget-usd 1 \
  --adapter-id OWNER/REPO --adapter-revision SHA40 --adapter-sha256 SHA64 --hold-margin M
Pure logic lives in jev_inference/action_replay.py; this file owns budget, watchdog and volume I/O.
"""
import json
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
requirements = [line.strip() for name in ('requirements-api.txt', 'requirements-gpu.txt')
                for line in (HERE / name).read_text().splitlines()
                if line.strip() and not line.startswith(('#', '-r '))] if modal.is_local() else []
cache = modal.Volume.from_name('jev-qwen-model-cache', create_if_missing=True)
# No adapter pin in the image: the adapter is a run parameter bound into the run identity.
image = (modal.Image.debian_slim(python_version='3.12').pip_install(*requirements)
         .env({'PYTHONPATH': '/opt/inference', 'HF_HOME': '/models/huggingface',
               'TOKENIZERS_PARALLELISM': 'false', 'INFERENCE_DEVICE': 'cuda'})
         .add_local_dir(str(HERE / 'jev_inference'), remote_path='/opt/inference/jev_inference', ignore=['__pycache__']))

app = modal.App('jev-action-replay')
results = modal.Volume.from_name('jev-action-replay', create_if_missing=True)
# Conservative H100 + 4 CPU + 32 GiB reservation, including 180s startup/teardown.
RATE_USD_SECOND = 0.0013
RESERVE_SECONDS = 180
RUN_ID = re.compile('[a-zA-Z0-9_-]{1,80}')


def settings_for(adapter_id: str, adapter_revision: str, adapter_sha256: str, hold_margin: float):
    """Validated offline Settings; an adapter must be fully pinned (id, revision and sha256)."""
    from jev_inference.settings import Settings
    if adapter_id and not adapter_sha256:
        raise ValueError('an adapter replay requires --adapter-sha256')
    return Settings(api_key='offline-replay-no-http-endpoint-0000', device='cuda', adapter_id=adapter_id or None,
                    adapter_revision=adapter_revision or None, adapter_sha256=adapter_sha256 or None,
                    action_hold_margin=hold_margin)


@app.function(image=image, gpu='H100', cpu=4, memory=32768, volumes={'/models': cache, '/replay': results},
              timeout=7200, startup_timeout=120, retries=0, max_containers=1, scaledown_window=2)
def run(run_id: str, budget_usd: float, max_decisions: int, adapter_id: str, adapter_revision: str,
        adapter_sha256: str, hold_margin: float):
    import os
    import threading
    from jev_inference import action_replay as ar
    from jev_inference.action_api import to_scoring_job
    from jev_inference.engine import QwenEngine
    from jev_inference.settings import MODEL_ID
    from jev_inference.replay import iso
    if not RUN_ID.fullmatch(run_id) or max_decisions < 1:
        raise ValueError('invalid run ID or max decisions')
    started = time.monotonic()
    seconds = ar.budget_seconds(budget_usd, RATE_USD_SECOND, RESERVE_SECONDS)
    # A blocking CUDA/download call cannot bypass the wall-clock spending guard.
    watchdog = threading.Timer(seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    settings = settings_for(adapter_id, adapter_revision, adapter_sha256, hold_margin)
    folder = Path('/replay') / run_id
    manifest = json.loads((folder / 'input.json').read_text())
    plan = ar.Plan(manifest, now=time.time())
    identity = ar.identity(manifest, adapter_id=adapter_id, adapter_revision=adapter_revision,
                           adapter_sha256=adapter_sha256, base_revision=settings.revision, hold_margin=hold_margin)
    checkpoint = folder / 'checkpoint.json'
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else ar.new_state(identity)
    if state['identity'] != identity:
        raise ValueError('immutable run input/model/margin mismatch')
    ar.restore_positions(plan, state, hold_margin)  # fail before loading the model
    engine = QwenEngine(settings)
    engine.load()
    durations: list[float] = []

    def score(jobs):
        tick = time.monotonic()
        # Serial engine scoring: identical to live /action (batched padding could flip near-ties and
        # then change every later carried position of a stateful replay).
        scores = engine.score(engine.prepare([to_scoring_job(j) for j in jobs]))
        durations.append(time.monotonic() - tick)
        return scores

    def may_continue():
        return time.monotonic() - started + max(30, 1.5 * (durations[-1] if durations else 0)) <= seconds

    def save():
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, allow_nan=False))
        temporary.replace(checkpoint)
        output = ar.output(plan, state, model=MODEL_ID, revision=settings.provenance_revision,
                           margin=hold_margin, generated_at=time.time())
        (folder / 'output.json').write_text(json.dumps(output, allow_nan=False))
        (folder / 'status.json').write_text(json.dumps(dict(
            completedDecisions=len(state['records']), completedCutoffs=state['completedCutoffs'],
            marketThrough=output['to'], elapsedSeconds=time.monotonic() - started, reservedBudgetUsd=budget_usd,
            conservativeRateUsdSecond=RATE_USD_SECOND, identity=identity, contractSha256=ar.CONTRACT_SHA256,
            scoring='serial')))
        results.commit()
        if state['completedCutoffs'] % 50 == 1:
            print(json.dumps(dict(completedDecisions=len(state['records']), marketThrough=output['to'],
                                  scoring='serial', generatedAt=iso(time.time()))), flush=True)

    calls = ar.run(plan, state, score, hold_margin, max_decisions=max_decisions,
                   may_continue=may_continue, on_cutoff=save)
    save()
    watchdog.cancel()
    return dict(runId=run_id, calls=calls, completedDecisions=len(state['records']),
                completedCutoffs=state['completedCutoffs'], elapsedSeconds=time.monotonic() - started,
                meanCutoffSeconds=sum(durations) / len(durations) if durations else None)


@app.local_entrypoint()
def main(input_file: str, run_id: str, budget_usd: float, adapter_id: str = '', adapter_revision: str = '',
         adapter_sha256: str = '', hold_margin: float = 0.0, max_decisions: int = 30, resume: bool = False):
    from jev_inference import action_replay as ar
    if not RUN_ID.fullmatch(run_id) or max_decisions < 1:
        raise ValueError('invalid run ID or max decisions')
    ar.budget_seconds(budget_usd, RATE_USD_SECOND, RESERVE_SECONDS)
    settings = settings_for(adapter_id, adapter_revision, adapter_sha256, hold_margin)
    manifest = json.loads(Path(input_file).read_text())
    # Validate ALL coverage (96 contiguous prior + execution candle per cutoff) before a paid GPU.
    plan = ar.Plan(manifest, now=time.time())
    print(json.dumps(dict(symbols=plan.symbols, cutoffs=len(plan.cutoffs()), revision=settings.provenance_revision,
                          holdMargin=hold_margin, maxDecisions=max_decisions)))
    # Each invocation reserves its entire budget. Caller must enforce any aggregate cap.
    if not resume:
        with results.batch_upload(force=False) as upload:
            upload.put_file(input_file, f'{run_id}/input.json')
    elif json.loads(b''.join(results.read_file(f'{run_id}/input.json'))) != manifest:
        raise ValueError('resume input differs')
    print(json.dumps(run.remote(run_id, budget_usd, max_decisions, adapter_id, adapter_revision,
                                adapter_sha256, hold_margin)))
    for name in ('output.json', 'status.json', 'checkpoint.json'):
        destination = Path(input_file).parent / f'{run_id}-{name}'
        destination.write_bytes(b''.join(results.read_file(f'{run_id}/{name}')))
        print(destination)
