"""Explicit-budget, resumable historical inference. Never invoked by live service."""
import json
import math
import re
import os
from pathlib import Path
import time
import modal
HERE = Path(__file__).resolve().parent
requirements = [line.strip() for name in ('requirements-api.txt','requirements-gpu.txt') for line in (HERE/name).read_text().splitlines() if line.strip() and not line.startswith(('#','-r '))] if modal.is_local() else []
cache = modal.Volume.from_name('jev-qwen-model-cache', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.12').pip_install(*requirements)
 .env({'PYTHONPATH':'/opt/inference','HF_HOME':'/models/huggingface','TOKENIZERS_PARALLELISM':'false','INFERENCE_DEVICE':'cuda','LORA_MODEL_ID':'guzus/jev-nyotti','LORA_REVISION':'73867def94f8b062700ad3f8d63128b4e1c9b1d4','LORA_SHA256':'918fdcd054e3d77116ddb7b708cc7c2a24aa443696f4639bd103408777051831'})
 .add_local_dir(str(HERE/'jev_inference'),remote_path='/opt/inference/jev_inference',ignore=['__pycache__']))

app = modal.App('jev-nyotti-historical-replay')
results = modal.Volume.from_name('jev-nyotti-replay', create_if_missing=True)
# Conservative H100 + 4 CPU + 32 GiB reservation, including 180s startup/teardown.
RATE_USD_SECOND = 0.0013
RESERVE_SECONDS = 180

@app.function(image=image,gpu='H100',cpu=4,memory=32768,volumes={'/models':cache,'/replay':results},timeout=7200,startup_timeout=120,retries=0,max_containers=1,scaledown_window=2)
def run(run_id: str, budget_usd: float, max_decisions: int):
    import threading
    if not re.fullmatch("[a-zA-Z0-9_-]{1,80}",run_id) or not math.isfinite(budget_usd) or not 0.32 <= budget_usd <= 10 or max_decisions < 1:
        raise ValueError("invalid run ID, budget, or max decisions")
    from jev_inference.engine import QwenEngine
    from jev_inference.settings import Settings
    from jev_inference.replay import ACTIONS, STEP, epoch, iso, fingerprint, index_candles, window, job
    started = time.monotonic()
    seconds = min(6900, budget_usd/RATE_USD_SECOND-RESERVE_SECONDS)
    if seconds < 60 or max_decisions < 1:
        raise ValueError('budget must reserve at least 60 inference seconds')
    # A blocking CUDA/download call cannot bypass the wall-clock spending guard.
    watchdog = threading.Timer(seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    folder = Path('/replay')/run_id
    manifest = json.loads((folder/'input.json').read_text())
    identity = fingerprint({'input':manifest,'adapter':'73867def94f8b062700ad3f8d63128b4e1c9b1d4','promptVersion':'replay-v1-hourly-hold4h'})
    start, end = epoch(manifest['from']),epoch(manifest['to'])
    if start % STEP or end % STEP or end <= start or end > time.time():
        raise ValueError('range must be completed UTC 4h boundaries')
    indexes = {s['symbol']:index_candles(s['candles']) for s in manifest['series']}
    if len(indexes) != len(manifest['series']):
        raise ValueError('duplicate symbols')
    checkpoint = folder/'checkpoint.json'
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {'identity':identity,'records':[],'completedCutoffs':0}
    if state['identity'] != identity:
        raise ValueError('immutable run input/model mismatch')
    def save():
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,allow_nan=False))
        temporary.replace(checkpoint)
        records = state['records']
        completed_end = start+state['completedCutoffs']*STEP
        output = dict(model='Qwen/Qwen3.5-4B',revision=settings.provenance_revision,generatedAt=iso(time.time()),source=manifest['source'],priorPolicy='previous_prediction',intervalMinutes=240,**{'from':iso(start),'to':iso(completed_end)},series=[])
        for symbol in indexes:
            rows = [r for r in records if r['symbol']==symbol]
            output['series'].append(dict(symbol=symbol,decisions=[{k:r[k] for k in ('marketAsOf','action','previousAction')} for r in rows],candles=[r['execution'] for r in rows]))
        (folder/'output.json').write_text(json.dumps(output,allow_nan=False))
        (folder/'status.json').write_text(json.dumps(dict(completedDecisions=len(records),completedCutoffs=state['completedCutoffs'],elapsedSeconds=time.monotonic()-started,reservedBudgetUsd=budget_usd,conservativeRateUsdSecond=RATE_USD_SECOND,identity=identity,batchGate=batch_scorer.gate)))
        results.commit()
    settings = Settings(api_key='offline-replay-no-http-endpoint-0000',device='cuda',adapter_id='guzus/jev-nyotti',adapter_revision='73867def94f8b062700ad3f8d63128b4e1c9b1d4',adapter_sha256='918fdcd054e3d77116ddb7b708cc7c2a24aa443696f4639bd103408777051831')
    engine = QwenEngine(settings)
    engine.load()
    from jev_inference.replay_batch import ReplayBatchScorer
    batch_scorer=ReplayBatchScorer()
    previous = {symbol:'flat' for symbol in indexes}
    for record in state['records']:
        if record['previousAction'] != previous[record['symbol']]:
            raise ValueError('checkpoint prior chain corrupted')
        previous[record['symbol']] = record['action']
    calls = 0
    durations = []
    for cutoff in range(start+state['completedCutoffs']*STEP,end,STEP):
        if calls+len(indexes)>max_decisions or time.monotonic()-started+max(30,sum(durations[-len(indexes):])*1.5)>seconds:
            break
        pending=[]
        prepared_inputs=[]
        executions=[]
        for symbol,index in indexes.items():
            prior,execution=window(index,cutoff)
            prepared_inputs.append(job(symbol,manifest['source'],cutoff,previous[symbol],prior))
            executions.append(execution)
        tick=time.monotonic()
        scores=batch_scorer.score(engine,engine.prepare(prepared_inputs))
        elapsed=(time.monotonic()-tick)/len(indexes)
        for (symbol,index),execution,score in zip(indexes.items(),executions,scores,strict=True):
            durations.append(elapsed)
            action=ACTIONS[max(range(3),key=lambda i:score.logits[i])]
            pending.append(dict(symbol=symbol,marketAsOf=iso(cutoff),action=action,previousAction=previous[symbol],execution=execution,logits=score.logits,inputTokens=score.inputTokens,elapsedSeconds=elapsed))
            calls+=1
        state['records'].extend(pending)
        state['completedCutoffs']+=1
        for row in pending:
            previous[row['symbol']]=row['action']
        if state['completedCutoffs']%50==0 or state['completedCutoffs']==1:
            print(json.dumps({'completedDecisions':len(state['records']),'marketThrough':iso(cutoff+STEP),'batchGate':batch_scorer.gate,'elapsedSeconds':time.monotonic()-started}),flush=True)
        save() # only complete portfolio cutoffs are published
    save()
    watchdog.cancel()
    return dict(runId=run_id,calls=calls,completedDecisions=len(state['records']),elapsedSeconds=time.monotonic()-started,meanDecisionSeconds=sum(durations)/len(durations) if durations else None)

@app.local_entrypoint()
def main(input_file: str, run_id: str, budget_usd: float, max_decisions: int = 30, resume: bool = False):
    if not re.fullmatch('[a-zA-Z0-9_-]{1,80}',run_id) or not 0.32<=budget_usd<=10 or max_decisions<1:
        raise ValueError('invalid run ID, budget (0.32..10 USD), or max decisions')
    from jev_inference.replay import STEP, epoch, index_candles, window
    manifest = json.loads(Path(input_file).read_text())
    start, end = epoch(manifest['from']), epoch(manifest['to'])
    if start % STEP or end % STEP or end <= start or end > time.time():
        raise ValueError('invalid completed 4h range')
    if not manifest['series'] or not manifest['source'].strip():
        raise ValueError('source and nonempty series are required')
    # Validate ALL source coverage locally before allocating a paid GPU.
    for series in manifest['series']:
        index = index_candles(series['candles'])
        for cutoff in range(start,end,STEP):
            window(index,cutoff)
    # Each invocation reserves its entire budget. Caller must enforce aggregate cap.
    if not resume:
        with results.batch_upload(force=False) as upload:
            upload.put_file(input_file,f'{run_id}/input.json')
    else:
        existing=b''.join(results.read_file(f'{run_id}/input.json'))
        if json.loads(existing)!=json.loads(Path(input_file).read_text()):
            raise ValueError('resume input differs')
    print(json.dumps(run.remote(run_id,budget_usd,max_decisions)))
    for name in ('output.json','status.json','checkpoint.json'):
        destination=Path(input_file).parent/f'{run_id}-{name}'
        destination.write_bytes(b''.join(results.read_file(f'{run_id}/{name}')))
        print(destination)
