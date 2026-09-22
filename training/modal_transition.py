"""One corrective diagnostic pilot, max $2 planned compute; no production promotion."""
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
import modal
if modal.is_local():
    from training.modal_real import image, inputs, artifacts, cache, ROOT
    sys.path[:0]=[str(ROOT/'training'),str(ROOT/'inference')]
else:
    ROOT=Path('/opt')
    image=modal.Image.debian_slim(python_version='3.12')
    inputs=modal.Volume.from_name('jev-qwen-real-pilot-inputs')
    artifacts=modal.Volume.from_name('jev-qwen-real-pilot-artifacts')
    cache=modal.Volume.from_name('jev-qwen-training-cache')
from run_real import INPUT_FILES, load_dataset

app=modal.App('jev-qwen-market-only-diagnostic')
image=(image.add_local_file(str(ROOT/'training'/'transition_data.py'),'/opt/training/transition_data.py')
       .add_local_file(str(ROOT/'training'/'real_data.py'),'/opt/training/real_data.py')) if modal.is_local() else image
# 1200s worker + 120s startup + 180s teardown at conservative $0.0013/s = $1.95.
RATE=.0013
MAX_SECONDS=1200
BUDGET_USD=2.0

@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=MAX_SECONDS,startup_timeout=120,
              retries=0,max_containers=1,scaledown_window=2,
              volumes={'/training-cache':cache,'/inputs':inputs,'/artifacts':artifacts})
def pilot(run_id: str,dataset_id: str,deadline_epoch: float):
    import threading, os
    from run_real import valid_id
    run_id,dataset_id=valid_id(run_id),valid_id(dataset_id)
    remaining=min(MAX_SECONDS-30,deadline_epoch-time.time()-30)
    if remaining<=0:raise TimeoutError('dispatch deadline expired')
    timer=threading.Timer(remaining+20,lambda:os._exit(124));timer.daemon=True;timer.start()
    report_path=Path('/artifacts')/run_id/'report.json'
    try:
        subprocess.run([sys.executable,'/opt/training/run_real.py',run_id,dataset_id,str(deadline_epoch),'--market-only-v2'],check=True,timeout=remaining)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        report=json.loads(report_path.read_text()) if report_path.exists() else dict(run_id=run_id,dataset_id=dataset_id)
        report.update(status='failed',worker_error_type=type(error).__name__,production_promoted=False)
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(report,indent=2)+'\n')
    finally:
        artifacts.commit();timer.cancel()
    return json.loads(report_path.read_text())

@app.local_entrypoint()
def main(dataset_dir: str, background: bool = False):
    directory=Path(dataset_dir).resolve(strict=True)
    manifest,_=load_dataset(directory,experiment=True)
    run_id='v2-'+uuid.uuid4().hex
    output=ROOT/'.runtime'/'training'/run_id;output.mkdir()
    assert RATE*(MAX_SECONDS+120+180)<=BUDGET_USD
    with inputs.batch_upload(force=True) as batch:
        for name in INPUT_FILES:batch.put_file(str(directory/name),f"/{manifest['dataset_id']}/{name}")
    deadline=time.time()+MAX_SECONDS
    function=modal.Function.from_name('jev-qwen-market-only-diagnostic','pilot') if background else pilot
    call=function.spawn(run_id,manifest['dataset_id'],deadline)
    dispatch=dict(run_id=run_id,dataset_id=manifest['dataset_id'],call_id=call.object_id,deadline_epoch=deadline,budget_usd=BUDGET_USD)
    (output/'dispatch.json').write_text(json.dumps(dispatch,indent=2)+'\n')
    print(json.dumps(dispatch),flush=True)
    if background:
        print("Durable deployed call dispatched; save the call_id to retrieve results. No local wait or cancellation.",flush=True)
        return
    try:
        report=call.get(timeout=MAX_SECONDS)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report.get(k) for k in ('status','completed_steps','after_cohorts','promotion_gate','gpu_cost_estimate_usd','reload_status')}),flush=True)
    finally:
        call.cancel(terminate_containers=True)
