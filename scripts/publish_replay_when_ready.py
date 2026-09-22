"""One-shot completion of an already-authorized replay. Never starts GPU work.

Polls only local files/Modal Volume; validates the exact run and publishes its two
public result artifacts. Refuses to overwrite intervening report edits.
"""
import argparse, hashlib, json, os, subprocess, time, urllib.request
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def command(args, root, timeout=90):
    return subprocess.run(args, cwd=root, check=True, capture_output=True, text=True, timeout=timeout).stdout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--expected-decisions',type=int,required=True)
    parser.add_argument('--timeout',type=int,default=9000)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if not args.run_id.replace('-','').replace('_','').isalnum():raise ValueError('Invalid run ID')
    requested=json.loads(args.input.read_text())
    output_dir=args.input.parent
    report_path=root/'reports/pnl-report.json'
    provenance_path=root/'reports/replay-provenance.json'
    originals={p:digest(p) for p in (report_path,provenance_path)}
    receipt=output_dir/(args.run_id+'-publication.json')
    deadline=time.monotonic()+args.timeout
    while time.monotonic()<deadline:
        try:
            status_file=output_dir/(args.run_id+'-status.json')
            command(['modal','volume','get','jev-nyotti-replay',args.run_id+'/status.json',str(status_file),'--force'],root,60)
            status=json.loads(status_file.read_text())
            print(json.dumps({'event':'progress','completed':status['completedDecisions'],'target':args.expected_decisions}),flush=True)
            if status['completedDecisions']==args.expected_decisions:break
        except (subprocess.SubprocessError,ValueError,OSError) as error:
            print(json.dumps({'event':'read_retry','error':type(error).__name__}),flush=True)
        time.sleep(60)
    else:raise TimeoutError('Replay did not fully complete; existing public report retained. No further GPU runs started.')
    output=output_dir/(args.run_id+'-output.json')
    command(['modal','volume','get','jev-nyotti-replay',args.run_id+'/output.json',str(output),'--force'],root,120)
    value=json.loads(output.read_text())
    if any(value[k]!=requested[k] for k in ('from','to','source')):raise ValueError('Replay source/range differs')
    if value['intervalMinutes']!=240 or value['priorPolicy']!='previous_prediction':raise ValueError('Replay policy differs')
    if value['revision']!='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a+lora:guzus/jev-nyotti@73867def94f8b062700ad3f8d63128b4e1c9b1d4':raise ValueError('Unexpected model')
    if {s['symbol'] for s in value['series']}!={s['symbol'] for s in requested['series']}:raise ValueError('Symbols differ')
    if sum(len(s['decisions']) for s in value['series'])!=args.expected_decisions:raise ValueError('Incomplete decisions')
    staging=output_dir/'published-report'
    command(['node','dist/server/pnl-import.js',str(output),str(staging)],root)
    generated=json.loads((staging/'pnl-report.json').read_text())
    if any(digest(p)!=old for p,old in originals.items()):raise ValueError('Public artifact changed during run; manual reconciliation required')
    if command(['git','branch','--show-current'],root).strip()!='main':raise ValueError('Not on authorized main branch')
    command(['git','fetch','origin','main'],root)
    baseline=command(['git','rev-parse','HEAD'],root).strip()
    if baseline!=command(['git','rev-parse','origin/main'],root).strip():raise ValueError('Unpublished or divergent main commits; manual reconciliation required')
    report_path.write_text(json.dumps(generated,separators=(',',':'))+'\n');os.chmod(report_path,0o644)
    metadata=json.loads(provenance_path.read_text())
    metadata.update(stage='completed',publishedRange=[value['from'],value['to']],publishedDecisions=args.expected_decisions,sourceSha256=digest(output),main=status)
    provenance_path.write_text(json.dumps(metadata,indent=2)+'\n')
    command(['git','add','--','reports/pnl-report.json','reports/replay-provenance.json'],root)
    command(['git','commit','--only','-m','Publish completed 270-day four-hour model replay','--','reports/pnl-report.json','reports/replay-provenance.json'],root)
    commit=command(['git','rev-parse','HEAD'],root).strip()
    if command(['git','rev-parse',commit+'^'],root).strip()!=baseline:raise ValueError('HEAD changed during publication')
    command(['git','push','origin',commit+':refs/heads/main'],root,120)
    receipt.write_text(json.dumps({'status':'pushed','commit':commit,'decisions':args.expected_decisions},indent=2))
    for _ in range(30):
        try:
            with urllib.request.urlopen('https://jt.memtherscan.xyz/api/performance',timeout=20) as response:live=json.load(response)
            if len(live.get('history',[]))==args.expected_decisions and live['report']['curve'][-1]['time']==generated['report']['curve'][-1]['time'] and live['report']['pnl']==generated['report']['pnl']:
                receipt.write_text(json.dumps({'status':'live_verified','commit':commit,'decisions':args.expected_decisions,'pnl':generated['report']['pnl'],'returnPct':generated['report']['returnPct']},indent=2));print(receipt.read_text(),flush=True);return
        except (ValueError,OSError,KeyError):pass
        time.sleep(30)
    raise TimeoutError('Pushed report but live verification timed out')


if __name__=='__main__':
    main()
