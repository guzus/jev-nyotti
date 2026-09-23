"""Download official Binance Spot USDT hourly (default) or 15-minute candles, without filling gaps.

Monthly/daily archives and REST latest candles are one coherent market source.
Output keys use application USD symbols; manifest states actual USDT quote.
--interval 15m also writes action-replay-input.json for modal_action_replay.py and
fails if any symbol lacks a single candle in [from - 96*15m, to).
"""
from __future__ import annotations
import argparse, calendar, csv, hashlib, io, json, math, subprocess, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

SYMBOLS='BTC ETH XRP SOL DOGE BNB SUI NEAR PEPE ZEC'.split()
BASE='https://data.binance.vision/data/spot'

def fetch(url):
    result=subprocess.run(['curl','-sS','-f','--retry','2','--max-time','45',url],capture_output=True)
    if result.returncode: raise RuntimeError(f'Download failed {url}: {result.stderr.decode()[:200]}')
    return result.stdout

def iso(t): return datetime.fromtimestamp(t,timezone.utc).isoformat().replace('+00:00','Z')
def epoch(s): return int(datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())
INTERVALS={'1h':3600,'15m':900}
def normalize(raw,step=3600):
    value=int(raw[0]); t=value//(1_000_000 if value>10**14 else 1000)
    o,h,l,c,v=map(float,raw[1:6])
    if t%step or not all(math.isfinite(x) for x in (o,h,l,c,v)) or min(o,h,l,c)<=0 or v<0 or h<max(o,c,l) or l>min(o,c): raise ValueError('invalid candle')
    return dict(zip(('time','open','high','low','close','volume'),(t,o,h,l,c,v)))

def download_asset(asset,start,end,directory,interval='1h'):
    step=INTERVALS[interval]
    pair=asset+'USDT'; symbol=asset+'USD'; rows={}; sources=[]; errors=[]
    now=datetime.fromtimestamp(end,timezone.utc); day=datetime.fromtimestamp(start,timezone.utc).replace(day=1,hour=0)
    urls=[]
    while day.year<now.year or (day.year==now.year and day.month<now.month):
        urls.append(f'{BASE}/monthly/klines/{pair}/{interval}/{pair}-{interval}-{day:%Y-%m}.zip')
        day=(day.replace(day=28)+timedelta(days=4)).replace(day=1)
    while day.date()<now.date():
        urls.append(f'{BASE}/daily/klines/{pair}/{interval}/{pair}-{interval}-{day:%Y-%m-%d}.zip'); day+=timedelta(days=1)
    cache=directory/'archives'; cache.mkdir(exist_ok=True)
    def add(raw):
        row=normalize(raw,step); t=row['time']
        if not start<=t or t+step>end:return
        if t in rows and rows[t]!=row:raise ValueError(f'conflicting candle {pair} {t}')
        rows[t]=row
    for url in urls:
        try:
            path=cache/url.rsplit('/',1)[-1]
            if path.exists():content=path.read_bytes()
            else:
                content=fetch(url);path.write_bytes(content)
            # Verify official per-archive SHA256 file, never merely trust ZIP CRC.
            checksum_path=path.with_suffix('.zip.CHECKSUM')
            if checksum_path.exists():checksum=checksum_path.read_bytes()
            else:checksum=fetch(url+'.CHECKSUM');checksum_path.write_bytes(checksum)
            digest=hashlib.sha256(content).hexdigest()
            if checksum.decode().split()[0]!=digest:raise ValueError('Archive SHA256 mismatch')
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                for name in archive.namelist():
                    if name.endswith('.csv'):
                        for raw in csv.reader(io.StringIO(archive.read(name).decode())):add(raw)
            sources.append({'url':url,'sha256':digest})
        except RuntimeError as error:errors.append(str(error))
    # Last 1000 candles overlap archives and cover any recent daily publishing lag.
    url=f'https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit=1000&endTime={end*1000-1}'
    content=fetch(url)
    for raw in json.loads(content):add(raw)
    sources.append({'url':url,'sha256':hashlib.sha256(content).hexdigest()})
    ordered=[rows[t] for t in sorted(rows)];gaps=[];expected=start
    for row in ordered:
        if row['time']>expected:gaps.append({'from':iso(expected),'to':iso(row['time']),'reason':'missing_source_candles'})
        expected=row['time']+step
    if expected<end:gaps.append({'from':iso(expected),'to':iso(end),'reason':'missing_source_candles'})
    body=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in ordered).encode()
    path=directory/f'{symbol}-{step//60}.jsonl';path.write_bytes(body)
    result={'symbol':symbol,'marketPair':pair,'quoteCurrency':'USDT','path':path.name,'rows':len(rows),'first':iso(ordered[0]['time']) if ordered else None,'last':iso(ordered[-1]['time']) if ordered else None,'sha256':hashlib.sha256(body).hexdigest(),'gaps':gaps,'sources':sources,'downloadErrors':errors}
    (directory/f'{symbol}.metadata.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'symbol':symbol,'rows':len(rows),'gaps':len(gaps),'errors':len(errors)}),flush=True)
    return symbol,result,ordered

LOOKBACK=96

def action_replay_input(start,end,results,retrieved_at):
    """ACTION_V1 replay input: cutoffs [start,end), each needing 96 prior candles and its execution candle."""
    first=start-LOOKBACK*900;series=[];provenance={'source':'Binance Spot','quoteCurrency':'USDT','intervalMinutes':15,'retrievedAt':retrieved_at,'syntheticFill':False,'timestampSemantics':'Candle OPEN UTC seconds, available only at time+900','symbols':{}}
    for symbol,result,ordered in results:
        candles=[row for row in ordered if first<=row['time']<end]
        if [row['time'] for row in candles]!=list(range(first,end,900)):
            raise ValueError(f'{symbol} is not contiguous over {iso(first)}..{iso(end)}; refusing to write replay input')
        series.append({'symbol':symbol,'market':f"{symbol.removesuffix('USD')}/USD",'candles':candles})
        provenance['symbols'][symbol]={k:result[k] for k in ('marketPair','rows','sha256','sources','downloadErrors')}
    return {'source':'Binance Spot 15m','from':iso(start),'to':iso(end),'series':series,'provenance':provenance}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--start',default='2026-06-20T00:00:00Z');parser.add_argument('--end');parser.add_argument('--interval',choices=sorted(INTERVALS),default='1h');parser.add_argument('--symbols',default=' '.join(SYMBOLS),help='space-separated base assets');args=parser.parse_args()
    step=INTERVALS[args.interval];assets=args.symbols.split()
    # 15m: --start/--end are the first cutoff and the exclusive last cutoff of the replay.
    floor=14400 if step==3600 else 900  # hourly default end keeps its historical 4h alignment
    start,end=epoch(args.start),epoch(args.end) if args.end else int(time.time())//floor*floor
    if start%step or end%step or start>=end or end>time.time():parser.error(f'completed UTC {args.interval} ordered boundaries required')
    if not assets or any(a not in SYMBOLS for a in assets):parser.error(f'symbols must be from {SYMBOLS}')
    fetch_start=start-LOOKBACK*step if step==900 else start
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={'source':'Binance Spot','quoteCurrency':'USDT','documentation':'https://github.com/binance/binance-public-data','range':[iso(fetch_start),iso(end)],'intervalMinutes':step//60,'timestampSemantics':f'Candle OPEN UTC seconds, available only at time+{step}','syntheticFill':False,'retrievedAt':iso(time.time()),'symbols':{}}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for symbol,result,ordered in pool.map(lambda asset:download_asset(asset,fetch_start,end,args.output_dir,args.interval),assets):
            manifest['symbols'][symbol]=result;results.append((symbol,result,ordered))
            (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if step==900:
        path=args.output_dir/'action-replay-input.json'
        path.write_text(json.dumps(action_replay_input(start,end,results,manifest['retrievedAt']),allow_nan=False)+'\n')
        print(json.dumps({'actionReplayInput':str(path)}),flush=True)
    print(json.dumps({'manifest':str(args.output_dir/'manifest.json'),'end':iso(end)}),flush=True)
if __name__=='__main__':main()
