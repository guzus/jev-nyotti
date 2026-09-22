import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApp } from '../server/app.js';
import { readConfig, type Config } from '../server/config.js';
import { parseKraken } from '../server/market.js';
import type { TradeRequest } from '../server/contracts.js';
import type { Scorer } from '../server/provider.js';
import { REFRESH_INTERVAL_MS,SCHEDULED_REQUESTS } from '../server/scheduler.js';
import { Store } from '../server/store.js';

const model='Qwen/Qwen3.5-4B';
const key='test-only-not-a-deployed-secret-123456789';
const query:TradeRequest={symbol:'BTCUSD',interval:15};
function market(q:TradeRequest,now:number) {
  const boundary=Math.floor(now/(q.interval*60000))*q.interval*60;
  const rows=Array.from({length:101},(_,i)=>[boundary-(100-i)*q.interval*60,'100','102','99','101','100','12',42]);
  return parseKraken({error:[],result:{pair:rows,last:boundary}},q,now);
}
function scores(jobs:Parameters<Scorer['score']>[0]) {
  return {model,revision:'test-revision',elapsedMs:1,scores:jobs.map(j=>({logits:j.options.map((_,i)=>i),inputTokens:10}))};
}
async function start(config:Config,scorer:Scorer,now:()=>number,readMarket=(q:TradeRequest)=>Promise.resolve(market(q,now()))) {
  const {app,store,scheduler}=createApp(config,{scorer,market:readMarket,now});
  const server=app.listen(0,'127.0.0.1');
  await new Promise<void>(resolve=>server.once('listening',resolve));
  const address=server.address();assert.ok(address&&typeof address!=='string');
  const base=`http://127.0.0.1:${address.port}`;
  return {
    scheduler,store,
    async current(q=query) {
      const response=await fetch(`${base}/api/market?symbol=${q.symbol}&interval=${q.interval}`);
      assert.equal(response.status,200);assert.equal(response.headers.get('cache-control'),'no-store');
      return response.json();
    },
    post:(q=query)=>fetch(base+'/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(q)}),
    share:(id:string)=>fetch(base+'/api/decisions/'+id),
    async close() {await scheduler.stop();await new Promise<void>((resolve,reject)=>server.close(error=>error?reject(error):resolve()));store.close();},
  };
}
function config(directory:string):Config {return {...readConfig(),dataDir:directory,apiKey:key,dailyLimit:5000,modelRevision:'test-revision',scheduledAnalysisEnabled:true};}

test('all 30 slots schedule, unchanged candles reuse inference, cache survives restart and public reads never infer',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-schedule-'));
  let now=Date.UTC(2026,8,22,6,1),calls=0;
  const scorer:Scorer={configured:true,score:async jobs=>{calls++;return scores(jobs);}};
  let service=await start(config(dir),scorer,()=>now);
  try {
    const empty=await service.current();assert.equal(empty.cachedDecision,null);assert.equal(empty.cache.refreshIntervalMs,600000);
    assert.equal((await service.post()).status,202);assert.equal(calls,0);
    await service.scheduler.tick();assert.equal(calls,30);
    const original=(await service.current()).cachedDecision;assert.equal(original.cached,true);
    for(const q of SCHEDULED_REQUESTS)assert.ok(service.scheduler.view(q).cachedDecision);
    for(let i=0;i<3;i++)assert.equal((await service.post()).status,200);
    await service.scheduler.tick();assert.equal(calls,30);
    now+=REFRESH_INTERVAL_MS;await service.scheduler.tick();assert.equal(calls,30); // 06:11: no new closed15m candle
    assert.equal((await service.current()).cachedDecision.id,original.id);
    now+=REFRESH_INTERVAL_MS;await service.scheduler.tick();assert.equal(calls,40); // 06:21: ten new15m snapshots
    const current=(await service.current()).cachedDecision;assert.notEqual(current.id,original.id);
    assert.deepEqual(await (await service.share(original.id)).json(),original);
    await service.close();service=await start(config(dir),scorer,()=>now);
    await service.scheduler.tick();assert.equal(calls,40);
    assert.equal((await service.current()).cachedDecision.id,current.id);
    assert.equal((await service.post()).status,200);
    await service.close();service=await start({...config(dir),modelRevision:'new-revision'},scorer,()=>now);
    assert.equal((await service.current()).cachedDecision,null);
    assert.equal((await service.post()).status,202);assert.equal(calls,40);
    assert.equal((await (await service.share(original.id)).json()).revision,'test-revision');
  }finally{await service.close();rmSync(dir,{recursive:true,force:true});}
});

test('failed refresh retains the previous result and market through restart and respects durable cooldown',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-schedule-fail-'));
  let now=Date.UTC(2026,8,22,6,1),calls=0,modelFails=false,marketFails=false;
  const scorer:Scorer={configured:true,score:async jobs=>{calls++;if(modelFails)throw Error('test model failure');return scores(jobs);}};
  const read=async(q:TradeRequest)=>{if(marketFails)throw Error('test market failure');return market(q,now);};
  let service=await start(config(dir),scorer,()=>now,read);
  try {
    await service.scheduler.tick();const original=(await service.current()).cachedDecision;
    now+=2*REFRESH_INTERVAL_MS;modelFails=true;
    await service.scheduler.tick();assert.equal(calls,40);
    let stale=await service.current();assert.equal(stale.cachedDecision.id,original.id);assert.ok(stale.cache.error);
    assert.equal(stale.cache.nextRefreshAt,new Date(now+REFRESH_INTERVAL_MS).toISOString());
    await service.close();service=await start(config(dir),scorer,()=>now,read);
    await service.scheduler.tick();assert.equal(calls,40);
    marketFails=true;stale=await service.current();assert.equal(stale.cachedDecision.id,original.id);assert.ok(stale.candles.length);
    assert.equal((await service.post()).status,200);assert.equal(calls,40);
    now+=REFRESH_INTERVAL_MS;modelFails=false;marketFails=false;await service.scheduler.tick();
    assert.equal(calls,50);assert.equal((await service.current()).cache.error,null);
  }finally{await service.close();rmSync(dir,{recursive:true,force:true});}
});

test('worker coalesces concurrent ticks and persisted lease excludes another process; stop prevents further slots',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-schedule-concurrency-'));
  const now=Date.UTC(2026,8,22,6,1);
  let calls=0,release!:()=>void;
  const gate=new Promise<void>(resolve=>{release=resolve;});
  const scorer:Scorer={configured:true,score:async jobs=>{calls++;await gate;return scores(jobs);}};
  const first=await start(config(dir),scorer,()=>now);
  const second=await start(config(dir),scorer,()=>now);
  try {
    const one=first.scheduler.tick(),two=first.scheduler.tick();
    await new Promise(resolve=>setImmediate(resolve));assert.equal(calls,1);
    await second.scheduler.tick();assert.equal(calls,1);
    assert.equal((await first.current()).cache.refreshing,true);
    assert.equal((await first.post()).status,202);assert.equal(calls,1);
    const stopping=first.scheduler.stop();release();await Promise.all([one,two,stopping]);
    assert.equal(calls,1);await first.scheduler.tick();assert.equal(calls,1);
    await second.scheduler.tick();assert.equal(calls,30);
  }finally{release();await first.close();await second.close();rmSync(dir,{recursive:true,force:true});}
});

test('quota failures do not discard cached decisions or allow visitors to bypass budget',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-schedule-quota-'));
  let now=Date.UTC(2026,8,22,6,1),calls=0;
  const service=await start({...config(dir),dailyLimit:1},{configured:true,score:async jobs=>{calls++;return scores(jobs);}},()=>now);
  try {
    await service.scheduler.tick();assert.equal(calls,1);
    const original=(await service.current()).cachedDecision;
    assert.equal((await service.post({symbol:'ETHUSD',interval:15})).status,202);
    now+=2*REFRESH_INTERVAL_MS;await service.scheduler.tick();assert.equal(calls,1);
    const current=await service.current();assert.equal(current.cachedDecision.id,original.id);assert.match(current.cache.error,/한도/);
    assert.equal((await service.post()).status,200);assert.equal(calls,1);
    now=Date.UTC(2026,8,23,0,1);await service.scheduler.tick();assert.equal(calls,2);
  }finally{await service.close();rmSync(dir,{recursive:true,force:true});}
});

test('abandoned durable leases and cooldown survive restart before eventual recovery',()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-schedule-lease-'));
  let store=new Store(dir);
  try {
    store.ensureSchedule('slot',query,1000);
    assert.equal(store.acquireWorker('old',1000,200000),true);
    assert.ok(store.claimSchedule(['slot'],1000,200000,600000));
    store.close();store=new Store(dir);
    assert.equal(store.acquireWorker('new',2000,200000),false);
    assert.equal(store.claimSchedule(['slot'],2000,200000,600000),null);
    assert.equal(store.acquireWorker('new',201000,200000),true);
    assert.equal(store.claimSchedule(['slot'],201000,200000,600000),null);
    assert.ok(store.claimSchedule(['slot'],601000,200000,600000));
  }finally{store.close();rmSync(dir,{recursive:true,force:true});}
});

test('trained trading decisions preserve position-side labels and publish their actual revision',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-trained-cache-'));
  const revision='test-revision+lora:guzus/jev-nyotti@pinned';
  const jobsSeen:unknown[]=[];
  const now=Date.UTC(2026,8,22,6,5);
  const config={...readConfig(),dataDir:dir,apiKey:key,dailyLimit:30,scheduledAnalysisEnabled:true,trainingStatus:'fine_tuned' as const,modelRevision:revision};
  const {app,store,scheduler}=createApp(config,{now:()=>now,market:async q=>market(q,now),scorer:{configured:true,score:async jobs=>{
    jobsSeen.push(...jobs);
    for(const job of jobs) {
      assert.deepEqual(job.options.map(option=>option.name),['long','short','flat']);
      const state=job.state as Record<string,unknown>;
      assert.equal(state.position_side_before_cutoff,'flat');
      assert.match(String(state.market),/Kraken/);
      assert.match(String(job.instructions),/next hour/);
      assert.match(String(job.instructions),/out of distribution/);
    }
    return {model,revision,elapsedMs:1,scores:jobs.map(()=>({logits:[0,0,3],inputTokens:10}))};
  }}});
  try {
    await scheduler.tick();
    const decision=scheduler.view(query).cachedDecision!;
    assert.equal(jobsSeen.length,30); assert.equal(decision.action,'flat');
    assert.equal(decision.semantics,'next_hour_position_side'); assert.equal(decision.revision,revision);
    assert.deepEqual(Object.keys(decision.scores),['long','short','flat']);
    assert.equal('hold' in decision.scores,false);
  } finally {await scheduler.stop();store.close();rmSync(dir,{recursive:true,force:true});}
});
