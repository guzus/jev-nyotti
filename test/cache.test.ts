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

const model='Qwen/Qwen3.5-4B';
const key='test-only-not-a-deployed-secret-123456789';
const query:TradeRequest={symbol:'BTCUSD',interval:15};
const now=Date.UTC(2026,8,22,6,5);
function market(q:TradeRequest,offset=0,close=101) {
  const boundary=Math.floor((now+offset)/(q.interval*60000))*q.interval*60;
  const rows=Array.from({length:101},(_,i)=>[boundary-(100-i)*q.interval*60,'100','102','99',String(close),'100','12',42]);
  return parseKraken({error:[],result:{pair:rows,last:boundary}},q,now+offset);
}
async function start(config:Config,scorer:Scorer,readMarket=(q:TradeRequest)=>Promise.resolve(market(q))) {
  const {app,store}=createApp(config,{scorer,market:readMarket});
  const server=app.listen(0,'127.0.0.1');
  await new Promise<void>(resolve=>server.once('listening',resolve));
  const address=server.address();assert.ok(address&&typeof address!=='string');
  const base=`http://127.0.0.1:${address.port}`;
  return {
    async current(q=query) {
      const response=await fetch(`${base}/api/market?symbol=${q.symbol}&interval=${q.interval}`);
      assert.equal(response.status,200);assert.equal(response.headers.get('cache-control'),'no-store');
      return response.json();
    },
    post:()=>fetch(base+'/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(query)}),
    share:(id:string)=>fetch(base+'/api/decisions/'+id),
    async close() {await new Promise<void>((resolve,reject)=>server.close(error=>error?reject(error):resolve()));store.close();},
  };
}

test('page views never infer; cache survives restart and quotas, with exact snapshot and model identity',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-cache-'));
  const config={...readConfig(),dataDir:dir,apiKey:key,dailyLimit:1,publicRate:1,modelRevision:'test-revision'};
  let calls=0,offset=0,close=101;
  const scorer:Scorer={configured:true,score:async jobs=>{
    calls++;
    return {model,revision:'test-revision',elapsedMs:1,scores:jobs.map(j=>({logits:j.options.map((_,i)=>i),inputTokens:10}))};
  }};
  const readMarket=async(q:TradeRequest)=>market(q,offset,close);
  let service=await start(config,scorer,readMarket);
  try {
    assert.equal((await service.current()).cachedDecision,null);assert.equal(calls,0);
    const response=await service.post();assert.equal(response.status,200);
    const original=await response.json();assert.equal(original.cached,false);assert.equal(calls,1);
    const cached=await service.current();
    assert.deepEqual(cached.cachedDecision,{...original,cached:true});
    for(let i=0;i<4;i++) {
      const hit=await service.post();assert.equal(hit.status,200);
      assert.deepEqual(await hit.json(),{...original,cached:true});
    }
    assert.equal(calls,1);
    assert.equal((await service.current({symbol:'ETHUSD',interval:15})).cachedDecision,null);
    assert.equal((await service.current({symbol:'BTCUSD',interval:60})).cachedDecision,null);
    close=100.5;assert.equal((await service.current()).cachedDecision,null);close=101;
    offset=900000;assert.equal((await service.current()).cachedDecision,null);
    assert.equal((await service.post()).status,429);assert.equal(calls,1);offset=0;
    await service.close();
    service=await start(config,{configured:false,score:async()=>{throw Error('cached reads must not score');}},readMarket);
    assert.deepEqual((await service.current()).cachedDecision,{...original,cached:true});
    assert.equal((await service.post()).status,200);
    assert.equal((await (await service.share(original.id)).json()).id,original.id);
    await service.close();
    service=await start({...config,modelRevision:'new-revision'},scorer,readMarket);
    assert.equal((await service.current()).cachedDecision,null);
    // Old shares remain addressable; they are never presented as a current-model cache hit.
    assert.equal((await (await service.share(original.id)).json()).revision,'test-revision');
    await service.close();
    service=await start({...config,trainingStatus:'fine_tuned'},scorer,readMarket);
    assert.equal((await service.current()).cachedDecision,null);
    assert.equal(calls,1);
  }finally{await service.close();rmSync(dir,{recursive:true,force:true});}
});

test('failed inference is not cached and an explicit retry can populate the cache',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-cache-retry-'));
  let calls=0;
  const service=await start({...readConfig(),dataDir:dir,apiKey:key,dailyLimit:3,publicRate:60,modelRevision:'test-revision'},{
    configured:true,score:async jobs=>{
      calls++;
      if(calls===1) throw Error('test inference failure');
      return {model,revision:'test-revision',elapsedMs:1,scores:jobs.map(j=>({logits:j.options.map((_,i)=>i),inputTokens:10}))};
    },
  });
  try {
    assert.equal((await service.post()).status,500);
    assert.equal((await service.current()).cachedDecision,null);
    const retried=await service.post();assert.equal(retried.status,200);
    const decision=await retried.json();assert.equal(calls,2);
    assert.equal((await service.current()).cachedDecision.id,decision.id);
    assert.equal((await service.post()).status,200);assert.equal(calls,2);
  }finally{await service.close();rmSync(dir,{recursive:true,force:true});}
});
