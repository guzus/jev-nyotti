import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApp } from '../server/app.js';
import { readConfig } from '../server/config.js';
import { answerFor, jobsFor } from '../server/classifier.js';
import { parseKraken } from '../server/market.js';
import { Store } from '../server/store.js';
import type { SystemOneRequest } from '../server/contracts.js';

const model='Qwen/Qwen3.5-4B';
const key='test-only-not-a-deployed-secret-123456789';
const request:SystemOneRequest={model,state:{price:100},questions:{direction:{type:'choice',instructions:'Choose',criteria:{long:'Up',short:'Down',hold:'Flat'}}}};
function marketFixture(now=Date.now()) {
  const boundary=Math.floor(now/900000)*900;
  const rows=Array.from({length:101},(_,i)=>[boundary-(100-i)*900,'100','102','99','101','100','12',42]);
  return {error:[],result:{XXBTZUSD:rows,last:boundary}};
}

test('classifier includes every option, computes expectation, keeps transport IDs out of model input',()=>{
  const answer=answerFor(request.questions.direction,[1000,999,998]);
  assert.equal(answer.type,'choice');
  if(answer.type!=='choice')return;
  assert.equal(answer.choice,'long');
  assert.ok(Math.abs(Object.values(answer.probabilities).reduce((a,b)=>a+b,0)-1)<1e-12);
  assert.equal(Object.keys(answer.probabilities).length,3);
  const score=answerFor({type:'score',instructions:'Score',criteria:['Bad','Okay','Good']},[0,0,0]);
  assert.equal(score.type,'score');if(score.type==='score'){assert.equal(score.score,1);assert.ok(score.confidence<1e-10);}
  assert.deepEqual(answerFor({type:'noul',instructions:'True?'},[0,0]),{type:'noul',noul:0.5});
  assert.deepEqual(jobsFor(request),jobsFor({...request,questions:{arbitrary_id:request.questions.direction}}));
  assert.throws(()=>answerFor(request.questions.direction,[1,2]),/점수/);
  assert.throws(()=>answerFor(request.questions.direction,[1,NaN,2]),/점수/);
});

test('market excludes unfinished candles and rejects missing, invalid and stale prices',()=>{
  const now=Date.now(); const fixture=marketFixture(now);
  const market=parseKraken(fixture,{symbol:'BTCUSD',interval:15},now);
  assert.equal(market.candles.length,96);
  assert.ok(market.candles.at(-1)!.time<Number(fixture.result.last));
  assert.equal(market.features.rsi14,50);
  assert.equal(market.features.volumeRatio,1);
  assert.throws(()=>parseKraken(fixture,{symbol:'BTCUSD',interval:15},now+3600000),/오래/);
  const gap=structuredClone(fixture);gap.result.XXBTZUSD.splice(60,1);
  assert.throws(()=>parseKraken(gap,{symbol:'BTCUSD',interval:15},now),/누락/);
  const bad=structuredClone(fixture);bad.result.XXBTZUSD[70][4]='9999';
  assert.throws(()=>parseKraken(bad,{symbol:'BTCUSD',interval:15},now),/검증/);
});

test('daily budget remains enforced after restart and cannot partially reserve beyond limit',()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-budget-'));
  let store=new Store(dir);
  try {
    store.reserveEvaluations(2,3);assert.throws(()=>store.reserveEvaluations(2,3),/한도/);
    store.close();store=new Store(dir);store.reserveEvaluations(1,3);
    assert.throws(()=>store.reserveEvaluations(1,3),/한도/);
  }finally{store.close();rmSync(dir,{recursive:true,force:true});}
});

test('API requires auth, validates model, coalesces public analyses, persists shares, and limits paid calls',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-api-'));
  let calls=0;
  const config={...readConfig(),dataDir:dir,apiKey:key,dailyLimit:2,publicRate:1,modelRevision:'test-revision'};
  const {app,store}=createApp(config,{
    market:async q=>parseKraken(marketFixture(),q),
    scorer:{configured:true,score:async jobs=>{calls++;await new Promise(r=>setTimeout(r,15));return {model,revision:'test-revision',elapsedMs:1,scores:jobs.map(j=>({logits:j.options.map((_,i)=>i),inputTokens:10}))};}},
  });
  const server=app.listen(0,'127.0.0.1');await new Promise<void>(r=>server.once('listening',r));
  const address=server.address();assert.ok(address&&typeof address!=='string');
  const base=`http://127.0.0.1:${address.port}`;
  const post=(path:string,body:unknown,authorized=true)=>fetch(base+path,{method:'POST',headers:{'Content-Type':'application/json',...(authorized?{Authorization:`Bearer ${key}`}:{})},body:JSON.stringify(body)});
  try {
    assert.equal((await post('/v1/systemone',request,false)).status,401);
    const models=await (await fetch(base+'/v1/models',{headers:{Authorization:`Bearer ${key}`}})).json();
    assert.equal(models.models[0].name,model);assert.ok(models.models[0].description);assert.match(models.models[0].release_date,/^\d{4}-\d{2}-\d{2}$/);
    assert.equal((await post('/v1/systemone',{...request,model:'jev-1.13'})).status,422);
    assert.equal((await post('/v1/systemone',{...request,questions:{}})).status,422);
    const response=await post('/v1/systemone',request);assert.equal(response.status,200);
    const body=await response.json();assert.equal(body.model,model);assert.equal(body.answers.direction.choice,'hold');assert.equal(body.usage.input_tokens,10);assert.equal(body.usage.output_tokens,0);
    const payload={symbol:'BTCUSD',interval:15};
    const results=await Promise.all([post('/api/analyze',payload,false),post('/api/analyze',payload,false)]);
    const decisions=await Promise.all(results.map(async r=>{assert.equal(r.status,200);return r.json();}));
    assert.equal(decisions[0].id,decisions[1].id);assert.equal(calls,2);
    assert.deepEqual(decisions.map(d=>d.cached).sort(),[false,true]);
    const market=await (await fetch(base+'/api/market?symbol=BTCUSD&interval=15')).json();
    assert.equal(market.cachedDecision.id,decisions[0].id);assert.equal(market.cachedDecision.cached,true);
    assert.equal((await (await fetch(base+'/api/decisions/'+decisions[0].id)).json()).action,'hold');
    assert.equal((await post('/api/analyze',payload,false)).status,200);assert.equal(calls,2);
    assert.equal((await post('/api/analyze',{symbol:'ETHUSD',interval:15},false)).status,429);assert.equal(calls,2);
    const apiCached=await post('/v1/trading/decisions',payload);
    assert.equal(apiCached.status,200);assert.equal((await apiCached.json()).id,decisions[0].id);assert.equal(calls,2);
    assert.equal((await post('/v1/systemone',request)).status,429);assert.equal(calls,2);
    assert.equal((await fetch(base+'/v1/chat/completions')).status,404);
    const schema=await (await fetch(base+'/openapi.json')).json();assert.ok(schema.paths['/v1/systemone']);
  }finally{await new Promise<void>((r,e)=>server.close(x=>x?e(x):r()));store.close();rmSync(dir,{recursive:true,force:true});}
});

test('unconfigured model reports unavailable and never returns a fabricated decision',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-offline-'));
  const {app,store}=createApp({...readConfig(),apiKey:key,dataDir:dir},{scorer:{configured:false,score:async()=>{throw Error('must not run');}}});
  const server=app.listen(0,'127.0.0.1');await new Promise<void>(r=>server.once('listening',r));
  const address=server.address();assert.ok(address&&typeof address!=='string');
  try {
    const response=await fetch(`http://127.0.0.1:${address.port}/v1/systemone`,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${key}`},body:JSON.stringify(request)});
    assert.equal(response.status,503);assert.equal((await response.json()).error.code,'model_unconfigured');
  }finally{await new Promise<void>(r=>server.close(()=>r()));store.close();rmSync(dir,{recursive:true,force:true});}
});
