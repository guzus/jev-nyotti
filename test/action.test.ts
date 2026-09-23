import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApp } from '../server/app.js';
import { readConfig } from '../server/config.js';
import { parseKraken } from '../server/market.js';
import type { ActionRequestBody, Actor } from '../server/provider.js';
import { optionsFor, type ActionName } from '../server/paper.js';

const revision='action-test-revision';
const key='test-only-not-a-deployed-secret-123456789';
// Close price rises 1 per 15m candle so paper PnL is non-trivial and deterministic.
function kraken(clock:number) {
  const boundary=Math.floor(clock/900000)*900;
  const rows=Array.from({length:101},(_,i)=>{const t=boundary-(100-i)*900;const c=String(t/900%1000+100);return [t,c,c,c,c,c,'5',3];});
  return {error:[],result:{XXBTZUSD:rows,last:boundary}};
}

test('ACTION_V1 applies each 15m cutoff once, carries the paper position and records gaps without backfill',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-action-'));
  let clock=Date.UTC(2026,8,23,0,1),lag=0;
  const bodies:ActionRequestBody[]=[];
  const plan:Record<string,ActionName>={flat:'open_long',long:'add',short:'close'};
  const actor:Actor={configured:true,act:async body=>{
    bodies.push(body);
    const names=optionsFor(body.position.side);
    const action=body.market.startsWith('Kraken BTCUSD')?plan[body.position.side]:'hold';
    return {model:'Qwen/Qwen3.5-4B',revision,task:'ACTION_V1',action,options:names.map(name=>({name,probability:1/names.length})),holdMargin:0.5,inputTokens:900,elapsedMs:3};
  }};
  const config={...readConfig(),dataDir:dir,apiKey:key,trainingStatus:'action_v1' as const,modelRevision:revision,scheduledAnalysisEnabled:true};
  const {app,store,scheduler}=createApp(config,{actor,market:async q=>parseKraken(kraken(clock-lag),q,clock-lag),now:()=>clock,
    scorer:{configured:true,score:async()=>{throw Error('legacy scorer must not run in action mode');}}});
  const server=app.listen(0,'127.0.0.1');await new Promise<void>(r=>server.once('listening',r));
  const address=server.address();assert.ok(address&&typeof address!=='string');
  const base=`http://127.0.0.1:${address.port}`;
  const post=(path:string,body:unknown)=>fetch(base+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  try {
    await scheduler.tick();
    assert.equal(bodies.length,10,'one decision per scheduled symbol on 15m only');
    const btc=bodies.find(b=>b.market==='Kraken BTCUSD spot')!;
    assert.equal(btc.candles.length,96);assert.equal(btc.cutoff,'2026-09-23T00:00:00Z');
    assert.equal((btc.candles.at(-1)!.time+900)*1000,Date.parse(btc.cutoff));
    assert.deepEqual(btc.position,{side:'flat',entry_price:null,opened_at:null,last_trade_at:null});
    await scheduler.tick();assert.equal(bodies.length,10,'same cutoff is never re-sent');

    let res=await post('/api/analyze',{symbol:'BTCUSD',interval:15});assert.equal(res.status,200);
    let decision=await res.json();
    assert.equal(decision.task,'ACTION_V1');assert.equal(decision.action,'open_long');assert.equal(decision.transfer,'venue_transfer');
    assert.equal(decision.paper.side,'long');assert.equal(decision.paper.units,1);assert.equal(decision.paper.realizedPct,-0.075);
    assert.equal(decision.actionLog.length,1);assert.equal(decision.options.length,3);
    assert.equal((await post('/api/analyze',{symbol:'BTCUSD',interval:60})).status,422);
    const eth=await (await post('/api/analyze',{symbol:'ETHUSD',interval:15})).json();
    assert.equal(eth.action,'hold');assert.equal(eth.transfer,'untested_transfer');

    lag=120000;clock+=900000;await scheduler.tick(); // provider has not published the 00:15 candle yet
    assert.equal(bodies.length,10,'no decision on an unpublished candle');
    const pending=await (await fetch(base+'/api/market?symbol=BTCUSD&interval=15')).json();
    assert.ok(pending.cache.error);assert.equal(Date.parse(pending.cache.nextRefreshAt),clock+90000);
    lag=0;clock+=90000;await scheduler.tick();
    const second=bodies.filter(b=>b.market==='Kraken BTCUSD spot').at(-1)!;
    assert.equal(second.position.side,'long');assert.equal(second.position.entry_price,btc.candles.at(-1)!.close);
    assert.equal(second.position.last_trade_at,Date.parse(btc.cutoff)/1000);

    clock+=3*900000;await scheduler.tick(); // two cutoffs missed (downtime): resume from latest
    decision=await (await post('/api/analyze',{symbol:'BTCUSD',interval:15})).json();
    assert.equal(decision.missedCutoffs,2);assert.equal(decision.actionLog.filter((r:{kind:string})=>r.kind==='gap').length,1,'a delayed candle is retried, not logged as a gap');assert.equal(decision.paper.units,3);
    assert.deepEqual(decision.actionLog.slice(0,2).map((r:{kind:string})=>r.kind),['action','gap'],'newest first: the gap precedes the resumed action');
    assert.equal(decision.actionLog.filter((r:{kind:string})=>r.kind==='action').length,3);
    assert.equal(bodies.filter(b=>b.market==='Kraken BTCUSD spot').length,3,'missed cutoffs are not backfilled');

    const paper=await (await fetch(base+'/api/paper')).json();
    assert.equal(paper.task,'ACTION_V1');assert.equal(paper.symbols.length,10);
    const row=paper.symbols.find((s:{symbol:string})=>s.symbol==='BTCUSD');
    assert.equal(row.position.units,3);assert.equal(row.lastAction.action,'add');assert.ok(row.unrealizedPct>0);
    const status=await (await fetch(base+'/api/status')).json();
    assert.equal(status.task,'ACTION_V1');assert.equal(status.providerConfigured,true);assert.deepEqual(status.decisionIntervals,[15]);
    assert.throws(()=>store.db.exec("DELETE FROM paper_actions"),/append-only/);
  }finally{await new Promise<void>((r,e)=>server.close(x=>x?e(x):r()));await scheduler.stop();store.close();rmSync(dir,{recursive:true,force:true});}
});
