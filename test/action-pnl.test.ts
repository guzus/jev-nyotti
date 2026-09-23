import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { importPnlReport, makePnlReport } from '../server/pnl-import.js';
import { performanceReader } from '../server/performance.js';
import { flatPosition, optionsFor, stepPaper, type ActionName } from '../server/paper.js';

const from=Date.UTC(2018,4,2)/1000;
const iso=(t:number)=>new Date(t*1000).toISOString();
// inference/REPLAY.md layout: decision i at cutoff from+i·15m executes at prices[i] (close of the
// candle ending at the cutoff); candles[i] is the execution candle opening at that cutoff, closing at prices[i+1].
function replay(prices:number[],actions:ActionName[],symbol='BTCUSD') {
  let p=flatPosition();
  const candles=actions.map((_,i)=>({time:from+i*900,open:prices[i],high:Math.max(prices[i],prices[i+1]),low:Math.min(prices[i],prices[i+1]),close:prices[i+1],volume:1}));
  const decisions=actions.map((action,i)=>{
    const names=optionsFor(p.side);
    p=stepPaper(p,action,prices[i],from+i*900).after;
    return {marketAsOf:iso(from+i*900),action,sideAfter:p.side,unitsAfter:p.units,price:prices[i],probabilities:Object.fromEntries(names.map(n=>[n,1/names.length]))};
  });
  return {symbol,decisions,candles};
}
const closes=[100,110,121,99,90,90];
const actions:ActionName[]=['open_long','add','reduce','close','hold'];
const input=(series=[replay(closes,actions)])=>({model:'Qwen/Qwen3.5-4B',revision:'jev/action@abc1234',task:'ACTION_V1',intervalMinutes:15,holdMargin:0.4,
  from:iso(from),to:iso(from+actions.length*900),source:'BitMEX XBTUSD 1m aggregated to 15m',generatedAt:iso(from+actions.length*900+60),series});

test('ACTION_V1 replay computes unit-based paper PnL with 7.5 bps per unit traded',()=>{
  const result=makePnlReport(input());
  assert.ok('task' in result&&result.task==='ACTION_V1');
  const r=result.report as ReturnType<typeof import('../server/pnl.js').simulateActionReplay>;
  const entry=2/(1/100+1/110);
  const expected=(121/entry-1)*100-3*0.075+(99/entry-1)*100-0.075; // open+add fees, reduce 1 of 2 at 121, close 1 at 99
  assert.ok(Math.abs(r.pnlPct-expected)<1e-9,`${r.pnlPct} vs ${expected}`);
  assert.ok(Math.abs(r.feesPct-0.3)<1e-12);
  const row=r.perSymbol[0];
  assert.equal(row.trades,4);assert.equal(row.opens,1);assert.equal(row.closes,1);assert.equal(row.endingSide,'flat');
  assert.equal(row.timeInPositionPct,60);
  assert.equal(r.curve.length,6);assert.equal(r.curve[0].pnlPct,0);
  assert.ok(Math.abs(r.buyHoldPct-(-10.075))<1e-9);
  assert.ok(r.maxDrawdownPts>0);
  assert.deepEqual(result.history.map(h=>h.action),['close','reduce','add','open_long'],'history holds executed actions, newest first');
});

test('ACTION_V1 replay fails closed on broken chains, wrong prices, gaps and bad probabilities',()=>{
  const bad=(mutate:(s:ReturnType<typeof replay>)=>void,pattern:RegExp)=>{const s=replay(closes,actions);mutate(s);assert.throws(()=>makePnlReport(input([s])),pattern);};
  bad(s=>{s.decisions[1].unitsAfter=3;},/paper execution rule/);
  bad(s=>{s.decisions[2].price=120;},/candle ending at the cutoff/);
  bad(s=>{s.decisions[1].marketAsOf=iso(from+2*900);},/open at its decision cutoff/);
  bad(s=>{s.candles.splice(2,1);},/one 15m candle/);
  bad(s=>{s.decisions[0].probabilities={hold:0.5,open_long:0.5};},/probabilities/);
  bad(s=>{s.decisions[1].action='open_short';},/carried state/);
  assert.throws(()=>makePnlReport({...input(),generatedAt:iso(from)}),/generatedAt/);
  assert.throws(()=>makePnlReport({...input(),feeBps:5}),/Unrecognized|unrecognized/);
});

test('ACTION_V1 report round-trips through the importer and the public performance reader',async()=>{
  const dir=mkdtempSync(join(tmpdir(),'jev-action-pnl-'));
  try {
    const file=join(dir,'input.json');
    writeFileSync(file,JSON.stringify(input([replay(closes,actions),replay([10,10,10,10,10,10],['hold','hold','open_short','hold','close'],'ETHUSD')])));
    importPnlReport(file,dir);
    const served=await performanceReader(dir)() as {task:string;report:{perSymbol:{symbol:string}[];curve:unknown[]}};
    assert.equal(served.task,'ACTION_V1');assert.equal(served.report.perSymbol.length,2);assert.equal(served.report.curve.length,6);
  } finally {rmSync(dir,{recursive:true,force:true});}
});
