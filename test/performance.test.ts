import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {performanceReader} from '../server/performance.js';
test('missing replay is pending, malformed report is unavailable, not a zero return',async()=>{
 const directory=await mkdtemp(join(tmpdir(),'jev-performance-'));
 try{const read=performanceReader(directory);assert.deepEqual(await read(),{status:'pending',reason:'backfill_not_run',report:null});await writeFile(join(directory,'pnl-report.json'),'{}');await assert.rejects(read(),/성과/);await writeFile(join(directory,'pnl-report.json'),JSON.stringify({status:'ready',model:'x',revision:'x',report:{curve:[{}],perSymbol:[]}}));await assert.rejects(read(),/성과/);}finally{await rm(directory,{recursive:true,force:true});}
});

test('published importer report is served with exact accounting values',async()=>{
 const {makePnlReport}=await import('../server/pnl-import.js');
 const directory=await mkdtemp(join(tmpdir(),'jev-performance-ready-'));
 try{
 const cutoff='2026-09-01T00:00:00.000Z';
 const value=makePnlReport({model:'test',revision:'test-revision',generatedAt:'2026-09-02T00:00:00.000Z',source:'test-only',priorPolicy:'previous_prediction',series:[{symbol:'BTCUSD',decisions:[{marketAsOf:cutoff,action:'long',previousAction:'flat'}],candles:[{time:Date.parse(cutoff)/1000,open:100,high:110,low:100,close:110,volume:1}]}]});
 await writeFile(join(directory,'pnl-report.json'),JSON.stringify(value));
 const actual=await performanceReader(directory)() as typeof value;
 assert.equal(actual.report.pnl,value.report.pnl);assert.equal(actual.report.curve.at(-1)!.equity,value.report.finalEquity);
 }finally{await rm(directory,{recursive:true,force:true});}
});


test('four-hour report preserves explicit hourly-model and holding assumptions', async () => {
 const {makePnlReport}=await import('../server/pnl-import.js');
 const directory=await mkdtemp(join(tmpdir(),'jev-performance-four-hour-'));
 try {
  const cutoff='2026-09-01T00:00:00.000Z';
  const value=makePnlReport({model:'test',revision:'test',generatedAt:'2026-09-01T04:00:00Z',source:'test-only',priorPolicy:'previous_prediction',intervalMinutes:240,series:[{symbol:'BTCUSD',decisions:[{marketAsOf:cutoff,action:'flat',previousAction:'flat'}],candles:[{time:Date.parse(cutoff)/1000,open:100,high:110,low:100,close:110,volume:1}]}]});
  await writeFile(join(directory,'pnl-report.json'),JSON.stringify(value));
  const actual=await performanceReader(directory)() as typeof value;
  assert.equal(actual.report.assumptions.intervalMinutes,240);
  assert.equal(actual.report.assumptions.predictionHorizonMinutes,60);
  assert.equal(actual.report.assumptions.holdingPolicy,'hold_target_until_next_decision');
  assert.equal(actual.report.curve.at(-1)!.time,'2026-09-01T04:00:00.000Z');
 } finally { await rm(directory,{recursive:true,force:true}); }
});

test('bundled release report is used only when mutable report is absent',async()=>{
 const {makePnlReport}=await import('../server/pnl-import.js');
 const dir=await mkdtemp(join(tmpdir(),'jev-release-'));
 try{
 const cutoff='2026-09-01T00:00:00.000Z';
 const value=makePnlReport({model:'release',revision:'pinned',generatedAt:'2026-09-02T00:00:00.000Z',source:'test',priorPolicy:'previous_prediction',intervalMinutes:240,series:[{symbol:'BTCUSD',decisions:[{marketAsOf:cutoff,action:'flat',previousAction:'flat'}],candles:[{time:Date.parse(cutoff)/1000,open:100,high:110,low:90,close:100,volume:1}]}]});
 const bundled=join(dir,'bundled.json');await writeFile(bundled,JSON.stringify(value));const read=performanceReader(dir,bundled);
 assert.equal((await read() as typeof value).model,'release');
 await writeFile(join(dir,'pnl-report.json'),'{}');await assert.rejects(read(),/성과/);
 }finally{await rm(dir,{recursive:true,force:true});}
});

test('published history retains every prior/action transition in latest-first order',async()=>{
 const {makePnlReport}=await import('../server/pnl-import.js');
 const d=makePnlReport({model:'test',revision:'pinned',generatedAt:'2026-09-02T00:00:00Z',source:'Binance Spot USDT',priorPolicy:'previous_prediction',intervalMinutes:240,series:[{symbol:'BTCUSD',decisions:[{marketAsOf:'2026-09-01T00:00:00Z',action:'long',previousAction:'flat'},{marketAsOf:'2026-09-01T04:00:00Z',action:'flat',previousAction:'long'}],candles:[{time:1788220800,open:100,high:110,low:90,close:105,volume:1},{time:1788235200,open:105,high:110,low:100,close:108,volume:1}]}]});
 assert.equal(d.history.length,2);assert.equal(d.history[0].action,'flat');assert.equal(d.history[0].previousAction,'long');assert.equal(d.history[1].previousAction,'flat');
});
