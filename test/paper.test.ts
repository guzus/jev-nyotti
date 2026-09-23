import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { applyAction, flatPosition, optionsFor, stepPaper, unrealizedPct, type ActionName, type PaperPosition } from '../server/paper.js';

const python=process.env.ACTION_PYTHON??[resolve(import.meta.dirname,'../.runtime/action-venv/bin/python'),resolve(import.meta.dirname,'../../../../.runtime/action-venv/bin/python')].find(p=>existsSync(p))??'';
const pythonBin=existsSync(python)?python:null;

// Scripted sequence covering every transition, including add at MAX_UNITS and repeated reduce.
const script:[ActionName,number][]=[
  ['hold',100],['open_long',100],['add',110],['add',90],['add',95],['reduce',105],['reduce',104],['hold',103],['close',120],
  ['open_short',120],['add',100],['hold',101],['reduce',90],['close',130],['open_long',50],['close',40],
];
const t0=1525219200;
function runTs() {
  let p=flatPosition();const out:PaperPosition[]=[];
  script.forEach(([action,price],i)=>{p=applyAction(p,action,price,t0+i*900);out.push(p);});
  return out;
}

test('applyAction matches Python action_task.apply_action exactly on a scripted sequence',{skip:pythonBin?false:'action venv missing'},()=>{
  const code=`
import json,sys
from jev_inference.action_task import apply_action, flat_position
p=flat_position(); out=[]
for i,(a,price) in enumerate(json.loads(sys.stdin.read())):
    p=apply_action(p,a,float(price),${t0}+i*900); out.append(p)
print(json.dumps(out))`;
  const result=spawnSync(pythonBin!,['-c',code],{input:JSON.stringify(script),encoding:'utf8',
    env:{...process.env,PYTHONPATH:resolve(import.meta.dirname,'../inference')}});
  assert.equal(result.status,0,result.stderr);
  const py=JSON.parse(result.stdout) as PaperPosition[];
  const ts=runTs();
  assert.equal(py.length,ts.length);
  ts.forEach((p,i)=>{
    assert.equal(p.side,py[i].side,`side @${i}`);
    assert.equal(p.units,py[i].units,`units @${i}`);
    assert.equal(p.opened_at,py[i].opened_at,`opened_at @${i}`);
    assert.equal(p.last_trade_at,py[i].last_trade_at,`last_trade_at @${i}`);
    if(py[i].entry_price===null)assert.equal(p.entry_price,null);
    else assert.ok(Math.abs(p.entry_price!-py[i].entry_price!)<1e-9,`entry @${i}`);
  });
});

test('options depend on state and invalid actions fail closed',()=>{
  assert.deepEqual([...optionsFor('flat')],['hold','open_long','open_short']);
  assert.deepEqual([...optionsFor('long')],['hold','add','reduce','close']);
  assert.throws(()=>applyAction(flatPosition(),'add',100,t0),/not allowed/);
  assert.throws(()=>applyAction({...flatPosition(),side:'long',units:1,entry_price:100,opened_at:t0},'open_short',100,t0),/not allowed/);
  assert.throws(()=>stepPaper(flatPosition(),'open_long',0,t0),/positive/);
});

test('harmonic entry, max three units, halving and flat close',()=>{
  const ts=runTs();
  assert.equal(ts[0].last_trade_at,null);
  assert.ok(Math.abs(ts[2].entry_price!-2/(1/100+1/110))<1e-9);
  assert.equal(ts[3].units,3);
  assert.equal(ts[4].units,3);assert.equal(ts[4].entry_price,ts[3].entry_price);assert.equal(ts[4].last_trade_at,t0+4*900);
  assert.equal(ts[5].units,1.5);assert.equal(ts[6].units,0.75);
  assert.deepEqual(ts[8],{side:'flat',units:0,entry_price:null,opened_at:null,last_trade_at:t0+8*900});
});

test('stepPaper charges 7.5 bps per unit traded and realizes only closed units',()=>{
  const open=stepPaper(flatPosition(),'open_long',100,t0);
  assert.equal(open.unitsTraded,1);assert.equal(open.feePct,0.075);assert.equal(open.realizedPct,-0.075);
  const add=stepPaper(open.after,'add',100,t0+900);
  assert.equal(add.unitsTraded,1);
  const reduce=stepPaper(add.after,'reduce',110,t0+1800);
  assert.equal(reduce.unitsTraded,1);assert.ok(Math.abs(reduce.realizedPct-(10-0.075))<1e-9);
  assert.ok(Math.abs(unrealizedPct(reduce.after,110)-10)<1e-9);
  const close=stepPaper(reduce.after,'close',90,t0+2700);
  assert.ok(Math.abs(close.realizedPct-(-10-0.075))<1e-9);
  const short=stepPaper(flatPosition(),'open_short',100,t0);
  const cover=stepPaper(short.after,'close',80,t0+900);
  assert.ok(Math.abs(cover.realizedPct-(25-0.075))<1e-9); // short: entry/mark − 1
  const full=stepPaper({...add.after,units:3},'add',100,t0);
  assert.equal(full.unitsTraded,0);assert.equal(full.feePct,0);
  const hold=stepPaper(open.after,'hold',50,t0);
  assert.equal(hold.unitsTraded,0);assert.equal(hold.realizedPct,0);assert.deepEqual(hold.after,open.after);
});
