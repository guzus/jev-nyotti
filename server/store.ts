import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import type { Market } from './market.js';
import type { ScheduleRow } from './scheduler.js';
import type { Decision, TradeRequest } from './contracts.js';
import { ApiError } from './errors.js';
import type { ActionDecision, ActionLogRow } from './contracts.js';
import type { PaperPosition, PaperSide } from './paper.js';

export type PaperInventory = {position:PaperPosition;realizedPct:number;feesPct:number;trades:number;updatedCutoff:number|null;markPrice:number|null};
type InventoryRow = {symbol:string;revision:string;side:PaperSide;units:number;entry_price:number|null;opened_at:number|null;last_trade_at:number|null;
  realized_pct:number;fees_pct:number;trades:number;updated_cutoff:number;mark_price:number};
function inventoryFrom(row:InventoryRow):PaperInventory {
  return {position:{side:row.side,units:row.units,entry_price:row.entry_price,opened_at:row.opened_at,last_trade_at:row.last_trade_at},
    realizedPct:row.realized_pct,feesPct:row.fees_pct,trades:row.trades,updatedCutoff:row.updated_cutoff,markPrice:row.mark_price};
}

export class Store {
  readonly db:DatabaseSync;
  constructor(directory:string) {
    mkdirSync(directory,{recursive:true});
    this.db=new DatabaseSync(join(directory,'jev.sqlite'));
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS budget (day TEXT PRIMARY KEY, used INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS rates (bucket TEXT PRIMARY KEY, used INTEGER NOT NULL, expires INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, created INTEGER NOT NULL, payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS analysis_schedule (key TEXT PRIMARY KEY, request TEXT NOT NULL, next_due INTEGER NOT NULL, checked_at INTEGER, lease_until INTEGER NOT NULL DEFAULT 0, error TEXT, market TEXT, decision_id TEXT);
      CREATE TABLE IF NOT EXISTS analysis_worker (id INTEGER PRIMARY KEY CHECK(id=1), owner TEXT NOT NULL, lease_until INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS paper_inventory (symbol TEXT NOT NULL, revision TEXT NOT NULL,
        side TEXT NOT NULL CHECK(side IN ('flat','long','short')), units REAL NOT NULL, entry_price REAL, opened_at INTEGER, last_trade_at INTEGER,
        realized_pct REAL NOT NULL, fees_pct REAL NOT NULL, trades INTEGER NOT NULL, updated_cutoff INTEGER NOT NULL, mark_price REAL NOT NULL,
        PRIMARY KEY(symbol,revision));
      CREATE TABLE IF NOT EXISTS paper_actions (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT NOT NULL, revision TEXT NOT NULL,
        cutoff INTEGER NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('action','gap')), payload TEXT NOT NULL, created INTEGER NOT NULL,
        UNIQUE(symbol,revision,cutoff,kind));
      CREATE TRIGGER IF NOT EXISTS paper_actions_no_update BEFORE UPDATE ON paper_actions BEGIN SELECT RAISE(ABORT,'paper_actions is append-only'); END;
      CREATE TRIGGER IF NOT EXISTS paper_actions_no_delete BEFORE DELETE ON paper_actions BEGIN SELECT RAISE(ABORT,'paper_actions is append-only'); END;`);
    this.prune();
  }
  reserveEvaluations(count:number,limit:number,now=Date.now()) {
    const day=new Date(now).toISOString().slice(0,10);
    this.db.prepare('INSERT INTO budget(day,used) VALUES(?,0) ON CONFLICT DO NOTHING').run(day);
    const changed=this.db.prepare('UPDATE budget SET used=used+? WHERE day=? AND used+?<=?').run(count,day,count,limit);
    if (changed.changes!==1) throw new ApiError(429,'daily_limit','오늘의 새 분석 한도에 도달했습니다. 저장된 결과는 계속 확인할 수 있습니다.');
    // Reserve before requests. Failed upstream calls also consume this conservative cost allowance.
  }
  rateLimit(bucket:string,limit:number,now=Date.now()) {
    const window=Math.floor(now/60000);
    const id=`${window}:${bucket}`;
    const row=this.db.prepare('INSERT INTO rates(bucket,used,expires) VALUES(?,1,?) ON CONFLICT(bucket) DO UPDATE SET used=used+1 RETURNING used').get(id,now+120000) as {used:number};
    if (row.used>limit) throw new ApiError(429,'rate_limit','요청이 많습니다. 1분 뒤 다시 시도해 주세요.');
  }
  getDecision(id:string):Decision|null {
    const row=this.db.prepare('SELECT payload FROM decisions WHERE id=?').get(id) as {payload:string}|undefined;
    return row?JSON.parse(row.payload):null;
  }
  getCached(key:string):Decision|null {
    const row=this.db.prepare('SELECT payload FROM decisions WHERE cache_key=?').get(key) as {payload:string}|undefined;
    return row?{...JSON.parse(row.payload),cached:true}:null;
  }
  saveDecision(key:string,decision:Decision) {
    this.db.prepare('INSERT INTO decisions(id,cache_key,created,payload) VALUES(?,?,?,?)').run(decision.id,key,Date.now(),JSON.stringify(decision));
  }
  ensureSchedule(key:string,request:TradeRequest,now:number) {
    // Never adopt an older decision by model name alone: the scheduler key also
    // isolates task/prompt versions. The first tick can reuse its exact snapshot.
    this.db.prepare('INSERT INTO analysis_schedule(key,request,next_due) VALUES(?,?,?) ON CONFLICT DO NOTHING').run(key,JSON.stringify(request),now);
  }
  getSchedule(key:string):ScheduleRow|null {
    return (this.db.prepare('SELECT * FROM analysis_schedule WHERE key=?').get(key) as ScheduleRow|undefined)??null;
  }
  hasDueSchedule(keys:string[],now:number) {
    if(!keys.length)return false;
    return !!this.db.prepare(`SELECT 1 FROM analysis_schedule WHERE key IN (${keys.map(()=>'?').join(',')}) AND next_due<=? AND lease_until<=? LIMIT 1`).get(...keys,now,now);
  }
  acquireWorker(owner:string,now:number,leaseMs:number) {
    const result=this.db.prepare(`INSERT INTO analysis_worker(id,owner,lease_until) VALUES(1,?,?) ON CONFLICT(id) DO UPDATE SET owner=excluded.owner,lease_until=excluded.lease_until WHERE analysis_worker.lease_until<=? OR analysis_worker.owner=?`).run(owner,now+leaseMs,now,owner);
    return result.changes===1;
  }
  releaseWorker(owner:string) {this.db.prepare('DELETE FROM analysis_worker WHERE id=1 AND owner=?').run(owner);}
  claimSchedule(keys:string[],now:number,leaseMs:number,refreshMs:number):ScheduleRow|null {
    if(!keys.length)return null;
    const placeholders=keys.map(()=>'?').join(',');
    return (this.db.prepare(`UPDATE analysis_schedule SET lease_until=?,next_due=? WHERE key=(SELECT key FROM analysis_schedule WHERE key IN (${placeholders}) AND next_due<=? AND lease_until<=? ORDER BY next_due,rowid LIMIT 1) RETURNING *`).get(now+leaseMs,now+refreshMs,...keys,now,now) as ScheduleRow|undefined)??null;
  }
  saveScheduledMarket(key:string,market:Market) {
    this.db.prepare('UPDATE analysis_schedule SET market=? WHERE key=?').run(JSON.stringify(market),key);
  }
  finishSchedule(key:string,now:number,decisionId:string|null,error:string|null) {
    this.db.prepare('UPDATE analysis_schedule SET checked_at=?,lease_until=0,error=?,decision_id=COALESCE(?,decision_id) WHERE key=?').run(now,error,decisionId,key);
  }
  rescheduleSooner(key:string,at:number) {
    this.db.prepare('UPDATE analysis_schedule SET next_due=MIN(next_due,?) WHERE key=?').run(at,key);
  }
  getInventory(symbol:string,revision:string):PaperInventory|null {
    const row=this.db.prepare('SELECT * FROM paper_inventory WHERE symbol=? AND revision=?').get(symbol,revision) as InventoryRow|undefined;
    return row?inventoryFrom(row):null;
  }
  listInventories(revision:string):(PaperInventory&{symbol:string})[] {
    return (this.db.prepare('SELECT * FROM paper_inventory WHERE revision=?').all(revision) as InventoryRow[]).map(row=>({symbol:row.symbol,...inventoryFrom(row)}));
  }
  listActions(symbol:string,revision:string,limit=50):ActionLogRow[] {
    return (this.db.prepare('SELECT payload FROM paper_actions WHERE symbol=? AND revision=? ORDER BY cutoff DESC,id DESC LIMIT ?').all(symbol,revision,limit) as {payload:string}[]).map(r=>JSON.parse(r.payload));
  }
  /** Applies one ACTION_V1 cutoff atomically and at most once. `expectedCutoff`
   * is the inventory cutoff whose position was sent to the model; if another
   * writer advanced it meanwhile, nothing is written. */
  commitPaperAction(input:{symbol:string;revision:string;expectedCutoff:number|null;cutoff:number;inventory:PaperInventory;
    rows:ActionLogRow[];decisionKey:string;decision:ActionDecision}):'applied'|'duplicate' {
    const {symbol,revision,cutoff,inventory:inv}=input;
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const current=this.getInventory(symbol,revision);
      const seen=current?.updatedCutoff??null;
      if(seen!==input.expectedCutoff||(seen!==null&&cutoff<=seen)){this.db.exec('ROLLBACK');return 'duplicate';}
      const now=Date.now();
      const insert=this.db.prepare('INSERT INTO paper_actions(symbol,revision,cutoff,kind,payload,created) VALUES(?,?,?,?,?,?)');
      for(const row of input.rows)insert.run(symbol,revision,cutoff,row.kind,JSON.stringify(row),now);
      this.db.prepare('INSERT INTO decisions(id,cache_key,created,payload) VALUES(?,?,?,?)').run(input.decision.id,input.decisionKey,now,JSON.stringify(input.decision));
      const p=inv.position;
      this.db.prepare(`INSERT INTO paper_inventory(symbol,revision,side,units,entry_price,opened_at,last_trade_at,realized_pct,fees_pct,trades,updated_cutoff,mark_price)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(symbol,revision) DO UPDATE SET side=excluded.side,units=excluded.units,entry_price=excluded.entry_price,
        opened_at=excluded.opened_at,last_trade_at=excluded.last_trade_at,realized_pct=excluded.realized_pct,fees_pct=excluded.fees_pct,trades=excluded.trades,
        updated_cutoff=excluded.updated_cutoff,mark_price=excluded.mark_price`)
        .run(symbol,revision,p.side,p.units,p.entry_price,p.opened_at,p.last_trade_at,inv.realizedPct,inv.feesPct,inv.trades,cutoff,inv.markPrice);
      this.db.exec('COMMIT');
      return 'applied';
    } catch(error) {
      if(this.db.isTransaction)this.db.exec('ROLLBACK');
      throw error;
    }
  }
  prune() {
    this.db.prepare('DELETE FROM rates WHERE expires<?').run(Date.now());
    this.db.prepare('DELETE FROM decisions WHERE created<?').run(Date.now()-30*86400000);
    this.db.prepare('DELETE FROM budget WHERE day<?').run(new Date(Date.now()-35*86400000).toISOString().slice(0,10));
  }
  close() {this.db.close();}
}
