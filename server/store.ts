import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import type { Decision } from './contracts.js';
import { ApiError } from './errors.js';

export class Store {
  readonly db:DatabaseSync;
  constructor(directory:string) {
    mkdirSync(directory,{recursive:true});
    this.db=new DatabaseSync(join(directory,'jev.sqlite'));
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS budget (day TEXT PRIMARY KEY, used INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS rates (bucket TEXT PRIMARY KEY, used INTEGER NOT NULL, expires INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, cache_key TEXT UNIQUE NOT NULL, created INTEGER NOT NULL, payload TEXT NOT NULL);`);
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
  prune() {
    this.db.prepare('DELETE FROM rates WHERE expires<?').run(Date.now());
    this.db.prepare('DELETE FROM decisions WHERE created<?').run(Date.now()-30*86400000);
    this.db.prepare('DELETE FROM budget WHERE day<?').run(new Date(Date.now()-35*86400000).toISOString().slice(0,10));
  }
  close() {this.db.close();}
}
