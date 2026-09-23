import { createHash, randomUUID } from 'node:crypto';
import { actionTarget, type Config } from './config.js';
import { ANALYSIS_CACHE_VERSION,type Decision,type TradeRequest } from './contracts.js';
import type { Market } from './market.js';
import { ApiError } from './errors.js';
import type { Store } from './store.js';
import { nextActionDue } from './action.js';

export const ACTION_REFRESH_INTERVAL_MS=900000;
/** One bounded early retry inside the same 15m window after a failed ACTION_V1 call. */
export const ACTION_RETRY_MS=300000;
/** Short retry while the data provider has not yet published the just-closed candle. */
export const ACTION_CANDLE_RETRY_MS=90000;

export const REFRESH_INTERVAL_MS=600000;
export const SCHEDULED_SYMBOLS:TradeRequest['symbol'][]=['BTCUSD','ETHUSD','XRPUSD','SOLUSD','DOGEUSD','BNBUSD','SUIUSD','NEARUSD','PEPEUSD','ZECUSD'];
export const SCHEDULED_REQUESTS:TradeRequest[]=SCHEDULED_SYMBOLS.flatMap(symbol=>([15,60,240] as const).map(interval=>({symbol,interval})));
export type CacheStatus={refreshIntervalMs:number;checkedAt:string|null;nextRefreshAt:string|null;refreshing:boolean;error:string|null};
export type ScheduleRow={key:string;request:string;next_due:number;checked_at:number|null;lease_until:number;error:string|null;market:string|null;decision_id:string|null};

export function createScheduler(config:Config,store:Store,refresh:(request:TradeRequest,saveMarket:(market:Market)=>void)=>Promise<Decision>,now=Date.now) {
  const owner=randomUUID();
  const action=config.trainingStatus==='action_v1'?actionTarget(config):null;
  const identity={model:action?.model??config.modelId,revision:action?.revision??config.modelRevision,trainingStatus:config.trainingStatus};
  const keyFor=(request:TradeRequest)=>createHash('sha256').update(JSON.stringify({version:ANALYSIS_CACHE_VERSION,...identity,...request})).digest('hex');
  const actionMode=config.trainingStatus==='action_v1';
  // ACTION_V1 decides only on 15m candles. 1h/4h rows stay so market views keep their persisted fallback.
  const claimable=(request:TradeRequest)=>!actionMode||request.interval===15;
  const keys=SCHEDULED_REQUESTS.filter(claimable).map(request=>keyFor(request));
  for(const request of SCHEDULED_REQUESTS)store.ensureSchedule(keyFor(request),request,now());
  const refreshMs=()=>actionMode?nextActionDue(now())-now():REFRESH_INTERVAL_MS;
  const leaseMs=config.inferenceTimeout+90000; // market timeout + inference timeout + durable-write margin
  let running:Promise<void>|null=null;
  let timer:ReturnType<typeof setTimeout>|undefined;
  let stopped=false;
  let started=false;
  function view(request:TradeRequest) {
    const row=store.getSchedule(keyFor(request));
    const decision=row?.decision_id?store.getDecision(row.decision_id):null;
    return {
      cachedDecision:decision?{...decision,cached:true}:null,
      market:row?.market?JSON.parse(row.market) as Market:null,
      cache:{refreshIntervalMs:actionMode?ACTION_REFRESH_INTERVAL_MS:REFRESH_INTERVAL_MS,
        checkedAt:row?.checked_at===null||row?.checked_at===undefined?null:new Date(row.checked_at).toISOString(),
        nextRefreshAt:row&&config.scheduledAnalysisEnabled&&claimable(request)?new Date(Math.max(row.next_due,row.lease_until)).toISOString():null,
        refreshing:!!row&&row.lease_until>now(),error:row?.error??null} satisfies CacheStatus,
    };
  }
  async function work() {
    if(stopped||!config.scheduledAnalysisEnabled||!store.hasDueSchedule(keys,now())||!store.acquireWorker(owner,now(),leaseMs))return;
    try {
      // At most one pass per tick, even if a slow pass lasts longer than ten minutes.
      const attempted=new Set<string>();
      while(!stopped) {
        if(!store.acquireWorker(owner,now(),leaseMs))break;
        const row=store.claimSchedule(keys.filter(key=>!attempted.has(key)),now(),leaseMs,refreshMs());
        if(!row)break;
        attempted.add(row.key);
        try {
          const decision=await refresh(JSON.parse(row.request),market=>store.saveScheduledMarket(row.key,market));
          store.finishSchedule(row.key,now(),decision.id,null);
        } catch(error) {
          const message=error instanceof ApiError?error.message:'자동 분석을 갱신하지 못했습니다. 저장된 결과를 표시합니다.';
          store.finishSchedule(row.key,now(),null,message);
          const code=error instanceof ApiError?error.code:'';
          if(actionMode&&!['awaiting_candle','daily_limit','action_interval_unsupported'].includes(code))store.rescheduleSooner(row.key,now()+(code==='candle_not_ready'?ACTION_CANDLE_RETRY_MS:ACTION_RETRY_MS));
        }
      }
    } finally {store.releaseWorker(owner);}
  }
  function tick():Promise<void> {
    if(running)return running;
    running=work().finally(()=>{running=null;});return running;
  }
  function start() {
    if(started||stopped||!config.scheduledAnalysisEnabled)return;
    started=true;
    const loop=()=>{void tick().catch(error=>console.error(JSON.stringify({event:'analysis_scheduler_failed',error:error instanceof Error?error.name:'unknown'}))).finally(()=>{
      if(!stopped)timer=setTimeout(loop,1000).unref();
    });};
    loop();
  }
  async function stop() {stopped=true;if(timer)clearTimeout(timer);if(running)await running;}
  return {view,tick,start,stop,keyFor,saveMarket:(request:TradeRequest,market:Market)=>store.saveScheduledMarket(keyFor(request),market)};
}
