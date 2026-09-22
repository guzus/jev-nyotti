import type { TradeRequest } from './contracts.js';
import { ApiError } from './errors.js';

export type Candle={time:number;open:number;high:number;low:number;close:number;volume:number};
export type Market={symbol:TradeRequest['symbol'];interval:TradeRequest['interval'];source:'Kraken';asOf:string;fetchedAt:string;candles:Candle[];features:ReturnType<typeof features>};

export function features(candles:Candle[]) {
  const closes=candles.map(c=>c.close);
  const lastClose=closes.at(-1)!;
  const changes=closes.slice(1).map((c,i)=>c/closes[i]-1);
  const mean=changes.reduce((a,b)=>a+b,0)/changes.length;
  const volatilityPct=Math.sqrt(changes.reduce((a,b)=>a+(b-mean)**2,0)/changes.length)*100;
  const deltas=closes.slice(-15).slice(1).map((c,i)=>c-closes.slice(-15)[i]);
  const gain=deltas.reduce((s,n)=>s+Math.max(n,0),0)/14;
  const loss=deltas.reduce((s,n)=>s+Math.max(-n,0),0)/14;
  const rsi14=loss===0?(gain===0?50:100):100-100/(1+gain/loss);
  const prevVolume=candles.slice(-21,-1).reduce((s,c)=>s+c.volume,0)/20;
  return {lastClose,changePct:(lastClose/closes[0]-1)*100,rsi14,
    volatilityPct,volumeRatio:prevVolume>0?candles.at(-1)!.volume/prevVolume:null};
}

export function parseKraken(payload:unknown,request:TradeRequest,now=Date.now()):Market {
  const p=payload as {error?:unknown;result?:Record<string,unknown>};
  if (!p||!Array.isArray(p.error)||p.error.length||!p.result) throw new ApiError(503,'market_unavailable','시장 데이터를 가져오지 못했습니다.');
  const rows=Object.entries(p.result).find(([key,value])=>key!=='last'&&Array.isArray(value))?.[1];
  if (!Array.isArray(rows)) throw new ApiError(502,'invalid_market_data','시장 데이터 형식이 올바르지 않습니다.');
  // Kraken always includes the current unfinished candle. Exclude it, then independently verify the timestamp.
  const closed=rows.slice(0,-1).map((row:unknown)=>{
    if(!Array.isArray(row)||row.length<8) throw new ApiError(502,'invalid_market_data','시장 데이터 형식이 올바르지 않습니다.');
    const c={time:Number(row[0]),open:Number(row[1]),high:Number(row[2]),low:Number(row[3]),close:Number(row[4]),volume:Number(row[6])};
    if(!Object.values(c).every(Number.isFinite)||c.time<=0||c.open<=0||c.low<=0||c.close<=0||c.high<Math.max(c.open,c.close,c.low)||c.low>Math.min(c.open,c.close)||c.volume<0) throw new ApiError(502,'invalid_market_data','가격 데이터 검증에 실패했습니다.');
    return c;
  }).filter(c=>(c.time+request.interval*60)*1000<=now).slice(-96);
  if(closed.length<30) throw new ApiError(503,'insufficient_market_data','분석에 필요한 시장 데이터가 부족합니다.');
  if(closed.some((c,i)=>i>0&&c.time-closed[i-1].time!==request.interval*60)) throw new ApiError(503,'market_gap','시장 데이터에 누락된 구간이 있습니다.');
  const end=(closed.at(-1)!.time+request.interval*60)*1000;
  if(now-end>request.interval*60000+120000) throw new ApiError(503,'stale_market_data','시장 데이터가 오래되어 새 분석을 중단했습니다.');
  return {...request,source:'Kraken',asOf:new Date(end).toISOString(),fetchedAt:new Date(now).toISOString(),candles:closed,features:features(closed)};
}

export function createMarketReader(fetcher:typeof fetch=fetch) {
  const cache=new Map<string,{expires:number;value:Market}>();
  const inflight=new Map<string,Promise<Market>>();
  return async(request:TradeRequest):Promise<Market>=>{
    const key=`${request.symbol}:${request.interval}`;
    const cached=cache.get(key);
    if(cached&&cached.expires>Date.now()) return cached.value;
    const waiting=inflight.get(key);if(waiting)return waiting;
    const pending=(async()=>{
      try {
        const pair=request.symbol==='BTCUSD'?'XBTUSD':request.symbol;
        const response=await fetcher(`https://api.kraken.com/0/public/OHLC?pair=${pair}&interval=${request.interval}`,{signal:AbortSignal.timeout(15000),redirect:'error'});
        if(!response.ok) throw new ApiError(503,'market_unavailable','시장 데이터 공급자가 응답하지 않습니다.');
        const result=parseKraken(await response.json(),request);
        cache.set(key,{expires:Math.min(Date.now()+60000,Math.ceil(Date.now()/(request.interval*60000))*request.interval*60000+1000),value:result});
        return result;
      } catch(error) {
        if(error instanceof ApiError) throw error;
        throw new ApiError(503,'market_unavailable','시장 데이터에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.');
      } finally {inflight.delete(key);}
    })();inflight.set(key,pending);return pending;
  };
}
