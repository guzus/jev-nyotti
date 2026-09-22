import {readFile,stat} from 'node:fs/promises';
import {join} from 'node:path';
import {z} from 'zod';
import {ApiError} from './errors.js';
const finite=z.number().finite();
const point=z.object({time:z.string().datetime(),equity:finite,pnl:finite,returnPct:finite,drawdownPct:finite,buyHoldEquity:finite});
const reportSchema=z.object({status:z.literal('ready'),model:z.string().min(1),revision:z.string().min(1),generatedAt:z.string().datetime({offset:true}),source:z.string().min(1),priorPolicy:z.literal('previous_prediction'),report:z.object({initialCapital:finite.positive(),finalEquity:finite,pnl:finite,returnPct:finite,maxDrawdownPct:finite,buyHoldEquity:finite,buyHoldReturnPct:finite,fees:finite,slippage:finite,curve:z.array(point).min(2),perSymbol:z.array(z.object({symbol:z.string().min(1),initialCapital:finite,equity:finite,pnl:finite,returnPct:finite,maxDrawdownPct:finite,fees:finite,slippage:finite,fills:finite,endingAction:z.enum(['long','short','flat']),buyHoldEquity:finite})).min(1),assumptions:z.object({feeBps:finite,slippageBps:finite,intervalMinutes:z.literal(60),initialPosition:z.literal('flat'),allocation:z.literal('equal_initial_sleeves'),sizing:z.literal('1x_equity_on_side_change'),execution:z.literal('next_candle_open_at_input_cutoff'),mark:z.literal('hourly_close'),endingPosition:z.literal('marked_to_market_not_liquidated'),fundingIncluded:z.literal(false),borrowCostsIncluded:z.literal(false),liquidationModeled:z.literal(false),reconstruction:z.literal(true)})})});
// Only the offline importer writes this artifact. Public reads never start inference.
export function performanceReader(directory:string){
 let modified=-1,cached:unknown;
 return async()=>{
  try{
   const file=join(directory,'pnl-report.json'),info=await stat(file);
   if(info.size>20_000_000)throw Error('Report too large');
   if(info.mtimeMs!==modified){
    const data=reportSchema.parse(JSON.parse(await readFile(file,'utf8')));
    if(data.status!=='ready'||!data.report?.curve?.length||!Array.isArray(data.report.perSymbol)||!data.model||!data.revision)throw Error('Invalid report');
    // Bound chart payload while retaining exact full-resolution summary metrics.
    const curve=data.report.curve;
    const step=Math.max(1,Math.ceil(curve.length/1500));
    data.report.curve=curve.filter((_:unknown,i:number)=>i%step===0||i===curve.length-1);
    cached=data;modified=info.mtimeMs;
   }
   return cached;
  }catch(error){
   if((error as NodeJS.ErrnoException).code==='ENOENT')return {status:'pending',reason:'backfill_not_run',report:null};
   throw new ApiError(503,'performance_unavailable','성과 데이터를 확인하지 못했습니다. 잠시 후 다시 시도해 주세요.');
  }
 };
}
