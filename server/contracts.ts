import { z } from 'zod';
import type { ActionName, PaperSide } from './paper.js';

// Bump whenever public analysis prompts, task semantics or feature definitions change.
export const ANALYSIS_CACHE_VERSION=2;

const jsonContent = z.union([z.string().max(24000), z.array(z.unknown()), z.record(z.string(), z.unknown())]);
const key = z.string().min(1).max(128);
export const questionSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('choice'), instructions: jsonContent,
    criteria: z.record(key, jsonContent.nullable()).refine(v => Object.keys(v).length >= 1 && Object.keys(v).length <= 255, 'Choice requires 1–255 options') }).strict(),
  z.object({ type: z.literal('score'), instructions: jsonContent, criteria: z.array(jsonContent).min(2).max(10) }).strict(),
  z.object({ type: z.literal('noul'), instructions: jsonContent,
    criteria: z.object({ true: jsonContent.optional(), false: jsonContent.optional() }).strict().optional() }).strict(),
]);
export const systemOneSchema = z.object({
  model: z.string().min(1).max(200), state: jsonContent,
  questions: z.record(key, questionSchema).refine(v => Object.keys(v).length >= 1 && Object.keys(v).length <= 8, 'Provide 1–8 questions'),
}).strict();
export const tradeSchema = z.object({
  symbol: z.enum(['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 'DOGEUSD', 'ADAUSD', 'AVAXUSD', 'LINKUSD', 'DOTUSD', 'LTCUSD', 'BNBUSD', 'SUIUSD', 'NEARUSD', 'PEPEUSD', 'ZECUSD']), interval: z.union([z.literal(15), z.literal(60), z.literal(240)]),
}).strict();
export type Question = z.infer<typeof questionSchema>;
export type SystemOneRequest = z.infer<typeof systemOneSchema>;
export type TradeRequest = z.infer<typeof tradeSchema>;
export type ScoringJob = { state: SystemOneRequest['state']; instructions: Question['instructions']; options: { name: string; description: unknown }[] };
export type RawScores = { model: string; revision: string; scores: { logits: number[]; inputTokens: number }[]; elapsedMs: number };
export type Answer =
  | {type:'choice';choice:string;probabilities:Record<string,number>;confidence:number}
  | {type:'score';score:number;legend:Record<string,string>;probabilities:Record<string,number>;confidence:number}
  | {type:'noul';noul:number};
export type LegacyDecision = {
  id:string; symbol:TradeRequest['symbol']; interval:TradeRequest['interval']; action:'long'|'short'|'hold'|'flat';
  summary:string; model:string; revision:string; trainingStatus:'base'|'fine_tuned';
  generatedAt:string; marketAsOf:string; latencyMs:number; cached:boolean;
  semantics?:'next_hour_position_side';
  scores:{long:number;short:number;hold:number}|{long:number;short:number;flat:number}; scoreType:'model_relative_likelihood';
};
export type PaperSnapshot = {
  side:PaperSide; units:number; entryPrice:number|null; openedAt:string|null; lastTradeAt:string|null;
  /** Cutoff candle close used as the mark. */
  markPrice:number;
  /** Per-unit signed return (%), same convention as the model input. */
  unitReturnPct:number;
  /** units × unitReturnPct: open paper PnL in % of one unit notional. */
  unrealizedPct:number;
  /** Cumulative realized PnL after fees, % of one unit notional. */
  realizedPct:number; feesPct:number; trades:number;
};
export type ActionLogRow = {
  cutoff:string; kind:'action'|'gap'; action:ActionName|null;
  sideBefore:PaperSide; unitsBefore:number; sideAfter:PaperSide; unitsAfter:number;
  price:number|null; unitsTraded:number; feePct:number; realizedPct:number;
  /** gap rows: 15m cutoffs skipped (downtime/failures); never backfilled. */
  missedCutoffs:number; decisionId:string|null;
};
export type ActionDecision = {
  id:string; symbol:TradeRequest['symbol']; interval:15; task:'ACTION_V1'; action:ActionName;
  summary:string; model:string; revision:string; trainingStatus:'action_v1';
  generatedAt:string; marketAsOf:string; latencyMs:number; cached:boolean;
  scores:Partial<Record<ActionName,number>>; scoreType:'model_relative_likelihood';
  options:{name:ActionName;probability:number}[]; holdMargin:number; inputTokens:number;
  transfer:'in_distribution'|'untested_transfer';
  positionBefore:{side:PaperSide;units:number;entryPrice:number|null};
  execution:{price:number;unitsTraded:number;feePct:number;realizedPct:number};
  paper:PaperSnapshot; missedCutoffs:number;
  /** Attached on cached views only (latest 50, newest first); not part of the immutable record. */
  actionLog?:ActionLogRow[];
};
export type Decision = LegacyDecision | ActionDecision;
