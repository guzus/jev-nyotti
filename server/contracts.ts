import { z } from 'zod';

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
export type Decision = {
  id:string; symbol:TradeRequest['symbol']; interval:TradeRequest['interval']; action:'long'|'short'|'hold';
  summary:string; model:string; revision:string; trainingStatus:'base'|'fine_tuned';
  generatedAt:string; marketAsOf:string; latencyMs:number; cached:boolean;
  scores:{long:number;short:number;hold:number}; scoreType:'model_relative_likelihood';
};
