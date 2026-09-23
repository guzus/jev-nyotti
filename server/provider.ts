import { z } from 'zod';
import type { Config } from './config.js';
import type { RawScores,ScoringJob } from './contracts.js';
import { ApiError } from './errors.js';
import type { Candle } from './market.js';
import { ALL_ACTIONS,type PaperSide } from './paper.js';

export interface Scorer { configured:boolean; score(jobs:ScoringJob[]):Promise<RawScores>; }
const rawSchema = z.object({
  model:z.string(),revision:z.string().min(1),elapsedMs:z.number().finite().nonnegative(),
  scores:z.array(z.object({logits:z.array(z.number().finite()).min(1).max(255),inputTokens:z.number().int().positive()})).min(1).max(8),
});
export function createScorer(config:Config):Scorer {
  return {
    configured:!!config.inferenceUrl,
    async score(jobs) {
      if (!config.inferenceUrl) throw new ApiError(503,'model_unconfigured','모델 서버 연결을 준비하고 있습니다. 시장 데이터는 확인할 수 있습니다.');
      try {
        const response=await fetch(`${config.inferenceUrl}/score`,{
          method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${config.inferenceKey}`,
            ...(config.modalKey?{'Modal-Key':config.modalKey,'Modal-Secret':config.modalSecret}:{})},
          body:JSON.stringify({jobs}),signal:AbortSignal.timeout(config.inferenceTimeout),redirect:'error',
        });
        if (!response.ok) {
          if (response.status===422 || response.status===413) throw new ApiError(422,'model_input_limit','입력이 모델의 길이 또는 형식 제한을 초과했습니다.');
          if (response.status===429 || response.status===529) throw new ApiError(529,'model_busy','모델이 사용 중입니다. 잠시 후 다시 시도해 주세요.');
          throw new ApiError(503,'model_unavailable','모델 서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.');
        }
        const text=await response.text();
        if (text.length>128000) throw new ApiError(502,'invalid_model_output','모델 응답이 너무 큽니다.');
        const parsed=rawSchema.safeParse(JSON.parse(text));
        if (!parsed.success) throw new ApiError(502,'invalid_model_output','모델 점수 응답을 확인할 수 없습니다.');
        return parsed.data;
      } catch(error) {
        if (error instanceof ApiError) throw error;
        throw new ApiError(503,'model_unavailable','모델 서버 응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요.');
      }
    },
  };
}

// ACTION_V1: the inference service owns prompt construction (action_task.build_job).
// The gateway sends raw closed candles and the carried paper position only.
export type ActionRequestBody = {
  /** ISO UTC `YYYY-MM-DDTHH:MM:SSZ`; candle and position times are epoch seconds. */
  market:string; cutoff:string; candles:Candle[];
  position:{side:PaperSide;entry_price:number|null;opened_at:number|null;last_trade_at:number|null};
};
export const actionResponseSchema = z.object({
  model:z.string().min(1),revision:z.string().min(1),task:z.literal('ACTION_V1'),action:z.enum(ALL_ACTIONS),
  options:z.array(z.object({name:z.enum(ALL_ACTIONS),probability:z.number().finite().min(0).max(1)})).min(3).max(4),
  holdMargin:z.number().finite(),inputTokens:z.number().int().positive(),elapsedMs:z.number().finite().nonnegative(),
});
export type ActionResponse = z.infer<typeof actionResponseSchema>;
export interface Actor { configured:boolean; act(body:ActionRequestBody):Promise<ActionResponse>; }

export function createActor(config:Config):Actor {
  return {
    configured:!!config.inferenceUrl,
    async act(body) {
      if (!config.inferenceUrl) throw new ApiError(503,'model_unconfigured','모델 서버 연결을 준비하고 있습니다. 시장 데이터는 확인할 수 있습니다.');
      try {
        const response=await fetch(`${config.inferenceUrl}/action`,{
          method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${config.inferenceKey}`,
            ...(config.modalKey?{'Modal-Key':config.modalKey,'Modal-Secret':config.modalSecret}:{})},
          body:JSON.stringify(body),signal:AbortSignal.timeout(config.inferenceTimeout),redirect:'error',
        });
        if (!response.ok) {
          if (response.status===422 || response.status===413) throw new ApiError(422,'model_input_limit','입력이 모델의 길이 또는 형식 제한을 초과했습니다.');
          if (response.status===429 || response.status===529) throw new ApiError(529,'model_busy','모델이 사용 중입니다. 잠시 후 다시 시도해 주세요.');
          throw new ApiError(503,'model_unavailable','모델 서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.');
        }
        const text=await response.text();
        if (text.length>32000) throw new ApiError(502,'invalid_model_output','모델 응답이 너무 큽니다.');
        const parsed=actionResponseSchema.safeParse(JSON.parse(text));
        if (!parsed.success) throw new ApiError(502,'invalid_model_output','모델 행동 응답을 확인할 수 없습니다.');
        return parsed.data;
      } catch(error) {
        if (error instanceof ApiError) throw error;
        throw new ApiError(503,'model_unavailable','모델 서버 응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요.');
      }
    },
  };
}
