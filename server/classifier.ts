import type { Answer, Question, RawScores, ScoringJob, SystemOneRequest } from './contracts.js';
import { ApiError } from './errors.js';

export function optionsFor(question:Question):ScoringJob['options'] {
  if (question.type === 'choice') return Object.entries(question.criteria).map(([name,description]) => ({name,description}));
  if (question.type === 'score') return question.criteria.map((description,i) => ({name:String(i),description}));
  return [
    {name:'false',description:question.criteria?.false ?? 'No. The proposition is false.'},
    {name:'true',description:question.criteria?.true ?? 'Yes. The proposition is true.'},
  ];
}

export function jobsFor(request:SystemOneRequest):ScoringJob[] {
  // IDs are transport keys, never part of the model input. Each question is evaluated independently.
  return Object.values(request.questions).map(q => ({state:request.state,instructions:q.instructions,options:optionsFor(q)}));
}

export function probabilitiesFromLogits(logits:number[],expected:number):number[] {
  if (logits.length !== expected || !logits.every(Number.isFinite)) throw new ApiError(502,'invalid_model_output','모델 점수 응답이 올바르지 않습니다.');
  const max = Math.max(...logits);
  const values = logits.map(x=>Math.exp(x-max));
  const total = values.reduce((a,b)=>a+b,0);
  return values.map(x=>x/total);
}

export function answerFor(question:Question,logits:number[]):Answer {
  const options = optionsFor(question);
  const p = probabilitiesFromLogits(logits,options.length);
  if (question.type === 'noul') return {type:'noul',noul:p[1]};
  const probabilities = Object.fromEntries(options.map((o,i)=>[o.name,p[i]]));
  const entropy = -p.reduce((sum,n)=>sum+(n>0?n*Math.log(n):0),0);
  // Our documented compatibility measure, not TypeSafe's proprietary confidence formula.
  const confidence = p.length===1 ? 1 : Math.max(0,Math.min(1,1-entropy/Math.log(p.length)));
  if (question.type === 'choice') {
    const best = p.reduce((winner,n,i)=>n>p[winner]?i:winner,0);
    return {type:'choice',choice:options[best].name,probabilities,confidence};
  }
  return {type:'score',score:p.reduce((sum,n,i)=>sum+n*i,0),probabilities,confidence,
    legend:Object.fromEntries(options.map(o=>[o.name,typeof o.description==='string'?o.description:JSON.stringify(o.description)]))};
}

export function assembleResponse(request:SystemOneRequest,raw:RawScores,modelId:string) {
  const entries = Object.entries(request.questions);
  if (raw.model!==modelId || raw.scores.length!==entries.length || !raw.revision) throw new ApiError(502,'model_mismatch','모델 응답의 버전 또는 질문 수가 일치하지 않습니다.');
  if (!raw.scores.every(s=>Number.isSafeInteger(s.inputTokens)&&s.inputTokens>0)) throw new ApiError(502,'invalid_usage','모델의 토큰 사용량이 올바르지 않습니다.');
  return {
    model:modelId,
    answers:Object.fromEntries(entries.map(([id,q],i)=>[id,answerFor(q,raw.scores[i].logits)])),
    usage:{input_tokens:raw.scores.reduce((n,s)=>n+s.inputTokens,0),output_tokens:0},
    metadata:{model_revision:raw.revision,score_semantics:'conditional_next_label_softmax_uncalibrated',
      confidence_method:'one_minus_normalized_entropy',usage_semantics:'separate_forward_pass_per_question_no_generated_tokens',
      compatibility:'TypeSafe wire format; Qwen model, not Jev weights or calibration'},
  };
}
