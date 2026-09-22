import express,{type Request,type Response,type NextFunction} from 'express';
import helmet from 'helmet';
import { createHash,createHmac,randomUUID,timingSafeEqual } from 'node:crypto';
import { resolve } from 'node:path';
import { z } from 'zod';
import type { Config } from './config.js';
import { systemOneSchema,tradeSchema,type Decision,type SystemOneRequest,type TradeRequest } from './contracts.js';
import { assembleResponse,jobsFor } from './classifier.js';
import { ApiError } from './errors.js';
import { createMarketReader,type Market } from './market.js';
import { createScorer,type Scorer } from './provider.js';
import { Store } from './store.js';

export function createApp(config:Config,deps:{store?:Store;scorer?:Scorer;market?:(q:TradeRequest)=>Promise<Market>}={}) {
  const store=deps.store??new Store(config.dataDir);
  const scorer=deps.scorer??createScorer(config);
  const readMarket=deps.market??createMarketReader();
  const app=express();
  app.disable('x-powered-by');
  app.set('trust proxy',config.trustProxy);
  app.use(helmet({contentSecurityPolicy:{directives:{defaultSrc:["'self'"],scriptSrc:["'self'"],styleSrc:["'self'","'unsafe-inline'"],imgSrc:["'self'",'data:'],fontSrc:["'self'"],connectSrc:["'self'"],frameAncestors:["'none'"],upgradeInsecureRequests:process.env.NODE_ENV==='production'?[]:null}}}));
  app.use(express.json({limit:'64kb',strict:true}));
  app.use((_req,res,next)=>{res.setHeader('X-Request-Id',randomUUID());next();});
  const keyHash=createHash('sha256').update(config.apiKey).digest();
  function auth(req:Request,_res:Response,next:NextFunction) {
    if(!config.apiKey)return next(new ApiError(503,'api_not_configured','API 인증이 설정되지 않았습니다.'));
    const header=req.get('authorization')??'';
    const supplied=header.startsWith('Bearer ')?header.slice(7):'';
    if(!supplied||!timingSafeEqual(createHash('sha256').update(supplied).digest(),keyHash))return next(new ApiError(401,'unauthorized','유효한 Bearer API 키가 필요합니다.'));
    next();
  }
  function consumeRate(req:Request,scope:string,count:number) {
    const digest=createHmac('sha256',config.apiKey||'local-development').update(req.ip??'unknown').digest('hex');
    store.rateLimit(`${scope}:${digest}`,count);
  }
  function limit(scope:string,count:number) {
    return(req:Request,_res:Response,next:NextFunction)=>{
      try {
        consumeRate(req,scope,count);next();
      }catch(e){next(e);}
    };
  }
  let active=0;
  async function evaluate(request:SystemOneRequest) {
    if(request.model!==config.modelId) throw new ApiError(422,'unsupported_model',`사용 가능한 모델: ${config.modelId}`);
    if(!scorer.configured) throw new ApiError(503,'model_unconfigured','모델 서버 연결을 준비하고 있습니다. 시장 데이터는 확인할 수 있습니다.');
    if(active>=2) throw new ApiError(529,'model_busy','모델이 다른 요청을 처리하고 있습니다. 잠시 후 다시 시도해 주세요.');
    store.reserveEvaluations(Object.keys(request.questions).length,config.dailyLimit);
    active++;
    try {
      const raw=await scorer.score(jobsFor(request));
      if(config.modelRevision!=='base'&&raw.revision!==config.modelRevision) throw new ApiError(502,'model_revision_mismatch','설정된 모델과 실행 중인 모델의 버전이 다릅니다.');
      return assembleResponse(request,raw,config.modelId);
    } finally {active--;}
  }

  const decisionsInFlight=new Map<string,Promise<Decision>>();
  function decisionCacheKey(request:TradeRequest,market:Market) {
    // Bump version when the analysis prompt or feature definitions change.
    return createHash('sha256').update(JSON.stringify({version:1,model:config.modelId,revision:config.modelRevision,training:config.trainingStatus,request,asOf:market.asOf,candles:market.candles})).digest('hex');
  }
  async function analyze(request:TradeRequest,beforeInference:()=>void):Promise<Decision> {
    const market=await readMarket(request);
    const cacheKey=decisionCacheKey(request,market);
    const cached=store.getCached(cacheKey);if(cached)return cached;
    const pending=decisionsInFlight.get(cacheKey);if(pending)return {...await pending,cached:true};
    const task=(async()=>{
      beforeInference();
      const started=performance.now();
      const result=await evaluate({
        model:config.modelId,
        state:{symbol:market.symbol,market:'Kraken spot USD, not futures',interval_minutes:market.interval,
          data_cutoff:market.asOf,position:'flat',features:market.features,
          feature_definitions:{changePct:'change across 96 closed candles',rsi14:'simple mean gain/loss over 14 intervals',volatilityPct:'population standard deviation of close returns in percent',volumeRatio:'last closed candle volume / preceding 20-candle mean'},
          recent_closed_candles:market.candles.slice(-24),missing:['order_book','funding','news','trader_history']},
        questions:{direction:{type:'choice',instructions:'Given only the supplied closed-candle market snapshot, classify a hypothetical research stance for the next candle. No orders will be executed. Prefer hold if there is no clear directional evidence; a short is a directional research stance, not a spot short order. Do not infer missing news or trader history.',
          criteria:{long:'Upward directional stance supported by the snapshot.',short:'Downward directional stance supported by the snapshot.',hold:'Insufficient directional evidence; remain flat.'}}},
      });
      const answer=result.answers.direction;
      if(answer.type!=='choice'||!['long','short','hold'].includes(answer.choice)) throw new ApiError(502,'invalid_decision','모델 판단 형식을 확인할 수 없습니다.');
      const action=answer.choice as Decision['action'];
      const labels={long:'상승 방향',short:'하락 방향',hold:'관망'};
      const decision:Decision={id:randomUUID(),...request,action,
        summary:`제공된 종가·거래량 지표에서 ${labels[action]}의 모델 상대 점수가 가장 높았습니다. 이 문장은 결과 요약이며 모델이 생성한 매매 근거가 아닙니다.`,
        model:config.modelId,revision:result.metadata.model_revision,trainingStatus:config.trainingStatus,
        generatedAt:new Date().toISOString(),marketAsOf:market.asOf,latencyMs:Math.round(performance.now()-started),cached:false,
        scores:answer.probabilities as Decision['scores'],scoreType:'model_relative_likelihood'};
      store.saveDecision(cacheKey,decision);return decision;
    })();
    decisionsInFlight.set(cacheKey,task);
    try{return await task;}finally{decisionsInFlight.delete(cacheKey);}
  }

  app.get('/healthz',(_req,res)=>res.json({status:'ok',service:'jev-trading-gateway'}));
  app.get('/api/status',(_req,res)=>res.json({model:config.modelId,revision:config.modelRevision,trainingStatus:config.trainingStatus,
    providerConfigured:scorer.configured,apiAuthRequired:true,inferenceMode:scorer.configured?'live':'unconfigured',
    scoreSemantics:'uncalibrated_model_relative_likelihood',executionEnabled:false}));
  app.get('/api/market',limit('market',60),async(req,res)=>{
    const q=tradeSchema.parse({symbol:req.query.symbol,interval:Number(req.query.interval)});
    const market=await readMarket(q);
    // A page view only reads SQLite; it must never wake the GPU on a cache miss.
    res.setHeader('Cache-Control','no-store');res.json({...market,cachedDecision:store.getCached(decisionCacheKey(q,market))});
  });
  app.post('/api/analyze',limit('analyze-read',120),async(req,res)=>{
    res.setHeader('Cache-Control','no-store');res.json(await analyze(tradeSchema.parse(req.body),()=>consumeRate(req,'analyze',config.publicRate)));
  });
  app.get('/api/decisions/:id',limit('read',120),(req,res)=>{
    const id=z.uuid().safeParse(req.params.id);
    if(!id.success) throw new ApiError(404,'not_found','저장된 분석을 찾을 수 없습니다.');
    const value=store.getDecision(id.data);if(!value) throw new ApiError(404,'not_found','저장된 분석을 찾을 수 없습니다. 공유 결과는 30일간 보관됩니다.');
    res.json({...value,cached:true});
  });
  app.get('/v1/models',auth,(_req,res)=>res.json({models:[{name:config.modelId,
    description:'Qwen3.5-4B logits classifier; TypeSafe wire-compatible, uncalibrated.',
    // Base model release date: official QwenLM repository news, 2026-03-02.
    release_date:'2026-03-02',training_status:config.trainingStatus,revision:config.modelRevision}]}));
  app.post('/v1/systemone',auth,limit('systemone',30),async(req,res)=>{
    res.setHeader('Cache-Control','no-store');
    const body=systemOneSchema.parse(req.body);
    res.json(await evaluate(body));
  });
  app.post('/v1/trading/decisions',auth,limit('trading-read',120),async(req,res)=>{
    res.setHeader('Cache-Control','no-store');res.json(await analyze(tradeSchema.parse(req.body),()=>consumeRate(req,'trading',30)));
  });
  app.get('/openapi.json',(_req,res)=>res.json({
    openapi:'3.1.0',info:{title:'jev뇨띠 — Qwen TypeSafe-compatible API',version:'0.1.0',description:'TypeSafe wire-format compatibility using Qwen logits. No TypeSafe model weights or calibration. Choice/score confidence = 1 − normalized entropy. Maximum 8 independent questions; input token limit enforced by model service. No order execution.'},
    servers:[{url:'/'}],
    components:{securitySchemes:{bearerAuth:{type:'http',scheme:'bearer'}},schemas:{SystemOne:z.toJSONSchema(systemOneSchema),TradingRequest:z.toJSONSchema(tradeSchema)}},
    paths:{
      '/v1/systemone':{post:{operationId:'systemOne',security:[{bearerAuth:[]}],requestBody:{required:true,content:{'application/json':{schema:{$ref:'#/components/schemas/SystemOne'}}}},responses:{'200':{description:'TypeSafe answers: choice / score / noul, usage and score semantics metadata'},'401':{description:'Invalid API key'},'422':{description:'Invalid request / model / token limit'},'429':{description:'Rate or daily quota exceeded'},'503':{description:'Model unavailable'},'529':{description:'At capacity'}}}},
      '/v1/trading/decisions':{post:{operationId:'tradingDecision',description:'Reuses the persisted result for identical closed candles, symbol, interval, model revision and prompt version. Concurrent identical requests share one inference. Cache hits retain the original id, generatedAt, marketAsOf and latencyMs, set cached=true and consume no inference quota. New snapshots require explicit requests; there is no background inference.',security:[{bearerAuth:[]}],requestBody:{required:true,content:{'application/json':{schema:{$ref:'#/components/schemas/TradingRequest'}}}},responses:{'200':{description:'Stored research stance from actual closed Kraken candles'},'429':{description:'Request or new-inference quota exceeded'},'503':{description:'Market or model unavailable'}}}},
      '/v1/models':{get:{operationId:'listModels',security:[{bearerAuth:[]}],responses:{'200':{description:'Actual configured model identity'}}}},
    },
  }));
  app.use(['/api','/v1'],(_req,_res,next)=>next(new ApiError(404,'not_found','존재하지 않는 API 경로입니다.')));
  app.use(express.static(resolve('dist/web'),{index:false,maxAge:'1h'}));
  app.get('/{*path}',(_req,res)=>{res.setHeader('Cache-Control','no-cache');res.sendFile(resolve('dist/web/index.html'));});
  app.use((error:unknown,_req:Request,res:Response,_next:NextFunction)=>{
    if(res.headersSent)return;
    if(error instanceof z.ZodError) {res.status(422).json({error:{code:'invalid_request',message:'요청 형식을 확인해 주세요.',details:error.issues.map(i=>({path:i.path,message:i.message}))}});return;}
    if(error instanceof ApiError){if(error.status===429||error.status===529)res.setHeader('Retry-After','60');res.status(error.status).json({error:{code:error.code,message:error.message}});return;}
    if(typeof error==='object'&&error&&'type' in error&&error.type==='entity.too.large'){res.status(413).json({error:{code:'body_too_large',message:'요청은 64KB 이하여야 합니다.'}});return;}
    if(error instanceof SyntaxError){res.status(400).json({error:{code:'invalid_json',message:'올바른 JSON 요청이 필요합니다.'}});return;}
    console.error(JSON.stringify({event:'request_failed',error:error instanceof Error?error.name:'unknown'}));
    res.status(500).json({error:{code:'internal_error',message:'요청을 처리하지 못했습니다.'}});
  });
  return {app,store};
}
