import { resolve } from 'node:path';

function integer(name: string, fallback: number, min: number, max: number) {
  const n = Number(process.env[name] ?? fallback);
  if (!Number.isSafeInteger(n) || n < min || n > max) throw new Error(`Invalid ${name}`);
  return n;
}

export type TrainingStatus = 'base'|'fine_tuned'|'action_v1';

function checkInferenceUrl(name: string, value: string, modalKey: string, modalSecret: string, inferenceKey: string) {
  const url = new URL(value);
  const privateHost = url.hostname.endsWith('.railway.internal') || ['localhost', '127.0.0.1'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && privateHost)) throw new Error(`${name} requires HTTPS outside private networking`);
  if (url.username || url.password || url.search || url.hash) throw new Error(`${name} must not contain credentials/query/fragment`);
  if (url.hostname.endsWith('.modal.run') && (!modalKey || !modalSecret)) throw new Error('Modal inference requires proxy credentials');
  if (inferenceKey.length < 32) throw new Error('INFERENCE_API_KEY must contain at least 32 characters');
}

export function readConfig() {
  const apiKey = process.env.API_KEY ?? '';
  if (apiKey && apiKey.length < 32) throw new Error('API_KEY must contain at least 32 characters');
  if (process.env.NODE_ENV === 'production' && !apiKey) throw new Error('API_KEY is required in production');
  const inferenceUrl = (process.env.INFERENCE_URL ?? '').replace(/\/$/, '');
  const inferenceKey = process.env.INFERENCE_API_KEY ?? '';
  const modalKey = process.env.MODAL_PROXY_KEY ?? '';
  const modalSecret = process.env.MODAL_PROXY_SECRET ?? '';
  if (!!modalKey !== !!modalSecret) throw new Error('Both Modal proxy credentials are required');
  if (inferenceUrl) checkInferenceUrl('INFERENCE_URL', inferenceUrl, modalKey, modalSecret, inferenceKey);
  // ACTION_V1 may be served by a separate origin (e.g. the CPU numeric app) with its own pinned identity.
  // Unset values fall back to INFERENCE_URL / MODEL_ID / MODEL_REVISION (see actionTarget).
  const actionInferenceUrl = (process.env.ACTION_INFERENCE_URL ?? '').replace(/\/$/, '') || null;
  if (actionInferenceUrl) checkInferenceUrl('ACTION_INFERENCE_URL', actionInferenceUrl, modalKey, modalSecret, inferenceKey);
  const actionModelId = process.env.ACTION_MODEL_ID || null;
  const actionModelRevision = process.env.ACTION_MODEL_REVISION || null;
  const trainingStatus = process.env.MODEL_TRAINING_STATUS ?? 'base';
  if (!['base', 'fine_tuned', 'action_v1'].includes(trainingStatus)) throw new Error('Invalid MODEL_TRAINING_STATUS');
  const modelId = process.env.MODEL_ID ?? 'Qwen/Qwen3.5-4B';
  const modelRevision = process.env.MODEL_REVISION ?? 'base';
  if (trainingStatus !== 'base' && modelRevision === 'base') throw new Error('Fine-tuned model requires a revision identifier');
  const gaMeasurementId=process.env.GA_MEASUREMENT_ID??'';
  if(gaMeasurementId&&!/^G-[A-Z0-9]+$/.test(gaMeasurementId))throw new Error('Invalid GA_MEASUREMENT_ID');
  const scheduled=process.env.SCHEDULED_ANALYSIS_ENABLED??'false';
  if(!['true','false'].includes(scheduled))throw new Error('Invalid SCHEDULED_ANALYSIS_ENABLED');
  return {
    gaMeasurementId:gaMeasurementId||null,scheduledAnalysisEnabled:scheduled==='true',
    port: integer('PORT', 3000, 1, 65535), dataDir: resolve(process.env.DATA_DIR ?? '.runtime'),
    apiKey, inferenceUrl, actionInferenceUrl, actionModelId, actionModelRevision, inferenceKey, modalKey, modalSecret, modelId, modelRevision, trainingStatus: trainingStatus as TrainingStatus,
    dailyLimit: integer('MAX_DAILY_EVALUATIONS', 300, 1, 10000),
    publicRate: integer('PUBLIC_REQUESTS_PER_MINUTE', 6, 1, 60),
    inferenceTimeout: integer('INFERENCE_TIMEOUT_MS', 120000, 1000, 300000),
    trustProxy: integer('TRUST_PROXY_HOPS', 0, 0, 3),
  };
}
export type Config = ReturnType<typeof readConfig>;

/** Where /action is sent and which model/revision its responses must report (fallbacks to the /score target). */
export function actionTarget(config: Config) {
  return { url: config.actionInferenceUrl ?? config.inferenceUrl, model: config.actionModelId ?? config.modelId, revision: config.actionModelRevision ?? config.modelRevision };
}
