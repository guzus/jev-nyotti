import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createApp } from '../server/app.js';
import { actionTarget, readConfig } from '../server/config.js';
import { validateActionResponse } from '../server/action.js';
import { parseKraken } from '../server/market.js';
import { actionResponseSchema, createActor, type Actor } from '../server/provider.js';
import { flatPosition, optionsFor } from '../server/paper.js';
import { actionModelLabel } from '../web/action.js';

const key = 'test-only-not-a-deployed-secret-123456789';
const numeric = { model: 'jev-numeric/logreg', revision: `numeric-logreg:sha256:${'a'.repeat(64)}` };
const response = (over: object = {}) => ({ ...numeric, policy: 'numeric', task: 'ACTION_V1', action: 'hold',
  options: optionsFor('flat').map(name => ({ name, probability: 1 / 3 })), holdMargin: 0.4, inputTokens: 0, elapsedMs: 1, ...over });

function withEnv<T>(env: Record<string, string | undefined>, run: () => T): T {
  const saved = Object.fromEntries(Object.keys(env).map(k => [k, process.env[k]]));
  for (const [k, v] of Object.entries(env)) { if (v === undefined) delete process.env[k]; else process.env[k] = v; }
  try { return run(); } finally { for (const [k, v] of Object.entries(saved)) { if (v === undefined) delete process.env[k]; else process.env[k] = v; } }
}

test('action response: policy defaults to lora for older services, numeric may report 0 input tokens', () => {
  const { policy: _omit, ...legacy } = response({ model: 'Qwen/Qwen3.5-4B', inputTokens: 900 });
  assert.equal(actionResponseSchema.parse(legacy).policy, 'lora');
  assert.equal(actionResponseSchema.parse(response()).policy, 'numeric');
  assert.equal(actionResponseSchema.safeParse(response({ policy: 'lora_with_prior' })).success, false);
});

test('ACTION_INFERENCE_URL and ACTION_MODEL_* override the /score target, with fallbacks', () => {
  const base = { INFERENCE_URL: 'https://gpu.example.com/', INFERENCE_API_KEY: key, MODEL_REVISION: 'r1', MODEL_TRAINING_STATUS: 'action_v1' };
  const fallback = withEnv({ ...base, ACTION_INFERENCE_URL: undefined, ACTION_MODEL_ID: undefined, ACTION_MODEL_REVISION: undefined }, readConfig);
  assert.deepEqual(actionTarget(fallback), { url: 'https://gpu.example.com', model: 'Qwen/Qwen3.5-4B', revision: 'r1' });
  const split = withEnv({ ...base, ACTION_INFERENCE_URL: 'https://cpu.example.com/', ACTION_MODEL_ID: numeric.model, ACTION_MODEL_REVISION: numeric.revision }, readConfig);
  assert.deepEqual(actionTarget(split), { url: 'https://cpu.example.com', ...numeric });
  assert.equal(split.inferenceUrl, 'https://gpu.example.com');
  assert.throws(() => withEnv({ ...base, ACTION_INFERENCE_URL: 'http://cpu.example.com' }, readConfig), /ACTION_INFERENCE_URL requires HTTPS/);
  assert.throws(() => withEnv({ ...base, ACTION_INFERENCE_URL: 'https://x-action-cpu-api.modal.run', MODAL_PROXY_KEY: undefined, MODAL_PROXY_SECRET: undefined }, readConfig), /proxy credentials/);
});

test('actor posts to the action origin; the gateway pins the reported numeric identity', async () => {
  const config = { ...readConfig(), inferenceUrl: 'https://gpu.example.com', actionInferenceUrl: 'https://cpu.example.com', inferenceKey: key,
    actionModelId: numeric.model, actionModelRevision: numeric.revision };
  const original = globalThis.fetch;
  let url = '';
  globalThis.fetch = (async (input: string) => { url = input; return new Response(JSON.stringify(response())); }) as typeof fetch;
  try {
    const raw = await createActor(config).act({} as never);
    assert.equal(url, 'https://cpu.example.com/action');
    validateActionResponse(raw, flatPosition(), config);
    assert.throws(() => validateActionResponse({ ...raw, model: 'Qwen/Qwen3.5-4B' }, flatPosition(), config), /버전이 다릅니다/);
  } finally { globalThis.fetch = original; }
});

test('UI label comes from the response identity and is honest about non-Qwen policies', () => {
  assert.equal(actionModelLabel({ ...numeric, policy: 'numeric' }), 'jev뇨띠 수치 정책 (로지스틱 회귀) · Qwen 아님');
  assert.equal(actionModelLabel({ model: 'jev-numeric/hgb', revision: 'x', policy: 'numeric' }), 'jev뇨띠 수치 정책 (GBM) · Qwen 아님');
  assert.equal(actionModelLabel({ model: 'Qwen/Qwen3.5-4B', revision: 'abc+lora:guzus/jev-nyotti@def' }), 'Qwen3.5-4B LoRA');
  assert.equal(actionModelLabel({ model: 'Qwen/Qwen3.5-4B', revision: 'abc', policy: 'lora' }), 'Qwen3.5-4B 기본 모델');
});

test('stored ACTION_V1 decision records the served model and policy', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'jev-policy-'));
  const clock = Date.UTC(2026, 8, 23, 0, 1);
  const boundary = Math.floor(clock / 900000) * 900;
  const rows = Array.from({ length: 101 }, (_, i) => { const t = boundary - (100 - i) * 900; const c = String(t / 900 % 1000 + 100); return [t, c, c, c, c, c, '5', 3]; });
  const actor: Actor = { configured: true, act: async body => actionResponseSchema.parse(response({ options: optionsFor(body.position.side).map((name, _i, all) => ({ name, probability: 1 / all.length })) })) };
  const config = { ...readConfig(), dataDir: dir, apiKey: key, trainingStatus: 'action_v1' as const, modelRevision: 'qwen-rev', actionModelId: numeric.model, actionModelRevision: numeric.revision, scheduledAnalysisEnabled: true };
  const { app, store, scheduler } = createApp(config, { actor, now: () => clock, market: async q => parseKraken({ error: [], result: { XXBTZUSD: rows, last: boundary } }, q, clock),
    scorer: { configured: true, score: async () => { throw Error('legacy scorer must not run'); } } });
  const server = app.listen(0, '127.0.0.1'); await new Promise<void>(r => server.once('listening', r));
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  try {
    await scheduler.tick();
    const res = await fetch(`http://127.0.0.1:${address.port}/api/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol: 'BTCUSD', interval: 15 }) });
    const decision = await res.json();
    assert.equal(res.status, 200);
    assert.deepEqual([decision.model, decision.revision, decision.policy, decision.inputTokens], [numeric.model, numeric.revision, 'numeric', 0]);
    const status = await (await fetch(`http://127.0.0.1:${address.port}/api/status`)).json();
    assert.deepEqual(status.action, numeric);
  } finally { await new Promise<void>((r, e) => server.close(x => x ? e(x) : r())); await scheduler.stop(); store.close(); rmSync(dir, { recursive: true, force: true }); }
});
