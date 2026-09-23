import { createHash, randomUUID } from 'node:crypto';
import type { Config } from './config.js';
import type { ActionDecision, ActionLogRow, TradeRequest } from './contracts.js';
import { ApiError } from './errors.js';
import type { Market } from './market.js';
import {
  ACTION_LOOKBACK, ACTION_STEP_SECONDS, flatPosition, optionsFor, stepPaper, unitReturnPct, unrealizedPct,
  type ActionName, type PaperPosition,
} from './paper.js';
import type { ActionRequestBody, ActionResponse, Actor } from './provider.js';
import type { PaperInventory, Store } from './store.js';

export const ACTION_LABELS: Record<ActionName, string> = {
  hold: '관망', open_long: '롱 진입', open_short: '숏 진입', add: '추가', reduce: '축소', close: '청산',
};
const SIDE_LABELS = { flat: '무포지션', long: '롱', short: '숏' } as const;
const iso = (seconds: number | null) => seconds === null ? null : new Date(seconds * 1000).toISOString();

/** The request body the gateway sends to POST {INFERENCE_URL}/action. Exported for contract tests. */
export function actionBody(market: Market, position: PaperPosition): ActionRequestBody {
  const cutoff = actionCutoff(market);
  return {
    // action_task.iso format: second precision, no milliseconds.
    market: `Kraken ${market.symbol} spot`, cutoff: new Date(cutoff * 1000).toISOString().replace('.000Z', 'Z'),
    candles: market.candles.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume })),
    position: { side: position.side, entry_price: position.entry_price, opened_at: position.opened_at, last_trade_at: position.last_trade_at },
  };
}

/** Cutoff = close of the last closed 15m candle (Kraken `time` is candle OPEN). */
export function actionCutoff(market: Market): number {
  if (market.interval !== 15) throw new ApiError(422, 'action_interval_unsupported', 'ACTION_V1은 15분 봉에서만 판단합니다.');
  const candles = market.candles;
  if (candles.length !== ACTION_LOOKBACK) throw new ApiError(503, 'insufficient_market_data', '15분 봉 96개가 필요합니다.');
  const cutoff = candles.at(-1)!.time + ACTION_STEP_SECONDS;
  if (cutoff % ACTION_STEP_SECONDS || cutoff * 1000 !== Date.parse(market.asOf)) throw new ApiError(502, 'invalid_market_data', '시장 데이터 기준 시각이 올바르지 않습니다.');
  if (candles.some((c, i) => c.time !== cutoff - (ACTION_LOOKBACK - i) * ACTION_STEP_SECONDS)) throw new ApiError(503, 'market_gap', '시장 데이터에 누락된 구간이 있습니다.');
  return cutoff;
}

/** Next due time: 60 s after the next 15m boundary, letting the final trades of the candle settle. */
export function nextActionDue(now: number) {
  const step = ACTION_STEP_SECONDS * 1000;
  return (Math.floor((now - 60000) / step) + 1) * step + 60000;
}

export function validateActionResponse(raw: ActionResponse, position: PaperPosition, config: Config) {
  if (raw.model !== config.modelId || raw.revision !== config.modelRevision) throw new ApiError(502, 'model_revision_mismatch', '설정된 모델과 실행 중인 모델의 버전이 다릅니다.');
  const expected = optionsFor(position.side);
  const names = raw.options.map(o => o.name);
  if (names.length !== expected.length || new Set(names).size !== names.length || !names.every(n => expected.includes(n))) throw new ApiError(502, 'invalid_model_output', '모델 선택지가 포지션 상태와 일치하지 않습니다.');
  if (!names.includes(raw.action)) throw new ApiError(502, 'invalid_model_output', '모델 행동이 허용된 선택지가 아닙니다.');
  const total = raw.options.reduce((s, o) => s + o.probability, 0);
  if (Math.abs(total - 1) > 1e-4) throw new ApiError(502, 'invalid_model_output', '모델 확률 합계가 올바르지 않습니다.');
  // Note: the served decision is argmax(logits + hold margin), so `action` need not be the top probability.
}

export function decisionKey(config: Config, symbol: string, cutoff: number) {
  return createHash('sha256').update(JSON.stringify({ task: 'ACTION_V1', model: config.modelId, revision: config.modelRevision, symbol, cutoff })).digest('hex');
}

export function createActionRunner(config: Config, store: Store, actor: Actor, now: () => number) {
  const inflight = new Map<string, Promise<ActionDecision>>();
  const revision = config.modelRevision;
  const inventoryOf = (symbol: string): PaperInventory => store.getInventory(symbol, revision) ?? { position: flatPosition(), realizedPct: 0, feesPct: 0, trades: 0, updatedCutoff: null, markPrice: null };

  async function decide(market: Market): Promise<ActionDecision> {
    const cutoff = actionCutoff(market);
    const inv = inventoryOf(market.symbol);
    if (inv.updatedCutoff !== null && cutoff <= inv.updatedCutoff) {
      // The provider has not published the newer closed candle yet: retry shortly instead of logging a gap later.
      if (now() >= (inv.updatedCutoff + ACTION_STEP_SECONDS) * 1000 + 60000) throw new ApiError(503, 'candle_not_ready', '새 15분 봉이 아직 공개되지 않았습니다. 잠시 후 다시 확인합니다.');
      // Never re-apply a cutoff. Serve the stored decision for the latest applied cutoff.
      const existing = store.getCached(decisionKey(config, market.symbol, inv.updatedCutoff));
      if (existing && 'task' in existing) return existing;
      throw new ApiError(503, 'awaiting_candle', '다음 15분 봉 마감을 기다리고 있습니다.');
    }
    if (!actor.configured) throw new ApiError(503, 'model_unconfigured', '모델 서버 연결을 준비하고 있습니다. 시장 데이터는 확인할 수 있습니다.');
    store.reserveEvaluations(1, config.dailyLimit, now());
    const started = performance.now();
    const raw = await actor.act(actionBody(market, inv.position));
    validateActionResponse(raw, inv.position, config);
    const price = market.candles.at(-1)!.close;
    const step = stepPaper(inv.position, raw.action, price, cutoff);
    const missed = inv.updatedCutoff === null ? 0 : (cutoff - inv.updatedCutoff) / ACTION_STEP_SECONDS - 1;
    const id = randomUUID();
    const next: PaperInventory = {
      position: step.after, realizedPct: inv.realizedPct + step.realizedPct, feesPct: inv.feesPct + step.feePct,
      trades: inv.trades + (step.unitsTraded > 0 ? 1 : 0), updatedCutoff: cutoff, markPrice: price,
    };
    const before = inv.position, after = step.after;
    const rows: ActionLogRow[] = [];
    if (missed > 0) rows.push({ cutoff: iso(cutoff)!, kind: 'gap', action: null, sideBefore: before.side, unitsBefore: before.units, sideAfter: before.side, unitsAfter: before.units,
      price: null, unitsTraded: 0, feePct: 0, realizedPct: 0, missedCutoffs: missed, decisionId: null });
    rows.push({ cutoff: iso(cutoff)!, kind: 'action', action: raw.action, sideBefore: before.side, unitsBefore: before.units, sideAfter: after.side, unitsAfter: after.units,
      price, unitsTraded: step.unitsTraded, feePct: step.feePct, realizedPct: step.realizedPct, missedCutoffs: 0, decisionId: id });
    const unitRet = unitReturnPct(after.side, after.entry_price, price);
    const decision: ActionDecision = {
      id, symbol: market.symbol, interval: 15, task: 'ACTION_V1', action: raw.action,
      summary: `${ACTION_LABELS[raw.action]} · 페이퍼 포지션 ${SIDE_LABELS[before.side]}${before.units ? ` ${before.units}단위` : ''} → ${SIDE_LABELS[after.side]}${after.units ? ` ${after.units}단위` : ''}. 기계적 결과 요약이며 모델이 생성한 매매 근거가 아닙니다.`,
      model: config.modelId, revision: raw.revision, trainingStatus: 'action_v1',
      generatedAt: new Date(now()).toISOString(), marketAsOf: market.asOf, latencyMs: Math.round(performance.now() - started), cached: false,
      scores: Object.fromEntries(raw.options.map(o => [o.name, o.probability])), scoreType: 'model_relative_likelihood',
      options: raw.options, holdMargin: raw.holdMargin, inputTokens: raw.inputTokens,
      transfer: market.symbol === 'BTCUSD' ? 'in_distribution' : 'untested_transfer',
      positionBefore: { side: before.side, units: before.units, entryPrice: before.entry_price },
      execution: { price, unitsTraded: step.unitsTraded, feePct: step.feePct, realizedPct: step.realizedPct },
      paper: { side: after.side, units: after.units, entryPrice: after.entry_price, openedAt: iso(after.opened_at), lastTradeAt: iso(after.last_trade_at),
        markPrice: price, unitReturnPct: unitRet, unrealizedPct: unrealizedPct(after, price), realizedPct: next.realizedPct, feesPct: next.feesPct, trades: next.trades },
      missedCutoffs: missed,
    };
    const result = store.commitPaperAction({ symbol: market.symbol, revision, expectedCutoff: inv.updatedCutoff, cutoff, inventory: next, rows, decisionKey: decisionKey(config, market.symbol, cutoff), decision });
    if (result === 'duplicate') {
      const existing = store.getCached(decisionKey(config, market.symbol, cutoff));
      if (existing && 'task' in existing) return existing;
      throw new ApiError(409, 'paper_conflict', '다른 작업이 페이퍼 포지션을 먼저 갱신했습니다.');
    }
    return decision;
  }

  async function run(request: TradeRequest, readMarket: (q: TradeRequest) => Promise<Market>, saveMarket: (market: Market) => void) {
    if (request.interval !== 15) throw new ApiError(422, 'action_interval_unsupported', 'ACTION_V1은 15분 봉에서만 판단합니다.');
    const market = await readMarket(request); saveMarket(market);
    const key = `${request.symbol}:${market.asOf}`;
    const pending = inflight.get(key); if (pending) return pending;
    const task = decide(market);
    inflight.set(key, task);
    try { return await task; } finally { inflight.delete(key); }
  }

  function log(symbol: string) { return store.listActions(symbol, revision, 50); }

  function summary() {
    const rows = new Map(store.listInventories(revision).map(r => [r.symbol, r]));
    return (symbols: readonly string[]) => symbols.map(symbol => {
      const inv = rows.get(symbol);
      const last = store.listActions(symbol, revision, 5).find(r => r.kind === 'action') ?? null;
      const p = inv?.position ?? flatPosition();
      const mark = inv?.markPrice ?? null;
      return {
        symbol, transfer: symbol === 'BTCUSD' ? 'in_distribution' : 'untested_transfer',
        updatedCutoff: iso(inv?.updatedCutoff ?? null),
        position: { side: p.side, units: p.units, entryPrice: p.entry_price, openedAt: iso(p.opened_at), lastTradeAt: iso(p.last_trade_at) },
        markPrice: mark, unitReturnPct: mark === null ? 0 : unitReturnPct(p.side, p.entry_price, mark),
        unrealizedPct: mark === null ? 0 : unrealizedPct(p, mark),
        realizedPct: inv?.realizedPct ?? 0, feesPct: inv?.feesPct ?? 0, trades: inv?.trades ?? 0,
        lastAction: last ? { cutoff: last.cutoff, action: last.action, decisionId: last.decisionId } : null,
      };
    });
  }

  return { run, log, summary };
}
