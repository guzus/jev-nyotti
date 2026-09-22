import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync, rmSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { simulatePortfolio, type PositionSide, type PnlSeries } from '../server/pnl.js';
import { makePnlReport, importPnlReport } from '../server/pnl-import.js';

const start = Date.UTC(2025, 0, 1) / 1000;
function series(actions: PositionSide[], prices: number[], symbol = 'BTCUSD'): PnlSeries {
  return { symbol, decisions: actions.map((action, i) => ({ action, previousAction: i ? actions[i - 1] : 'flat', marketAsOf: new Date((start + i * 3600) * 1000).toISOString() })), candles: actions.map((_, i) => ({ time: start + i * 3600, open: prices[i], high: Math.max(prices[i], prices[i + 1]), low: Math.min(prices[i], prices[i + 1]), close: prices[i + 1], volume: 1 })) };
}
function close(actual: number, expected: number) { assert.ok(Math.abs(actual - expected) < 1e-8, `${actual} != ${expected}`); }
const withoutCosts = { feeBps: 0, slippageBps: 0 };

test('long, reversal to short, then flat conserves cash and marks next-hour returns', () => {
  const report = simulatePortfolio({ series: [series(['long', 'short', 'flat'], [100, 110, 99, 130])], ...withoutCosts });
  assert.deepEqual(report.curve.map(p => p.equity), [10000, 11000, 12100, 12100]);
  close(report.buyHoldEquity, 13000);
  assert.equal(report.perSymbol[0].fills, 4);
  assert.equal(report.curve[0].time, '2025-01-01T00:00:00.000Z');
  assert.equal(report.curve[1].time, '2025-01-01T01:00:00.000Z');
});

test('unchanged target carries quantity and charges no repeated turnover; flat pays no cost', () => {
  const report = simulatePortfolio({ series: [series(['long', 'long'], [100, 120, 110])] });
  assert.equal(report.perSymbol[0].fills, 1);
  const quantity = 10000 / (100 * 1.0002 * 1.0005);
  close(report.finalEquity, quantity * 110);
  close(report.fees, quantity * 100 * 1.0002 * 0.0005);
  close(report.slippage, quantity * 100 * 0.0002);
  close(report.maxDrawdownPct, (1 - 110 / 120) * 100);
  const flat = simulatePortfolio({ series: [series(['flat', 'flat'], [100, 120, 80])] });
  close(flat.finalEquity, 10000); close(flat.fees, 0); close(flat.slippage, 0);
});

test('adverse execution costs apply to both close and entry on reversal', () => {
  const report = simulatePortfolio({ series: [series(['long', 'short'], [100, 100, 100])], feeBps: 10, slippageBps: 20 });
  const q1 = 10000 / (100 * 1.002 * 1.001);
  const afterClose = q1 * 100 * 0.998 * 0.999;
  const q2 = afterClose / (100 * 1.002 * 1.001);
  close(report.finalEquity, afterClose + q2 * 100 * 0.998 * 0.999 - q2 * 100);
  assert.equal(report.perSymbol[0].fills, 3);
});

test('equally funded sleeves sum correctly without cross-asset rebalancing', () => {
  const report = simulatePortfolio({ series: [series(['long'], [100, 120]), series(['short'], [100, 90], 'ETHUSD')], ...withoutCosts });
  close(report.finalEquity, 11500); close(report.buyHoldEquity, 10500);
  assert.deepEqual(report.perSymbol.map(s => s.initialCapital), [5000, 5000]);
});

test('next-open gap is not a pre-entry profit, and unseen future candles do not change prior marks', () => {
  const s = series(['long', 'long'], [200, 220, 100]);
  const first = simulatePortfolio({ series: [{ ...s, decisions: s.decisions.slice(0, 1), candles: s.candles.slice(0, 1) }], ...withoutCosts });
  const all = simulatePortfolio({ series: [s], ...withoutCosts });
  close(first.finalEquity, 11000);
  assert.deepEqual(first.curve, all.curve.slice(0, 2));
});

test('missing candles, hourly gaps, duplicate symbols and broken state chains fail closed', () => {
  const s = series(['long', 'short'], [100, 110, 100]);
  assert.throws(() => simulatePortfolio({ series: [{ ...s, candles: s.candles.slice(1) }] }), /one execution candle/);
  const gap = structuredClone(s); gap.candles[1].time += 3600;
  assert.throws(() => simulatePortfolio({ series: [gap] }), /next-open/);
  const chain = structuredClone(s); chain.decisions[1].previousAction = 'flat';
  assert.throws(() => simulatePortfolio({ series: [chain] }), /prior-position/);
  const timeGap = structuredClone(s); timeGap.decisions[1].marketAsOf = '2025-01-01T02:00:00Z';
  assert.throws(() => simulatePortfolio({ series: [timeGap] }), /contiguous/);
  assert.throws(() => simulatePortfolio({ series: [s, s] }), /unique/);
  assert.throws(() => simulatePortfolio({ series: [s], feeBps: NaN }), /costs/);
});

test('short intrabar insolvency is rejected even when closing equity would recover', () => {
  const s = series(['short'], [100, 90]); s.candles[0].high = 210;
  assert.throws(() => simulatePortfolio({ series: [s], ...withoutCosts }), /insolvent/);
});

const input = () => ({ model: 'test-model', revision: 'test-revision', source: 'test-fixture', generatedAt: '2025-01-02T00:00:00Z', priorPolicy: 'previous_prediction', series: [series(['long'], [100, 110])] });
test('import requires provenance, explicit prior state and valid completed range', () => {
  assert.equal(makePnlReport(input()).status, 'ready');
  assert.throws(() => makePnlReport({ ...input(), revision: ' ' }));
  assert.throws(() => makePnlReport({ ...input(), generatedAt: 'yesterday' }));
  assert.throws(() => makePnlReport({ ...input(), generatedAt: '2025-01-01T00:30:00Z' }), /precedes/);
  const missing = input(); delete (missing.series[0].decisions[0] as { previousAction?: string }).previousAction;
  assert.throws(() => makePnlReport(missing));
});

test('atomic import preserves previous report when next input is invalid', () => {
  const dir = mkdtempSync(join(tmpdir(), 'jev-pnl-'));
  try {
    const source = join(dir, 'input.json'); writeFileSync(source, JSON.stringify(input()));
    const result = importPnlReport(source, dir);
    const old = readFileSync(result.path, 'utf8');
    writeFileSync(source, JSON.stringify({ ...input(), revision: '' }));
    assert.throws(() => importPnlReport(source, dir));
    assert.equal(readFileSync(result.path, 'utf8'), old);
    assert.equal(readdirSync(dir).some(p => p.endsWith('.tmp')), false);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
