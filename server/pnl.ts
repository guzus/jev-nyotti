import type { Candle } from './market.js';
import {
  ACTION_INTERVAL_MINUTES, ACTION_STEP_SECONDS, FEE_BPS_PER_UNIT, MAX_UNITS, flatPosition, optionsFor, stepPaper, unrealizedPct,
  type ActionName, type PaperSide,
} from './paper.js';

export type PositionSide = 'long' | 'short' | 'flat';
export type PnlDecision = { marketAsOf: string; action: PositionSide; previousAction?: PositionSide };
export type PnlSeries = { symbol: string; decisions: PnlDecision[]; candles: Candle[] };
export type PnlInput = { series: PnlSeries[]; initialCapital?: number; feeBps?: number; slippageBps?: number; intervalMinutes?: 60 | 240; from?: string; to?: string };
export type EquityPoint = { time: string; equity: number; pnl: number; returnPct: number; drawdownPct: number; buyHoldEquity: number };

const sides = new Set(['long', 'short', 'flat']);
function requireValid(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Invalid PnL input: ${message}`);
}

/** Historical reconstruction only: a decision cutoff is the close of its input
 * candle, and the OPEN of the following execution candle. This idealized fill
 * does not assert that a later-generated live decision was executable then.
 * Each sleeve keeps its quantity until the target side changes; it is sized at
 * 1x sleeve equity on entry, without cross-symbol rebalancing. */
export function simulatePortfolio(input: PnlInput) {
  const { series, initialCapital = 10000, feeBps = 5, slippageBps = 2 } = input;
  requireValid(Number.isFinite(initialCapital) && initialCapital > 0, 'initial capital must be positive');
  requireValid([feeBps, slippageBps].every(x => Number.isFinite(x) && x >= 0 && x < 10000), 'costs must be in [0, 10000) bps');
  requireValid(series.length > 0 && new Set(series.map(s => s.symbol)).size === series.length, 'symbols must be nonempty and unique');
  const intervalMinutes = input.intervalMinutes ?? 60;
  requireValid(intervalMinutes === 60 || intervalMinutes === 240, 'interval must be 60 or 240 minutes');
  const step = intervalMinutes * 60;
  const first = series[0];
  requireValid(first.decisions.length > 0, 'at least one decision is required');
  requireValid((input.from === undefined) === (input.to === undefined), 'from and to must be supplied together');
  const from = input.from ? Date.parse(input.from) / 1000 : Date.parse(first.decisions[0].marketAsOf) / 1000;
  const to = input.to ? Date.parse(input.to) / 1000 : Date.parse(first.decisions.at(-1)!.marketAsOf) / 1000 + step;
  requireValid(Number.isFinite(from) && Number.isFinite(to) && from > 0 && from % step === 0 && to % step === 0 && to > from, 'range must be aligned UTC intervals');
  const times = Array.from({ length: (to - from) / step }, (_, i) => from + i * step);
  const offsets: number[] = [];
  for (const s of series) {
    requireValid(s.symbol.trim().length > 0, 'empty symbol');
    requireValid(s.decisions.length > 0 && s.candles.length === s.decisions.length, `${s.symbol}: require one execution candle per decision`);
    const start = Date.parse(s.decisions[0].marketAsOf) / 1000;
    requireValid(start >= from && start < to && (start - from) % step === 0, `${s.symbol}: invalid start`);
    if (!input.from) requireValid(start === from, `${s.symbol}: require a common range or explicit from/to`);
    offsets.push((start - from) / step);
    s.decisions.forEach((d, i) => {
      const time = start + i * step;
      requireValid(Date.parse(d.marketAsOf) / 1000 === time && sides.has(d.action), `${s.symbol}: decisions must be ordered and contiguous UTC intervals`);
      requireValid(d.previousAction === undefined || d.previousAction === (i ? s.decisions[i - 1].action : 'flat'), `${s.symbol}: broken prior-position chain`);
      const c = s.candles[i];
      requireValid(c.time === time, `${s.symbol}: missing next-open execution candle`);
      requireValid(Object.values(c).every(Number.isFinite) && c.open > 0 && c.low > 0 && c.close > 0 && c.volume >= 0 && c.low <= Math.min(c.open, c.close) && c.high >= Math.max(c.open, c.close), `${s.symbol}: invalid OHLC candle`);
    });
    requireValid(start + s.decisions.length * step === to, `${s.symbol}: missing trailing execution data`);
  }
  const fee = feeBps / 10000, slip = slippageBps / 10000;
  const allocation = initialCapital / series.length;
  const accounts = series.map(() => ({ cash: allocation, quantity: 0, side: 'flat' as PositionSide, fees: 0, slippage: 0, fills: 0, peak: allocation, maxDrawdownPct: 0, buyHoldCash: allocation, buyHoldQuantity: 0 }));
  const curve: EquityPoint[] = [{ time: new Date(times[0] * 1000).toISOString(), equity: initialCapital, pnl: 0, returnPct: 0, drawdownPct: 0, buyHoldEquity: initialCapital }];
  let peak = initialCapital, maxDrawdownPct = 0;
  for (let i = 0; i < times.length; i++) {
    let equity = 0, buyHoldEquity = 0;
    series.forEach((s, j) => {
      const a = accounts[j], index = i - offsets[j];
      // Before the asset's first available execution bar its allocated sleeve
      // remains cash; no invented historical price is used for either strategy.
      if (index < 0) { equity += a.cash; buyHoldEquity += a.buyHoldCash; return; }
      const c = s.candles[index], target = s.decisions[index].action;
      const trade = (delta: number) => {
        const fillPrice = c.open * (1 + Math.sign(delta) * slip);
        const cost = Math.abs(delta * fillPrice) * fee;
        a.cash -= delta * fillPrice + cost;
        a.quantity += delta;
        a.fees += cost;
        a.slippage += Math.abs(delta) * c.open * slip;
        a.fills++;
      };
      requireValid(a.cash + a.quantity * c.open > 0, `${s.symbol}: insolvent sleeve; liquidation is not modeled`);
      if (a.side !== target) {
        if (a.quantity !== 0) trade(-a.quantity);
        if (target !== 'flat') {
          requireValid(a.cash > 0, `${s.symbol}: costs exhausted sleeve`);
          // Reserve execution costs so entry never borrows principal. Short
          // uses the same conservative entry notional sizing as long.
          const quantity = a.cash / (c.open * (1 + slip) * (1 + fee));
          trade(target === 'long' ? quantity : -quantity);
        }
        a.side = target;
      }
      if (index === 0) {
        a.buyHoldQuantity = allocation / (c.open * (1 + slip) * (1 + fee));
        a.buyHoldCash -= a.buyHoldQuantity * c.open * (1 + slip) * (1 + fee);
      }
      const value = a.cash + a.quantity * c.close;
      // Reject intrabar insolvency too, instead of silently carrying a short
      // through a liquidation event and later reporting a recovered equity.
      const worst = a.cash + a.quantity * (a.quantity < 0 ? c.high : c.low);
      requireValid(worst > 0 && value > 0, `${s.symbol}: insolvent sleeve; liquidation is not modeled`);
      a.peak = Math.max(a.peak, value);
      a.maxDrawdownPct = Math.max(a.maxDrawdownPct, (1 - value / a.peak) * 100);
      equity += value;
      buyHoldEquity += a.buyHoldCash + a.buyHoldQuantity * c.close;
    });
    peak = Math.max(peak, equity);
    const drawdownPct = (1 - equity / peak) * 100;
    maxDrawdownPct = Math.max(maxDrawdownPct, drawdownPct);
    curve.push({ time: new Date((times[i] + step) * 1000).toISOString(), equity, pnl: equity - initialCapital, returnPct: (equity / initialCapital - 1) * 100, drawdownPct, buyHoldEquity });
  }
  const end = curve.at(-1)!;
  return {
    initialCapital, finalEquity: end.equity, pnl: end.pnl, returnPct: end.returnPct, maxDrawdownPct,
    buyHoldEquity: end.buyHoldEquity, buyHoldReturnPct: (end.buyHoldEquity / initialCapital - 1) * 100,
    fees: accounts.reduce((sum, a) => sum + a.fees, 0), slippage: accounts.reduce((sum, a) => sum + a.slippage, 0),
    curve,
    perSymbol: series.map((s, j) => {
      const a = accounts[j], last = s.candles.at(-1)!;
      const equity = a.cash + a.quantity * last.close;
      return { symbol: s.symbol, initialCapital: allocation, equity, pnl: equity - allocation, returnPct: (equity / allocation - 1) * 100, maxDrawdownPct: a.maxDrawdownPct, fees: a.fees, slippage: a.slippage, fills: a.fills, endingAction: a.side, buyHoldEquity: a.buyHoldCash + a.buyHoldQuantity * last.close };
    }),
    assumptions: { feeBps, slippageBps, intervalMinutes, inputIntervalMinutes: 60, predictionHorizonMinutes: 60, holdingPolicy: 'hold_target_until_next_decision', unavailableBeforeStart: 'cash', initialPosition: 'flat', allocation: 'equal_initial_sleeves', sizing: '1x_equity_on_side_change', execution: 'next_candle_open_at_input_cutoff', mark: intervalMinutes === 60 ? 'hourly_close' : 'four_hour_close', endingPosition: 'marked_to_market_not_liquidated', fundingIncluded: false, borrowCostsIncluded: false, liquidationModeled: false, dividendsIncluded: false, reconstruction: true },
  };
}

export type ActionReplayDecision = { marketAsOf: string; action: ActionName; sideAfter: PaperSide; unitsAfter: number; price: number; probabilities: Partial<Record<ActionName, number>> };
export type ActionReplayInput = { from: string; to: string; series: { symbol: string; decisions: ActionReplayDecision[]; candles: Candle[] }[] };
export type ActionCurvePoint = { time: string; pnlPct: number; drawdownPts: number; buyHoldPct: number };

/** ACTION_V1 closed-loop replay accounting. Every 15m cutoff in [from, to) has one
 * candle and one decision; the decision executes at that candle's close through the
 * same stepPaper rule the live server uses (1 unit, add +1 max 3, reduce halves,
 * 7.5 bps per unit traded). PnL is percent of one unit notional; the portfolio line
 * is the equal-weight mean of symbols. The imported sideAfter/unitsAfter chain must
 * match the rule exactly, otherwise the import fails closed. */
export function simulateActionReplay(input: ActionReplayInput) {
  const from = Date.parse(input.from) / 1000, to = Date.parse(input.to) / 1000;
  requireValid(Number.isFinite(from) && Number.isFinite(to) && from % ACTION_STEP_SECONDS === 0 && to % ACTION_STEP_SECONDS === 0 && to > from, 'range must be aligned 15m UTC boundaries');
  requireValid(input.series.length > 0 && new Set(input.series.map(s => s.symbol)).size === input.series.length, 'symbols must be nonempty and unique');
  const bars = (to - from) / ACTION_STEP_SECONDS;
  const sleeves = input.series.map(s => {
    requireValid(s.candles.length === bars && s.decisions.length === bars, `${s.symbol}: require one 15m candle and one decision per cutoff in range`);
    let position = flatPosition(), realized = 0, fees = 0, trades = 0, opens = 0, closes = 0, exposed = 0, peak = 0, maxDrawdownPts = 0;
    const first = s.candles[0].close, points: { pnl: number; buyHold: number }[] = [];
    s.candles.forEach((c, i) => {
      const cutoff = c.time + ACTION_STEP_SECONDS, d = s.decisions[i];
      requireValid(c.time === from + i * ACTION_STEP_SECONDS, `${s.symbol}: candles must be contiguous 15m bars`);
      requireValid(Object.values(c).every(Number.isFinite) && c.low > 0 && c.low <= Math.min(c.open, c.close) && c.high >= Math.max(c.open, c.close) && c.volume >= 0, `${s.symbol}: invalid OHLC candle`);
      requireValid(Date.parse(d.marketAsOf) / 1000 === cutoff, `${s.symbol}: decision must sit at its candle close`);
      requireValid(Math.abs(d.price / c.close - 1) < 1e-9, `${s.symbol}: execution price must be the cutoff candle close`);
      const allowed = optionsFor(position.side), probs = Object.entries(d.probabilities);
      requireValid(allowed.includes(d.action) && probs.length === allowed.length && probs.every(([k, v]) => allowed.includes(k as ActionName) && Number.isFinite(v) && v >= 0) && Math.abs(probs.reduce((a, [, v]) => a + v, 0) - 1) < 1e-4, `${s.symbol}: action or probabilities do not match the carried state`);
      const step = stepPaper(position, d.action, c.close, cutoff);
      requireValid(step.after.side === d.sideAfter && Math.abs(step.after.units - d.unitsAfter) < 1e-12, `${s.symbol}: decision chain does not match the paper execution rule`);
      position = step.after; realized += step.realizedPct; fees += step.feePct;
      if (step.unitsTraded > 0) trades++;
      if (d.action === 'open_long' || d.action === 'open_short') opens++;
      if (d.action === 'close') closes++;
      if (position.side !== 'flat') exposed++;
      const pnl = realized + unrealizedPct(position, c.close);
      peak = Math.max(peak, pnl); maxDrawdownPts = Math.max(maxDrawdownPts, peak - pnl);
      points.push({ pnl, buyHold: 100 * (c.close / first - 1) - FEE_BPS_PER_UNIT / 100 });
    });
    const mark = s.candles.at(-1)!.close;
    return { symbol: s.symbol, points, row: {
      symbol: s.symbol, pnlPct: realized + unrealizedPct(position, mark), realizedPct: realized, unrealizedPct: unrealizedPct(position, mark), feesPct: fees,
      trades, opens, closes, timeInPositionPct: 100 * exposed / bars, maxDrawdownPts, endingSide: position.side, endingUnits: position.units, buyHoldPct: points.at(-1)!.buyHold,
    } };
  });
  const mean = (f: (s: typeof sleeves[number]) => number) => sleeves.reduce((a, s) => a + f(s), 0) / sleeves.length;
  let peak = 0, maxDrawdownPts = 0;
  const curve: ActionCurvePoint[] = [{ time: new Date(from * 1000).toISOString(), pnlPct: 0, drawdownPts: 0, buyHoldPct: 0 }];
  for (let i = 0; i < bars; i++) {
    const pnlPct = mean(s => s.points[i].pnl);
    peak = Math.max(peak, pnlPct); maxDrawdownPts = Math.max(maxDrawdownPts, peak - pnlPct);
    curve.push({ time: new Date((from + (i + 1) * ACTION_STEP_SECONDS) * 1000).toISOString(), pnlPct, drawdownPts: peak - pnlPct, buyHoldPct: mean(s => s.points[i].buyHold) });
  }
  const perSymbol = sleeves.map(s => s.row);
  return {
    kind: 'action_units' as const, pnlPct: curve.at(-1)!.pnlPct, buyHoldPct: curve.at(-1)!.buyHoldPct, maxDrawdownPts,
    feesPct: mean(s => s.row.feesPct), trades: perSymbol.reduce((a, r) => a + r.trades, 0), curve, perSymbol,
    assumptions: { intervalMinutes: ACTION_INTERVAL_MINUTES, feeBpsPerUnit: FEE_BPS_PER_UNIT, maxUnits: MAX_UNITS, pnlUnit: 'percent_of_one_unit_notional' as const,
      portfolio: 'equal_weight_mean_of_symbols' as const, execution: 'cutoff_candle_close' as const, sizing: 'open_1_add_1_max_3_reduce_half_close_flat' as const,
      initialPosition: 'flat' as const, endingPosition: 'marked_to_market_not_liquidated' as const, buyHold: 'one_unit_long_from_first_cutoff_close' as const,
      fundingIncluded: false as const, slippageIncluded: false as const, reconstruction: true as const },
  };
}
