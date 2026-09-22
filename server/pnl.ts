import type { Candle } from './market.js';

export type PositionSide = 'long' | 'short' | 'flat';
export type PnlDecision = { marketAsOf: string; action: PositionSide; previousAction?: PositionSide };
export type PnlSeries = { symbol: string; decisions: PnlDecision[]; candles: Candle[] };
export type PnlInput = { series: PnlSeries[]; initialCapital?: number; feeBps?: number; slippageBps?: number };
export type EquityPoint = { time: string; equity: number; pnl: number; returnPct: number; drawdownPct: number; buyHoldEquity: number };

const hour = 3600;
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
  const first = series[0];
  requireValid(first.decisions.length > 0, 'at least one decision is required');
  const times = first.decisions.map(d => Date.parse(d.marketAsOf) / 1000);
  requireValid(times.every((t, i) => Number.isFinite(t) && t > 0 && t % hour === 0 && (!i || t - times[i - 1] === hour)), 'decisions must be ordered and contiguous UTC hours');
  for (const s of series) {
    requireValid(s.symbol.trim().length > 0, 'empty symbol');
    requireValid(s.decisions.length === times.length && s.candles.length === times.length, `${s.symbol}: require one execution candle per decision and a common range`);
    s.decisions.forEach((d, i) => {
      requireValid(Date.parse(d.marketAsOf) / 1000 === times[i] && sides.has(d.action), `${s.symbol}: invalid decision or unequal range`);
      requireValid(d.previousAction === undefined || d.previousAction === (i ? s.decisions[i - 1].action : 'flat'), `${s.symbol}: broken prior-position chain`);
      const c = s.candles[i];
      requireValid(c.time === times[i], `${s.symbol}: missing next-open execution candle`);
      requireValid(Object.values(c).every(Number.isFinite) && c.open > 0 && c.low > 0 && c.close > 0 && c.volume >= 0 && c.low <= Math.min(c.open, c.close) && c.high >= Math.max(c.open, c.close), `${s.symbol}: invalid OHLC candle`);
    });
  }
  const fee = feeBps / 10000, slip = slippageBps / 10000;
  const allocation = initialCapital / series.length;
  const accounts = series.map(() => ({ cash: allocation, quantity: 0, side: 'flat' as PositionSide, fees: 0, slippage: 0, fills: 0, peak: allocation, maxDrawdownPct: 0, buyHoldCash: allocation, buyHoldQuantity: 0 }));
  const curve: EquityPoint[] = [{ time: new Date(times[0] * 1000).toISOString(), equity: initialCapital, pnl: 0, returnPct: 0, drawdownPct: 0, buyHoldEquity: initialCapital }];
  let peak = initialCapital, maxDrawdownPct = 0;
  for (let i = 0; i < times.length; i++) {
    let equity = 0, buyHoldEquity = 0;
    series.forEach((s, j) => {
      const a = accounts[j], c = s.candles[i], target = s.decisions[i].action;
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
      if (i === 0) {
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
    curve.push({ time: new Date((times[i] + hour) * 1000).toISOString(), equity, pnl: equity - initialCapital, returnPct: (equity / initialCapital - 1) * 100, drawdownPct, buyHoldEquity });
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
    assumptions: { feeBps, slippageBps, intervalMinutes: 60, initialPosition: 'flat', allocation: 'equal_initial_sleeves', sizing: '1x_equity_on_side_change', execution: 'next_candle_open_at_input_cutoff', mark: 'hourly_close', endingPosition: 'marked_to_market_not_liquidated', fundingIncluded: false, borrowCostsIncluded: false, liquidationModeled: false, dividendsIncluded: false, reconstruction: true },
  };
}
