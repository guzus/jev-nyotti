// TypeScript port of inference/jev_inference/action_task.py paper execution
// (options_for + apply_action). The Python module is the frozen contract;
// test/paper.test.ts checks parity against it. Do not change semantics here
// without changing the contract.

export const ACTION_TASK = 'ACTION_V1';
export const ACTION_INTERVAL_MINUTES = 15;
export const ACTION_STEP_SECONDS = ACTION_INTERVAL_MINUTES * 60;
export const ACTION_LOOKBACK = 96;
export const MAX_UNITS = 3;
/** Taker fee charged per paper unit traded (7.5 bps of one unit's notional). */
export const FEE_BPS_PER_UNIT = 7.5;

export type PaperSide = 'flat' | 'long' | 'short';
export type ActionName = 'hold' | 'open_long' | 'open_short' | 'add' | 'reduce' | 'close';
export const FLAT_OPTIONS: readonly ActionName[] = ['hold', 'open_long', 'open_short'];
export const POSITION_OPTIONS: readonly ActionName[] = ['hold', 'add', 'reduce', 'close'];
export const ALL_ACTIONS: readonly ActionName[] = ['hold', 'open_long', 'open_short', 'add', 'reduce', 'close'];

/** Times are UTC epoch seconds, as in action_task.py. */
export type PaperPosition = {
  side: PaperSide; units: number; entry_price: number | null; opened_at: number | null; last_trade_at: number | null;
};

export function flatPosition(): PaperPosition {
  return { side: 'flat', units: 0, entry_price: null, opened_at: null, last_trade_at: null };
}

export function optionsFor(side: PaperSide): readonly ActionName[] {
  if (side === 'flat') return FLAT_OPTIONS;
  if (side === 'long' || side === 'short') return POSITION_OPTIONS;
  throw new Error('side must be flat, long or short');
}

/** Exact port of action_task.apply_action. */
export function applyAction(position: PaperPosition, action: ActionName, price: number, t: number): PaperPosition {
  if (!optionsFor(position.side).includes(action)) throw new Error('action not allowed for position side');
  const p: PaperPosition = { ...position };
  if (action === 'hold') return p;
  p.last_trade_at = t;
  if (action === 'open_long' || action === 'open_short') {
    Object.assign(p, { side: action === 'open_long' ? 'long' : 'short', units: 1, entry_price: price, opened_at: t });
  } else if (action === 'add') {
    const units = p.units;
    const added = Math.min(1, MAX_UNITS - units);
    if (added > 0) {
      p.entry_price = (units + added) / (units / p.entry_price! + added / price);
      p.units = units + added;
    }
  } else if (action === 'reduce') {
    p.units = p.units / 2;
  } else if (action === 'close') {
    Object.assign(p, { side: 'flat', units: 0, entry_price: null, opened_at: null });
  }
  return p;
}

/** Per-unit signed return in percent, identical to action_task.position_view
 * (long: mark/entry − 1, short: entry/mark − 1) so the displayed and accounted
 * return is exactly what the model is shown. Unrounded. */
export function unitReturnPct(side: PaperSide, entry: number | null, mark: number): number {
  if (side === 'flat' || !entry) return 0;
  return 100 * (side === 'long' ? mark / entry - 1 : entry / mark - 1);
}

export type PaperStep = {
  before: PaperPosition; after: PaperPosition; action: ActionName; price: number; t: number;
  /** |Δunits| actually traded; `add` at MAX_UNITS trades 0. */
  unitsTraded: number;
  /** Fee in percent of one unit notional. */
  feePct: number;
  /** Realized PnL of this step in percent of one unit notional (return on closed units − fee). */
  realizedPct: number;
};

/** Single accounting rule shared by the live worker and the replay importer.
 * PnL is expressed in percent of one paper unit's notional. */
export function stepPaper(before: PaperPosition, action: ActionName, price: number, t: number): PaperStep {
  if (!(Number.isFinite(price) && price > 0)) throw new Error('price must be positive');
  const after = applyAction(before, action, price, t);
  let unitsTraded = 0, closedUnits = 0;
  if (action === 'open_long' || action === 'open_short') unitsTraded = after.units;
  else if (action === 'add') unitsTraded = after.units - before.units;
  else if (action === 'reduce' || action === 'close') { closedUnits = before.units - after.units; unitsTraded = closedUnits; }
  const feePct = unitsTraded * FEE_BPS_PER_UNIT / 100;
  // `|| 0` normalizes -0 on no-trade steps so stored JSON stays stable.
  const realizedPct = (closedUnits * unitReturnPct(before.side, before.entry_price, price) - feePct) || 0;
  return { before, after, action, price, t, unitsTraded, feePct, realizedPct };
}

export function unrealizedPct(position: PaperPosition, mark: number): number {
  return position.units * unitReturnPct(position.side, position.entry_price, mark);
}
