"""PnL lab: closed-loop, fee-inclusive paper evaluation of numeric policy + decision rules.

Uses only candles (no teacher labels), so any Binance 15m window works. Decisions use the exact
serving path: action_task prompt state -> numeric_policy log-probs -> decision_rules.decide ->
action_task.apply_action. Gaps: a cutoff is decided only when its 96-bar lookback is contiguous;
positions carry across gaps and are marked through them.

Windows are directories written by scripts/fetch_binance_replay_market.py (<SYMBOL>-15.jsonl).
The confirmation window must not be passed here until a V5 rule set is frozen (ACTION_V5.md).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
from jev_inference import action_task as at, decision_rules, numeric_policy  # noqa: E402

FEE_RATE = 0.00075
STEP = at.STEP


def load(directory: Path, symbol: str) -> list[dict]:
    rows = [json.loads(line) for line in (directory / f'{symbol}-15.jsonl').read_text().splitlines()]
    rows.sort(key=lambda r: r['time'])
    return rows


def precompute(candles: list[dict], symbol: str) -> list[dict]:
    """Market view per decidable cutoff (contiguous 96-bar lookback)."""
    out, label = [], at.market_label(symbol.removesuffix('USD'))
    for i in range(at.LOOKBACK, len(candles) + 1):
        hist = candles[i - at.LOOKBACK:i]
        if hist[-1]['time'] - hist[0]['time'] != (at.LOOKBACK - 1) * STEP:
            continue
        cutoff = hist[-1]['time'] + STEP
        view = at.market_view(hist)
        nxt = candles[i] if i < len(candles) and candles[i]['time'] == cutoff else None
        out.append(dict(cutoff=cutoff, close=hist[-1]['close'], next_open=nxt['open'] if nxt else None,
                        features=view['features'], recent=view['recent'], market=label))
    return out


def _unit_return(side, entry, price):
    return price / entry - 1 if side == 'long' else entry / price - 1


def simulate(model: dict, rules: dict | None, steps: list[dict], fill: str = 'close') -> dict:
    rules = decision_rules.validate(rules)
    position = at.flat_position()
    realized = fees = 0.0
    peak = max_dd = 0.0
    counts = {a: 0 for a in at.ALL_ACTIONS}
    in_position = 0
    daily = {}
    for s in steps:
        mark = s['close']
        view = at.position_view(position, s['cutoff'], mark)
        state = dict(features=s['features'], recent_closed_candles=s['recent'], position=view)
        logps = numeric_policy.log_probs(model, state)
        names = list(at.options_for(position['side']))
        action = decision_rules.decide(names, [logps[n] for n in names], rules=rules, position=position,
                                       cutoff=s['cutoff'], features=s['features'])
        price = mark if fill == 'close' else s['next_open']
        if price is None:  # next bar missing (gap): cannot execute realistically
            action = 'hold'
            price = mark
        before = position
        after = at.apply_action(before, action, price, s['cutoff'])
        if action in ('reduce', 'close'):
            closed = before['units'] - (after['units'] if after['side'] != 'flat' else 0.0)
            realized += closed * _unit_return(before['side'], before['entry_price'], price)
        fees += FEE_RATE * abs(after['units'] - before['units'])
        position = after
        counts[action] += 1
        in_position += position['side'] != 'flat'
        open_pnl = 0.0 if position['side'] == 'flat' else position['units'] * _unit_return(position['side'], position['entry_price'], mark)
        equity = realized - fees + open_pnl
        peak, max_dd = max(peak, equity), max(max_dd, peak - equity)
        daily[s['cutoff'] // 86400] = equity
    n = len(steps)
    eq = [daily[k] for k in sorted(daily)]
    diffs = [b - a for a, b in zip(eq, eq[1:])]
    mean = sum(diffs) / len(diffs) if diffs else 0.0
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs)) if diffs else 0.0
    return dict(decisions=n, actions=counts, opens=counts['open_long'] + counts['open_short'], closes=counts['close'],
                trades=sum(v for k, v in counts.items() if k != 'hold'),
                time_in_position=in_position / n if n else 0.0,
                net_return_pct=100 * (eq[-1] if eq else 0.0), fees_pct=100 * fees, max_drawdown_pct=100 * max_dd,
                daily_sharpe=(mean / sd * math.sqrt(365)) if sd > 0 else 0.0,
                buy_hold_pct=100 * (steps[-1]['close'] / steps[0]['close'] - 1) if steps else 0.0)


def evaluate(model: dict, rules: dict | None, window: Path, symbols: list[str], fill: str = 'close',
             cache: dict | None = None) -> dict:
    per = {}
    for symbol in symbols:
        key = (str(window), symbol)
        if cache is None or key not in cache:
            steps = precompute(load(window, symbol), symbol)
            if cache is not None:
                cache[key] = steps
        else:
            steps = cache[key]
        per[symbol] = simulate(model, rules, steps, fill)
    agg = {k: sum(r[k] for r in per.values()) / len(per) for k in
           ('net_return_pct', 'fees_pct', 'max_drawdown_pct', 'daily_sharpe', 'time_in_position', 'buy_hold_pct')}
    agg['trades'] = sum(r['trades'] for r in per.values())
    return dict(window=window.name, fill=fill, mean=agg, per_symbol=per)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--rules', default='{}', help='JSON decision rules')
    parser.add_argument('--window', type=Path, action='append', required=True)
    parser.add_argument('--symbols', default='BTCUSD ETHUSD SOLUSD XRPUSD')
    parser.add_argument('--fill', choices=('close', 'next_open'), default='close')
    args = parser.parse_args()
    if any('confirm' in str(w) for w in args.window):
        raise SystemExit('confirmation windows are scored only by training/action_v5.py after the freeze')
    model = numeric_policy.load(args.model, args.sha256)
    rules = json.loads(args.rules)
    for window in args.window:
        result = evaluate(model, rules, window, args.symbols.split(), args.fill)
        print(json.dumps(dict(rules=rules, window=result['window'], fill=args.fill,
                              mean={k: round(v, 3) for k, v in result['mean'].items()},
                              per_symbol={k: {q: round(v[q], 2) for q in ('net_return_pct', 'fees_pct', 'trades', 'buy_hold_pct')}
                                          for k, v in result['per_symbol'].items()})))


if __name__ == '__main__':
    main()
