"""ACTION_V1 scoring: margin decision rule, teacher-forced metrics, closed-loop rollout and gate.

Pure numpy/stdlib. The decision rule is `argmax(logits + b)` where `b` subtracts a margin
from the `hold` logit only (ties go to the first option). The margin is tuned on validation
and frozen before test is scored. See training/ACTION_V1.md.
"""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from jev_inference import action_task as at  # noqa: E402

FEE_RATE = 0.00075  # 7.5 bps taker fee per unit traded
RATE_BOUNDS = (0.5, 2.0)
DEFAULT_GRID = tuple(round(x, 2) for x in np.arange(-10.0, 10.0 + 1e-9, 0.05))
EXECUTED = ('open_long', 'open_short', 'add', 'reduce', 'close')
GATE_ROLLOUT = {'min_opens': 5, 'min_closes': 5, 'time_in_position': (0.05, 0.95)}


def softmax_probs(logits) -> list[float]:
    x = np.asarray(logits, dtype=float)
    x = np.exp(x - x.max())
    return (x / x.sum()).tolist()


def decide(option_names, logits, margin: float) -> str:
    """argmax after subtracting `margin` from the hold logit only; ties -> first option."""
    names = list(option_names)
    if len(names) != len(logits):
        raise ValueError('option/logit length mismatch')
    adjusted = [float(v) - (margin if n == 'hold' else 0.0) for n, v in zip(names, logits)]
    return names[max(range(len(names)), key=lambda i: (adjusted[i], -i))]


def option_names(row: dict) -> list[str]:
    return [o['name'] for o in row['job']['options']]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)


def _trade_block(teacher: list[str], pred: list[str]) -> dict:
    n = len(teacher)
    tt = [a != 'hold' for a in teacher]
    pt = [a != 'hold' for a in pred]
    tp = sum(a and b for a, b in zip(tt, pt))
    fp = sum(b and not a for a, b in zip(tt, pt))
    fn = sum(a and not b for a, b in zip(tt, pt))
    p, rc, f1 = _prf(tp, fp, fn)
    return dict(n=n, teacher_trade_rate=sum(tt) / n if n else 0.0, predicted_trade_rate=sum(pt) / n if n else 0.0,
                trade_precision=p, trade_recall=rc, trade_f1=f1,
                accuracy=sum(a == b for a, b in zip(teacher, pred)) / n if n else 0.0)


def metrics(rows: list[dict], predictions: list[str]) -> dict:
    if len(rows) != len(predictions):
        raise ValueError('rows/predictions length mismatch')
    teacher = [r['target_action'] for r in rows]
    out = _trade_block(teacher, predictions)
    classes = sorted({a for a in teacher + list(predictions) if a != 'hold'}, key=at.ALL_ACTIONS.index)
    per_class = {}
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(teacher, predictions))
        fp = sum(t != c and p == c for t, p in zip(teacher, predictions))
        fn = sum(t == c and p != c for t, p in zip(teacher, predictions))
        prec, rec, f1 = _prf(tp, fp, fn)
        per_class[c] = dict(precision=prec, recall=rec, f1=f1, support=sum(t == c for t in teacher))
    out['action_macro_f1'] = sum(v['f1'] for v in per_class.values()) / len(per_class) if per_class else 0.0
    out['action_classes'] = classes
    out['per_class'] = per_class
    labels = [a for a in at.ALL_ACTIONS if a in set(teacher) | set(predictions)]
    pairs = Counter(zip(teacher, predictions))
    out['confusion'] = {'labels': labels, 'matrix': [[pairs[(t, p)] for p in labels] for t in labels],
                        'rows': 'teacher', 'columns': 'predicted'}
    out['per_side'] = {}
    for side in ('flat', 'long', 'short'):
        idx = [i for i, r in enumerate(rows) if r['side'] == side]
        if idx:
            out['per_side'][side] = _trade_block([teacher[i] for i in idx], [predictions[i] for i in idx])
    return out


def predict(rows: list[dict], logits_list, margin: float) -> list[str]:
    return [decide(option_names(r), lg, margin) for r, lg in zip(rows, logits_list)]


def tune_margin(rows: list[dict], logits_list, grid=None) -> dict:
    """Max validation trade F1 with predicted rate in [0.5, 2]x teacher; ties -> smallest |margin|.

    If no margin satisfies the rate band, pick the one minimizing |log(rate ratio)| (a zero
    predicted rate counts as infinitely far unless the teacher rate is also zero).
    """
    grid = list(DEFAULT_GRID if grid is None else grid)
    if not grid:
        raise ValueError('empty margin grid')
    candidates = []
    for m in grid:
        mt = metrics(rows, predict(rows, logits_list, float(m)))
        teacher, pred = mt['teacher_trade_rate'], mt['predicted_trade_rate']
        if teacher == 0:
            ratio_distance = 0.0 if pred == 0 else math.inf
        elif pred == 0:
            ratio_distance = math.inf
        else:
            ratio_distance = abs(math.log(pred / teacher))
        within = teacher > 0 and RATE_BOUNDS[0] <= pred / teacher <= RATE_BOUNDS[1]
        candidates.append((float(m), mt, within, ratio_distance))
    feasible = [c for c in candidates if c[2]]
    if feasible:
        best = min(feasible, key=lambda c: (-c[1]['trade_f1'], abs(c[0]), c[0]))
        rule = 'max_trade_f1_within_rate_band'
    else:
        best = min(candidates, key=lambda c: (c[3], abs(c[0]), c[0]))
        rule = 'closest_rate_ratio_fallback'
    return dict(margin=best[0], val_metrics=best[1], rule=rule, grid_size=len(grid))


def _unit_return(side: str, entry: float, price: float) -> float:
    """Per-unit return, same convention as action_task.position_view unrealized_return_pct."""
    return price / entry - 1 if side == 'long' else entry / price - 1


def rollout(candles15m: list[dict], start_epoch: int, end_epoch: int,
            decide_fn: Callable[[dict], str], market: str) -> dict:
    """Closed-loop paper rollout from flat. Candle `time` is OPEN; cutoff c executes at the close of
    the candle that closes at c. PnL is in % of 1 unit notional: realized unit returns minus
    7.5 bps per unit traded, plus mark-to-market of any open position at the close at `end`."""
    if start_epoch % at.STEP or end_epoch % at.STEP or end_epoch <= start_epoch:
        raise ValueError('rollout bounds must be ordered 15-minute boundaries')
    index = {c['time']: i for i, c in enumerate(candles15m)}
    position = at.flat_position()
    realized = fees = 0.0
    peak, max_dd = 0.0, 0.0
    counts = Counter()
    decisions, in_position, opens, closes = [], 0, 0, 0

    def equity(price: float) -> float:
        open_pnl = 0.0 if position['side'] == 'flat' else position['units'] * _unit_return(position['side'], position['entry_price'], price)
        return realized - fees + open_pnl

    for cutoff in range(start_epoch, end_epoch, at.STEP):
        last = index.get(cutoff - at.STEP)
        if last is None or last < at.LOOKBACK - 1:
            raise ValueError(f'missing candles for cutoff {at.iso(cutoff)}')
        hist = candles15m[last - at.LOOKBACK + 1:last + 1]
        price = hist[-1]['close']
        job = at.build_job(candles=hist, cutoff=cutoff, position=position, market=market)
        action = decide_fn(job)
        names = [o['name'] for o in job['options']]
        if action not in names:
            raise ValueError(f'decision {action!r} is not an option for side {position["side"]}')
        before = position
        after = at.apply_action(before, action, price, cutoff)
        traded = abs(after['units'] - before['units'])  # add at 3 units trades 0; reduce trades half
        if action in ('reduce', 'close'):
            closed_units = before['units'] - (after['units'] if after['side'] != 'flat' else 0.0)
            realized += closed_units * _unit_return(before['side'], before['entry_price'], price)
        fees += FEE_RATE * traded
        position = after
        counts[action] += 1
        opens += action in ('open_long', 'open_short')
        closes += action == 'close'
        in_position += position['side'] != 'flat'
        eq = equity(price)
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
        decisions.append(dict(cutoff=at.iso(cutoff), action=action, side_after=position['side'],
                              units_after=position['units'], price=price))
    final = index.get(end_epoch - at.STEP)
    if final is None:
        raise ValueError('missing final candle for mark-to-market')
    final_eq = equity(candles15m[final]['close'])
    peak = max(peak, final_eq)
    max_dd = max(max_dd, peak - final_eq)
    n = len(decisions)
    return dict(n_decisions=n, actions={a: counts.get(a, 0) for a in at.ALL_ACTIONS}, opens=opens, closes=closes,
                time_in_position_fraction=in_position / n if n else 0.0,
                net_return_pct=100 * final_eq, realized_return_pct=100 * realized, fees_pct=100 * fees,
                max_drawdown_pct=100 * max_dd, open_at_end=position['side'] != 'flat',
                accounting='unit notional; realized unit returns (position_view convention) - 7.5 bps x units traded; '
                           'open position marked to the close at end without a closing fee',
                decisions=decisions)


def gate(model_test: dict, baseline_test: dict, model_rollout: dict) -> dict:
    """Exactly the four ACTION_V1.md promotion criteria, all on test with the frozen margin."""
    detection_threshold = max(0.0, baseline_test['trade_f1'])  # always-hold trade F1 is 0
    ratio = (model_test['predicted_trade_rate'] / model_test['teacher_trade_rate']
             if model_test['teacher_trade_rate'] > 0 else math.inf)
    lo, hi = GATE_ROLLOUT['time_in_position']
    tip = model_rollout['time_in_position_fraction']
    criteria = {
        'trade_detection': dict(value=model_test['trade_f1'],
                                threshold={'greater_than': detection_threshold, 'always_hold': 0.0,
                                           'baseline': baseline_test['trade_f1']},
                                passed=model_test['trade_f1'] > detection_threshold),
        'trade_rate_sanity': dict(value=ratio if math.isfinite(ratio) else None,
                                  threshold={'min': RATE_BOUNDS[0], 'max': RATE_BOUNDS[1]},
                                  passed=math.isfinite(ratio) and RATE_BOUNDS[0] <= ratio <= RATE_BOUNDS[1]),
        'action_quality': dict(value=model_test['action_macro_f1'],
                               threshold={'greater_than': baseline_test['action_macro_f1']},
                               passed=model_test['action_macro_f1'] > baseline_test['action_macro_f1']),
        'no_absorbing_state': dict(value={'opens': model_rollout['opens'], 'closes': model_rollout['closes'],
                                          'time_in_position_fraction': tip},
                                   threshold={'min_opens': GATE_ROLLOUT['min_opens'],
                                              'min_closes': GATE_ROLLOUT['min_closes'],
                                              'time_in_position': [lo, hi]},
                                   passed=(model_rollout['opens'] >= GATE_ROLLOUT['min_opens']
                                           and model_rollout['closes'] >= GATE_ROLLOUT['min_closes']
                                           and lo <= tip <= hi)),
    }
    return dict(passed=all(c['passed'] for c in criteria.values()), criteria=criteria)
