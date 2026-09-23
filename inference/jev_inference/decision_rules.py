"""Deterministic decision layer on top of per-option log-probabilities (pure stdlib).

The same function serves live /action, the replay and the PnL lab, so a rule tuned in the lab
is exactly what production runs. Rules use only the carried paper position and the prompt
state (no future data). An empty/None rule set reproduces plain margin decisions.

rules = {
  "margins": {"flat": m, "position": m} | m,  # subtracted from hold (see action_task.margin_for)
  "allowed": ["open_long", ...],               # executed actions permitted (hold always allowed)
  "min_hold_bars": n,       # no reduce/close until the position is n bars old
  "cooldown_bars": n,       # no open until n bars after the last execution
  "open_prob_min": p,       # an open needs softmax probability >= p
  "trend_filter": "return_4h_pct" | "return_24h_pct" | None,  # longs need >0, shorts need <0
}
"""
from __future__ import annotations

import math

from . import action_task

STEP = action_task.STEP
KEYS = {'margins', 'allowed', 'min_hold_bars', 'cooldown_bars', 'open_prob_min', 'trend_filter'}
TREND_FEATURES = ('return_1h_pct', 'return_4h_pct', 'return_24h_pct')


def validate(rules: dict | None) -> dict:
    rules = dict(rules or {})
    if set(rules) - KEYS:
        raise ValueError(f'unknown decision rules: {sorted(set(rules) - KEYS)}')
    action_task.check_margin(rules.get('margins', 0.0))
    allowed = rules.get('allowed')
    if allowed is not None and (not set(allowed) <= set(action_task.ALL_ACTIONS) - {'hold'}):
        raise ValueError('allowed must list executed actions')
    for key in ('min_hold_bars', 'cooldown_bars'):
        value = rules.get(key, 0)
        if type(value) is not int or not 0 <= value <= 10_000:
            raise ValueError(f'{key} must be a non-negative integer')
    p = rules.get('open_prob_min', 0.0)
    if not isinstance(p, (int, float)) or not 0 <= p <= 1:
        raise ValueError('open_prob_min must be within [0, 1]')
    if rules.get('trend_filter') not in (None, *TREND_FEATURES):
        raise ValueError('unsupported trend_filter')
    return rules


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def decide(names: list[str], logits: list[float], *, rules: dict | None, position: dict, cutoff: int,
           features: dict) -> str:
    """Margin argmax over the actions the rules permit at this state; hold if none remain."""
    rules = rules or {}
    margin = action_task.margin_for(rules.get('margins', 0.0), position['side'])
    probs = dict(zip(names, _softmax(logits)))
    allowed = set(rules.get('allowed') or action_task.ALL_ACTIONS) | {'hold'}
    blocked = set()
    last, opened = position.get('last_trade_at'), position.get('opened_at')
    if position['side'] == 'flat':
        if rules.get('cooldown_bars') and last is not None and cutoff - last < rules['cooldown_bars'] * STEP:
            blocked |= {'open_long', 'open_short'}
        trend = rules.get('trend_filter')
        if trend:
            if not features[trend] > 0:
                blocked.add('open_long')
            if not features[trend] < 0:
                blocked.add('open_short')
        p_min = rules.get('open_prob_min', 0.0)
        blocked |= {n for n in ('open_long', 'open_short') if probs.get(n, 0.0) < p_min}
    elif rules.get('min_hold_bars') and opened is not None and cutoff - opened < rules['min_hold_bars'] * STEP:
        blocked |= {'reduce', 'close'}
    candidates = [(v - (margin if n == 'hold' else 0.0), -i, n) for i, (n, v) in enumerate(zip(names, logits))
                  if n in allowed and n not in blocked]
    return max(candidates)[2] if candidates else 'hold'
