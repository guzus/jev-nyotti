"""Numeric per-state-family action policy. Pure stdlib + math: no sklearn/numpy at serve time.

`feature_vector` is the canonical feature definition (training/action_baseline imports it), read
back from `action_task.build_job(...)['state']` so the numeric model sees exactly the prompt state.
Artifacts are JSON written by training/numeric_export.py and pinned by SHA-256:

  {"format": "jev-numeric-policy", "version": 1, "kind": "logreg" | "hgb", "task": "ACTION_V1",
   "prompt_revision": 2, "hold_margin": float, "families": {"flat": FAMILY, "position": FAMILY}}

  logreg FAMILY: {"classes": [option...], "n_features": n, "mean": [n], "scale": [n],
                  "coef": [[n] per class], "intercept": [per class]}
    raw_k = intercept_k + sum_j coef_kj * (x_j - mean_j) / scale_j
  hgb FAMILY:    {"classes": [...], "n_features": n, "baseline": [per class],
                  "trees": [[tree per class] per iteration]}; tree = list of nodes
                  [feature, threshold, left, right, missing_go_left, is_leaf, value]
    raw_k = baseline_k + sum_iterations tree_k(x); x <= threshold -> left, NaN -> missing side.
  log_probs = log_softmax(raw) over the family's options.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

from . import action_task

FORMAT = 'jev-numeric-policy'
FAMILY = {'flat': 'flat', 'long': 'position', 'short': 'position'}
OPTIONS = {'flat': action_task.FLAT_OPTIONS, 'position': action_task.POSITION_OPTIONS}
FEATURE_KEYS = ('return_1h_pct', 'return_4h_pct', 'return_24h_pct', 'rsi14', 'volatility_pct',
                'from_24h_high_pct', 'from_24h_low_pct', 'last_volume_rel')
N_FEATURES = {'flat': len(FEATURE_KEYS) + 4 * action_task.SHOWN, 'position': len(FEATURE_KEYS) + 4 * action_task.SHOWN + 2}


def feature_vector(state: dict) -> list[float]:
    """Prompt revision 2 state -> features: 8 summary features, 4 per shown candle, then side/unrealized."""
    f, p = state['features'], state['position']  # prompt revision 2: side + unrealized only
    x = [f[k] for k in FEATURE_KEYS]
    for c in state['recent_closed_candles']:
        x += [c['ret_pct'], c['high_pct'], c['low_pct'], c['vol_rel']]
    if p['side'] != 'flat':
        x += [1.0 if p['side'] == 'long' else -1.0, p['unrealized_return_pct']]
    return x


def _reject_constant(value: str) -> None:
    raise ValueError('non-finite numbers are not allowed in a numeric policy artifact')


def _finite(values, what: str) -> None:
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError(f'{what} must be finite numbers')


def _check_tree(tree: list, n: int) -> None:
    if not tree:
        raise ValueError('empty tree')
    for node in tree:
        feature, threshold, left, right, missing_left, is_leaf, value = node
        _finite([threshold, value], 'tree values')
        if is_leaf not in (True, False) or missing_left not in (True, False):
            raise ValueError('tree flags must be booleans')
        if not is_leaf and not (0 <= feature < n and 0 < left < len(tree) and 0 < right < len(tree)):
            raise ValueError('tree node index out of range')


def validate(model: dict) -> dict:
    if model.get('format') != FORMAT or model.get('version') != 1 or model.get('kind') not in ('logreg', 'hgb'):
        raise ValueError('unsupported numeric policy artifact')
    if model.get('task') != action_task.TASK or model.get('prompt_revision') != 2:
        raise ValueError('artifact was not fitted on ACTION_V1 prompt revision 2 features')
    action_task.check_margin(model.get('hold_margin'))
    if 'decision_rules' in model:
        from . import decision_rules
        rules = decision_rules.validate(model['decision_rules'])
        if rules.get('margins', 0.0) != model['hold_margin']:
            raise ValueError('decision_rules.margins must equal hold_margin')
    if set(model['families']) != set(OPTIONS):
        raise ValueError('artifact requires flat and position families')
    for fam, fm in model['families'].items():
        k, n = len(OPTIONS[fam]), fm['n_features']
        if sorted(fm['classes']) != sorted(OPTIONS[fam]) or n != N_FEATURES[fam]:
            raise ValueError(f'{fam}: classes or feature count do not match the ACTION_V1 contract')
        if model['kind'] == 'logreg':
            if len(fm['mean']) != n or len(fm['scale']) != n or len(fm['intercept']) != k or len(fm['coef']) != k:
                raise ValueError(f'{fam}: logreg shape mismatch')
            _finite(fm['mean'] + fm['scale'] + fm['intercept'] + [v for row in fm['coef'] for v in row], 'logreg')
            if any(len(row) != n for row in fm['coef']) or any(s <= 0 for s in fm['scale']):
                raise ValueError(f'{fam}: logreg shape or scale invalid')
        else:
            if len(fm['baseline']) != k or not fm['trees'] or any(len(it) != k for it in fm['trees']):
                raise ValueError(f'{fam}: hgb shape mismatch')
            _finite(fm['baseline'], 'hgb baseline')
            for iteration in fm['trees']:
                for tree in iteration:
                    _check_tree(tree, n)
    return model


def load(path: str, sha256: str) -> dict:
    """Read, hash-verify (before parsing) and validate an artifact. Fails closed."""
    if not re.fullmatch(r'[0-9a-f]{64}', sha256 or ''):
        raise ValueError('numeric policy SHA-256 must be 64 lowercase hex characters')
    with open(path, 'rb') as handle:
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError('numeric policy artifact SHA-256 mismatch')
    model = validate(json.loads(data, parse_constant=_reject_constant))
    model['sha256'] = sha256
    return model


def _tree_value(tree: list, x: list[float]) -> float:
    i = 0
    while True:
        feature, threshold, left, right, missing_left, is_leaf, value = tree[i]
        if is_leaf:
            return value
        v = x[feature]
        i = (left if missing_left else right) if math.isnan(v) else (left if v <= threshold else right)


def raw_scores(fm: dict, kind: str, x: list[float]) -> list[float]:
    if len(x) != fm['n_features']:
        raise ValueError('feature vector length does not match the artifact')
    if kind == 'logreg':
        z = [(v - m) / s for v, m, s in zip(x, fm['mean'], fm['scale'])]
        return [b + math.fsum(w * zj for w, zj in zip(row, z)) for row, b in zip(fm['coef'], fm['intercept'])]
    raw = list(fm['baseline'])
    for iteration in fm['trees']:  # same accumulation order as sklearn _raw_predict
        for k, tree in enumerate(iteration):
            raw[k] += _tree_value(tree, x)
    return raw


def log_probs(model: dict, state: dict) -> dict[str, float]:
    """{option: log p} for the state family of state['position']['side']."""
    fam = FAMILY[state['position']['side']]
    fm = model['families'][fam]
    x = feature_vector(state)
    _finite(x, 'features')
    raw = raw_scores(fm, model['kind'], x)
    top = max(raw)
    lse = top + math.log(math.fsum(math.exp(v - top) for v in raw))
    return {name: v - lse for name, v in zip(fm['classes'], raw)}


def model_name(model: dict) -> str:
    return f"jev-numeric/{model['kind']}"


def revision(model: dict) -> str:
    return f"numeric-{model['kind']}:sha256:{model['sha256']}"
