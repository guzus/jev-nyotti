"""ACTION_V1 numeric baseline: logistic regression on the same unit-free prompt state.

One multinomial model per state family (flat: hold/open_long/open_short; in position:
hold/add/reduce/close). Fit on train, margin tuned on validation, reported once on test.
Also reports always-hold. Features are read back from job['state'] so the baseline sees
exactly what the language model sees (minus text).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_eval as ev  # noqa: E402

FAMILY = {'flat': 'flat', 'long': 'position', 'short': 'position'}
OPTIONS = {'flat': ('hold', 'open_long', 'open_short'), 'position': ('hold', 'add', 'reduce', 'close')}


def vector(state: dict) -> list[float]:
    f, p = state['features'], state['position']
    since = p.get('minutes_since_last_execution')
    x = [f[k] for k in ('return_1h_pct', 'return_4h_pct', 'return_24h_pct', 'rsi14', 'volatility_pct',
                        'from_24h_high_pct', 'from_24h_low_pct', 'last_volume_rel')]
    for c in state['recent_closed_candles'][-4:]:
        x += [c['ret_pct'], c['high_pct'], c['low_pct'], c['vol_rel']]
    x += [0.0 if since is None else math.log1p(since), 1.0 if since is None else 0.0]
    if p['side'] != 'flat':
        x += [1.0 if p['side'] == 'long' else -1.0, p['unrealized_return_pct'], math.log1p(p['position_age_minutes'])]
    return x


class Baseline:
    def __init__(self, train_rows: list[dict]):
        self.models = {}
        for fam, names in OPTIONS.items():
            rows = [r for r in train_rows if FAMILY[r['side']] == fam]
            X = np.array([vector(r['job']['state']) for r in rows])
            y = np.array([names.index(r['target_action']) for r in rows])
            scaler = StandardScaler().fit(X)
            model = LogisticRegression(C=0.5, class_weight='balanced', max_iter=5000).fit(scaler.transform(X), y)
            self.models[fam] = (scaler, model, names)

    def logits(self, job: dict) -> list[float]:
        scaler, model, names = self.models[FAMILY[job['state']['position']['side']]]
        logp = model.predict_log_proba(scaler.transform([vector(job['state'])]))[0]
        by_name = {names[int(c)]: float(v) for c, v in zip(model.classes_, logp)}
        return [by_name.get(o['name'], -1e9) for o in job['options']]


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def strip(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != 'decisions'}


def run(dataset: Path) -> dict:
    train, val, test = (read(dataset / f'{s}.jsonl') for s in ('train', 'validation', 'test'))
    model = Baseline(train)
    tuned = ev.tune_margin(val, [model.logits(r['job']) for r in val])
    margin = tuned['margin']
    test_metrics = ev.metrics(test, ev.predict(test, [model.logits(r['job']) for r in test], margin))
    roll = json.loads((dataset / 'rollout.json').read_text())
    rollout = ev.rollout(roll['candles'], roll['start'], roll['end'],
                         lambda job: ev.decide([o['name'] for o in job['options']], model.logits(job), margin),
                         roll['market'])
    hold_test = ev.metrics(test, ['hold'] * len(test))
    hold_rollout = ev.rollout(roll['candles'], roll['start'], roll['end'], lambda job: 'hold', roll['market'])
    manifest = json.loads((dataset / 'manifest.json').read_text())
    return dict(task='ACTION_V1', dataset_id=manifest['dataset_id'], model='logistic_regression_per_state_family',
                margin=margin, margin_rule=tuned['rule'], validation_metrics=tuned['val_metrics'],
                test_metrics=test_metrics, rollout=strip(rollout),
                gate_if_this_were_the_model=ev.gate(test_metrics, test_metrics, rollout),
                always_hold=dict(test_metrics=hold_test, rollout=strip(hold_rollout)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--public-copy', type=Path)
    args = parser.parse_args()
    out = args.dataset / 'baseline_results.json'
    if out.exists():
        raise SystemExit(f'{out} exists; refusing to overwrite')
    result = run(args.dataset)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if args.public_copy:
        args.public_copy.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')  # aggregates only
    t, r = result['test_metrics'], result['rollout']
    print(json.dumps(dict(margin=result['margin'], rule=result['margin_rule'],
                          val_trade_f1=result['validation_metrics']['trade_f1'], test_trade_f1=t['trade_f1'],
                          test_rate=[t['teacher_trade_rate'], t['predicted_trade_rate']],
                          test_macro_f1=t['action_macro_f1'], rollout={k: r[k] for k in (
                              'opens', 'closes', 'time_in_position_fraction', 'net_return_pct', 'max_drawdown_pct')}),
                     indent=1))


if __name__ == '__main__':
    main()
