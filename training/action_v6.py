"""ACTION_V6 (see ACTION_V6.md): select on validation, then score the frozen winner once on test.

  python training/action_v6.py select --dataset .runtime/action-v6
  python training/action_v6.py test   --dataset .runtime/action-v6 --v4 .runtime/numeric-policy-v4.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_baseline as ab  # noqa: E402
import action_eval as ev  # noqa: E402

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[1]
FAMILY = ab.FAMILY
OPTIONS = ab.OPTIONS


def tod(state: dict) -> list[float]:
    t = datetime.strptime(state['data_cutoff'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    h = t.hour + t.minute / 60
    return [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)]


class Model:
    """Per-family classifier; `logits(job)` returns log-probs in the job's option order."""

    def __init__(self, rows, kind: str, c: float = 0.01, time_of_day: bool = False):
        self.kind, self.time_of_day, self.models = kind, time_of_day, {}
        for fam, names in OPTIONS.items():
            fam_rows = [r for r in rows if FAMILY[r['side']] == fam]
            X = np.array([self.vector(r['job']['state']) for r in fam_rows])
            y = np.array([names.index(r['target_action']) for r in fam_rows])
            if kind == 'logreg':
                scaler = StandardScaler().fit(X)
                clf = LogisticRegression(C=c, class_weight='balanced', max_iter=5000).fit(scaler.transform(X), y)
            else:
                scaler = None
                weights = {k: len(y) / (len(set(y)) * (y == k).sum()) for k in set(y)}
                clf = HistGradientBoostingClassifier(random_state=0, max_depth=3, learning_rate=0.05, max_iter=200,
                                                     l2_regularization=1.0, min_samples_leaf=40).fit(
                    X, y, sample_weight=np.array([weights[v] for v in y]))
            self.models[fam] = (scaler, clf, names)

    def vector(self, state):
        return ab.vector(state) + (tod(state) if self.time_of_day else [])

    def logits(self, job):
        scaler, clf, names = self.models[FAMILY[job['state']['position']['side']]]
        x = np.array([self.vector(job['state'])])
        lp = np.log(np.clip(clf.predict_proba(scaler.transform(x) if scaler is not None else x)[0], 1e-12, 1))
        by = {names[int(k)]: float(v) for k, v in zip(clf.classes_, lp)}
        return [by.get(o['name'], -27.0) for o in job['options']]


def rate_matched_margins(rows, logits) -> dict:
    out = {}
    for fam in ('flat', 'position'):
        idx = [i for i, r in enumerate(rows) if FAMILY[r['side']] == fam]
        sub = [rows[i] for i in idx]
        teacher = sum(r['target_action'] != 'hold' for r in sub) / len(sub)
        best = None
        for k in range(-600, 601):
            m = k / 100
            pred = sum(ev.decide([o['name'] for o in rows[i]['job']['options']], logits[i], m) != 'hold' for i in idx) / len(idx)
            d = abs(math.log(max(pred, 1e-9) / max(teacher, 1e-9)))
            if best is None or d < best[0] - 1e-12:
                best = (d, m)
        out[fam] = best[1]
    return out


def predict(rows, logits, margins):
    return [ev.decide([o['name'] for o in r['job']['options']], lg, margins['flat' if r['side'] == 'flat' else 'position'])
            for r, lg in zip(rows, logits)]


CANDIDATES = [dict(kind='logreg', c=c) for c in (0.003, 0.01, 0.03, 0.1)] + \
             [dict(kind='logreg', c=0.01, time_of_day=True), dict(kind='hgb')]


def select(dataset: Path) -> None:
    train, val = ab.read(dataset / 'train.jsonl'), ab.read(dataset / 'validation.jsonl')
    results = []
    for cand in CANDIDATES:
        model = Model(train, **cand)
        lg = [model.logits(r['job']) for r in val]
        margins = rate_matched_margins(val, lg)
        m = ev.metrics(val, predict(val, lg, margins))
        ratio = m['predicted_trade_rate'] / m['teacher_trade_rate']
        results.append(dict(candidate=cand, margins=margins, val_trade_f1=m['trade_f1'], val_macro=m['action_macro_f1'],
                            val_ratio=ratio, eligible=0.9 <= ratio <= 1.1))
        print(json.dumps(results[-1]), flush=True)
    winner = max((r for r in results if r['eligible']), key=lambda r: r['val_trade_f1'])
    (dataset / 'v6_selection.json').write_text(json.dumps(dict(results=results, winner=winner), indent=1))
    model = Model(train, **winner['candidate'])
    (dataset / 'v6_model.pkl').write_bytes(pickle.dumps(model))
    print('WINNER', json.dumps(winner))


def test(dataset: Path, v4_path: Path, v4_sha: str) -> None:
    sys.path.insert(0, str(ROOT / 'inference'))
    from jev_inference import numeric_policy
    sel = json.loads((dataset / 'v6_selection.json').read_text())['winner']
    model = pickle.loads((dataset / 'v6_model.pkl').read_bytes())
    tst = ab.read(dataset / 'test.jsonl')
    mv6 = ev.metrics(tst, predict(tst, [model.logits(r['job']) for r in tst], sel['margins']))
    v4 = numeric_policy.load(v4_path, v4_sha)

    def v4_logits(job):
        lp = numeric_policy.log_probs(v4, job['state'])
        return [lp[o['name']] for o in job['options']]
    mv4 = ev.metrics(tst, predict(tst, [v4_logits(r['job']) for r in tst], v4['hold_margin']))
    roll = json.loads((dataset / 'rollout.json').read_text())
    margins = sel['margins']
    r = ev.rollout(roll['candles'], roll['start'], roll['end'],
                   lambda job: ev.decide([o['name'] for o in job['options']], model.logits(job),
                                         margins['flat' if job['state']['position']['side'] == 'flat' else 'position']),
                   roll['market'])
    ratio = mv6['predicted_trade_rate'] / mv6['teacher_trade_rate']
    crit = {
        'trade_f1_vs_v4': dict(value=mv6['trade_f1'], threshold=mv4['trade_f1'], passed=mv6['trade_f1'] > mv4['trade_f1']),
        'macro_f1_vs_v4': dict(value=mv6['action_macro_f1'], threshold=mv4['action_macro_f1'],
                               passed=mv6['action_macro_f1'] > mv4['action_macro_f1']),
        'trade_rate': dict(value=ratio, threshold=[0.5, 2], passed=0.5 <= ratio <= 2),
        'non_absorbing_2022': dict(value={k: r[k] for k in ('opens', 'closes', 'time_in_position_fraction')},
                                   passed=r['opens'] >= 5 and r['closes'] >= 5 and 0.05 <= r['time_in_position_fraction'] <= 0.95),
    }
    keep = ('n', 'teacher_trade_rate', 'predicted_trade_rate', 'trade_precision', 'trade_recall', 'trade_f1',
            'action_macro_f1', 'per_class', 'confusion', 'per_side')
    out = dict(task='ACTION_V6', selection=sel, gate=dict(passed=all(c['passed'] for c in crit.values()), criteria=crit),
               test_v6={k: mv6[k] for k in keep}, test_v4={k: mv4[k] for k in keep},
               rollout_2022={k: v for k, v in r.items() if k != 'decisions'})
    (ROOT / 'training' / 'ACTION_V6_RESULTS.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: (v['passed'], v['value'] if not isinstance(v['value'], dict) else v['value'],
                          v.get('threshold')) for k, v in crit.items()}, default=str))
    print('ROLLOUT', {k: r[k] for k in ('opens', 'closes', 'time_in_position_fraction', 'net_return_pct', 'fees_pct')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('select', 'test'))
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--v4', type=Path)
    p.add_argument('--v4-sha', default='9101f3f5d92418f0de055962354c729c47c05c3b864ea9288bbccfc666992392')
    a = p.parse_args()
    select(a.dataset) if a.stage == 'select' else test(a.dataset, a.v4, a.v4_sha)
