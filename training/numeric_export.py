"""Export fitted per-state-family sklearn models to the pure-Python numeric policy artifact.

Format and scoring: inference/jev_inference/numeric_policy.py (the serving side never imports sklearn).

  python training/numeric_export.py --pickle .runtime/action-v3-model.pkl --hold-margin 0.1 \
      --out .runtime/numeric-policy.json
prints the artifact SHA-256 to pin in inference/jev_inference/deployment.py. The hold margin must be
the one frozen for that model (ACTION_V3: 0.10, training/ACTION_V3_RESULTS.json "margin").
Before writing, `verify` re-checks the artifact against sklearn predict_proba on random inputs.
The pickle is an `action_baseline.Baseline` (models[family] = (StandardScaler, LogisticRegression, names))
or a dict {family: (scaler_or_None, HistGradientBoostingClassifier, names)}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from jev_inference import numeric_policy as npol  # noqa: E402


def _floats(values) -> list[float]:
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError('non-finite parameter')
    return out


def _classes(model, names) -> list[str]:
    classes = [names[int(c)] for c in model.classes_]
    if len(classes) != len(names):
        raise ValueError('every family option must be a fitted class (refusing to pad missing classes)')
    return classes


def export_logreg(scaler, model, names) -> dict:
    if model.coef_.shape[0] != len(names):
        raise ValueError('expected a multinomial model with one coefficient row per class')
    n = model.coef_.shape[1]
    mean = _floats(scaler.mean_) if scaler is not None and scaler.with_mean else [0.0] * n
    scale = _floats(scaler.scale_) if scaler is not None and scaler.with_std else [1.0] * n
    return dict(classes=_classes(model, names), n_features=n, mean=mean, scale=scale,
                coef=[_floats(row) for row in model.coef_], intercept=_floats(model.intercept_))


def export_hgb(model, names) -> dict:
    if getattr(model, '_preprocessor', None) is not None:
        raise ValueError('categorical preprocessing is not supported')
    classes = _classes(model, names)
    trees = []
    for iteration in model._predictors:
        if len(iteration) != len(classes):
            raise ValueError('expected one tree per class per iteration (multiclass log loss)')
        row = []
        for predictor in iteration:
            nodes = predictor.nodes
            if nodes['is_categorical'].any():
                raise ValueError('categorical splits are not supported')
            # Leaf `value` already includes shrinkage (learning rate).
            row.append([[int(nd['feature_idx']), float(nd['num_threshold']), int(nd['left']), int(nd['right']),
                         bool(nd['missing_go_to_left']), bool(nd['is_leaf']), float(nd['value'])] for nd in nodes])
        trees.append(row)
    return dict(classes=classes, n_features=int(model.n_features_in_),
                baseline=_floats(model._baseline_prediction.reshape(-1)), trees=trees)


def artifact(kind: str, families: dict, hold_margin: float, **meta) -> dict:
    model = dict(format=npol.FORMAT, version=1, kind=kind, task='ACTION_V1', prompt_revision=2,
                 hold_margin=float(hold_margin), families=families, meta=meta)
    return npol.validate(model)


def from_models(models: dict, hold_margin: float, **meta) -> dict:
    """models: {family: (scaler_or_None, sklearn_model, names)} -> validated artifact dict."""
    kinds = {type(m).__name__ for _, m, _ in models.values()}
    if kinds == {'LogisticRegression'}:
        return artifact('logreg', {f: export_logreg(s, m, n) for f, (s, m, n) in models.items()}, hold_margin, **meta)
    if kinds == {'HistGradientBoostingClassifier'}:
        if any(s is not None for s, _, _ in models.values()):
            raise ValueError('hgb export expects raw (unscaled) features')
        return artifact('hgb', {f: export_hgb(m, n) for f, (_, m, n) in models.items()}, hold_margin, **meta)
    raise ValueError(f'unsupported model types {sorted(kinds)}')


def verify(models: dict, model: dict, n: int = 300, seed: int = 0) -> float:
    """Refuse an artifact whose pure-Python probabilities differ from sklearn predict_proba.

    Catches non-softmax models (liblinear/OvR), sklearn private-attribute drift and pickle changes.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    tol = 1e-9 if model['kind'] == 'logreg' else 1e-6
    worst = 0.0
    for fam, (scaler, sk, names) in models.items():
        fm = model['families'][fam]
        centre = np.array(fm['mean']) if model['kind'] == 'logreg' else np.zeros(fm['n_features'])
        spread = np.array(fm['scale']) if model['kind'] == 'logreg' else np.full(fm['n_features'], 3.0)
        X = centre + spread * rng.standard_normal((n, fm['n_features']))
        with np.errstate(all='ignore'):  # macOS Accelerate emits spurious matmul warnings; parity is checked below
            ref = sk.predict_proba(scaler.transform(X) if scaler is not None else X)
        for x, row in zip(X, ref):
            raw = npol.raw_scores(fm, model['kind'], [float(v) for v in x])
            top = max(raw)
            z = math.fsum(math.exp(v - top) for v in raw)
            ours = {c: math.exp(v - top) / z for c, v in zip(fm['classes'], raw)}
            for c, p in zip(sk.classes_, row):
                worst = max(worst, abs(ours[names[int(c)]] - float(p)))
    if worst > tol:
        raise ValueError(f'exported artifact disagrees with sklearn predict_proba (max {worst:.3g} > {tol})')
    return worst


def write(path: Path, model: dict) -> str:
    data = (json.dumps(model, allow_nan=False, sort_keys=True) + '\n').encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pickle', type=Path, required=True)
    parser.add_argument('--hold-margin', type=float, required=True, help='margin frozen on validation')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--note', default='', help='free-text provenance stored in meta')
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f'{args.out} exists; refusing to overwrite')
    import action_baseline  # noqa: F401  (pickle resolves the Baseline class through this module)
    loaded = pickle.loads(args.pickle.read_bytes())  # local trusted training output only
    models = loaded.models if hasattr(loaded, 'models') else loaded
    model = from_models(models, args.hold_margin, source=args.pickle.name, note=args.note,
                        source_sha256=hashlib.sha256(args.pickle.read_bytes()).hexdigest())
    worst = verify(models, model)
    sha = write(args.out, model)
    npol.load(str(args.out), sha)  # round-trip through the serving loader
    print(json.dumps(dict(out=str(args.out), kind=model['kind'], sha256=sha, hold_margin=model['hold_margin'],
                          max_prob_diff_vs_sklearn=worst)))


if __name__ == '__main__':
    main()
