"""Train/serve parity: exported pure-Python numeric policy vs sklearn predict_proba on real prompt states."""
import hashlib
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'inference')]

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import action_baseline  # noqa: E402
import numeric_export as ne  # noqa: E402
from jev_inference import action_task as at  # noqa: E402
from jev_inference import numeric_policy as npol  # noqa: E402

CUTOFF = 1_758_585_600


def states(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        price, candles = 100.0 + k, []
        for i in range(at.LOOKBACK):
            o = price
            price *= 1 + rng.gauss(0, 0.004)
            candles.append(dict(time=CUTOFF - (at.LOOKBACK - i) * at.STEP, open=o, high=max(o, price) * (1 + rng.random() / 500),
                                low=min(o, price) * (1 - rng.random() / 500), close=price, volume=rng.random() * 50))
        side = ('flat', 'long', 'short')[k % 3]
        pos = at.flat_position() if side == 'flat' else dict(side=side, entry_price=price * (1 + rng.gauss(0, 0.01)),
                                                             opened_at=CUTOFF - 900, last_trade_at=None)
        out.append(at.build_job(candles=candles, cutoff=CUTOFF, position=pos, market='BTC/USD')['state'])
    return out


def fit(kind: str, data: list[dict]) -> dict:
    models = {}
    for fam, names in ne.npol.OPTIONS.items():
        rows = [s for s in data if npol.FAMILY[s['position']['side']] == fam]
        X = np.array([npol.feature_vector(s) for s in rows])
        # Label depends on a feature so the model learns non-trivial structure; cycle guarantees every class.
        y = np.array([(i if i < len(names) else int(x[0] > 0) + int(x[1] > 0)) % len(names) for i, x in enumerate(X)])
        if kind == 'logreg':
            scaler = StandardScaler().fit(X)
            models[fam] = (scaler, LogisticRegression(C=0.5, max_iter=5000).fit(scaler.transform(X), y), names)
        else:
            models[fam] = (None, HistGradientBoostingClassifier(max_iter=15, max_leaf_nodes=7, min_samples_leaf=5,
                                                                early_stopping=False, random_state=0).fit(X, y), names)
    return models


def sklearn_probs(models, state) -> dict:
    scaler, model, names = models[npol.FAMILY[state['position']['side']]]
    x = np.array([npol.feature_vector(state)])
    p = model.predict_proba(scaler.transform(x) if scaler is not None else x)[0]
    return {names[int(c)]: float(v) for c, v in zip(model.classes_, p)}


class NumericExportParity(unittest.TestCase):
    train, test = states(240, 1), states(60, 2)

    def roundtrip(self, kind: str, tol: float) -> dict:
        models = fit(kind, self.train)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.json'
            exported = ne.from_models(models, 0.25, note='test')
            self.assertLessEqual(ne.verify(models, exported), tol)
            sha = ne.write(path, exported)
            loaded = npol.load(str(path), sha)
            with self.assertRaises(ValueError):
                npol.load(str(path), '0' * 64)
        self.assertEqual((loaded['kind'], loaded['hold_margin']), (kind, 0.25))
        for state in self.test:
            ours = npol.log_probs(loaded, state)
            ref = sklearn_probs(models, state)
            self.assertEqual(set(ours), set(at.options_for(state['position']['side'])))
            self.assertTrue(all(math.isfinite(v) for v in ours.values()))
            for name, p in ref.items():
                self.assertAlmostEqual(math.exp(ours[name]), p, delta=tol)
        return loaded

    def test_logreg_parity(self):
        self.roundtrip('logreg', 1e-9)

    def test_hgb_parity_including_missing_branch(self):
        loaded = self.roundtrip('hgb', 1e-6)
        _, model, _ = fit('hgb', self.train)['flat']
        state = self.test[0]
        x = npol.feature_vector(state)
        x[0] = float('nan')  # not produced by build_job, but the tree walk must follow sklearn's missing side
        raw = npol.raw_scores(loaded['families']['flat'], 'hgb', x)
        ref = model._raw_predict(np.array([x]))[0]
        for a, b in zip(raw, ref):
            self.assertAlmostEqual(a, float(b), delta=1e-9)

    def test_feature_vector_is_canonical_and_sized(self):
        self.assertIs(action_baseline.vector, npol.feature_vector)
        for s in self.test[:3]:
            self.assertEqual(len(npol.feature_vector(s)), npol.N_FEATURES[npol.FAMILY[s['position']['side']]])

    def test_missing_class_refused(self):
        X = np.array([npol.feature_vector(s) for s in self.train if s['position']['side'] == 'flat'])
        y = np.arange(len(X)) % 2
        model = LogisticRegression(max_iter=5000).fit(StandardScaler().fit_transform(X), y)
        with self.assertRaises(ValueError):
            ne.export_logreg(None, model, at.FLAT_OPTIONS)

    def test_verify_refuses_non_softmax_model(self):
        models = fit('logreg', self.train)
        scaler, _, names = models['flat']
        X = np.array([npol.feature_vector(s) for s in self.train if s['position']['side'] == 'flat'])
        y = np.arange(len(X)) % 3
        ovr = LogisticRegression(solver='liblinear').fit(scaler.transform(X), y)  # one-vs-rest probabilities
        models['flat'] = (scaler, ovr, names)
        with self.assertRaises(ValueError):
            ne.verify(models, ne.from_models(models, 0.0))

    def test_nonfinite_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            path.write_bytes(b'{"hold_margin": NaN}')
            with self.assertRaises(ValueError):
                npol.load(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
            json.loads(path.read_text())  # plain json would have accepted it


if __name__ == '__main__':
    unittest.main()
