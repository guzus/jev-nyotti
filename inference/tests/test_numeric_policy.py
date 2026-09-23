"""ACTION_POLICY=numeric serving: hash-pinned pure-Python artifact, no Qwen engine, honest identity."""
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jev_inference import action_task, numeric_policy
from jev_inference.action_api import decide
from jev_inference.app import create_app
from jev_inference.settings import Settings

from test_action_api import CUTOFF, HEADERS, KEY, LONG, body, candles

ROOT = Path(__file__).resolve().parents[1]


def artifact(margin=0.5, seed=3) -> dict:
    rng = random.Random(seed)
    families = {}
    for fam, names in numeric_policy.OPTIONS.items():
        n = numeric_policy.N_FEATURES[fam]
        families[fam] = dict(classes=list(names), n_features=n, mean=[rng.gauss(0, 0.1) for _ in range(n)],
                             scale=[0.5 + rng.random() for _ in range(n)], intercept=[rng.gauss(0, 1) for _ in names],
                             coef=[[rng.gauss(0, 0.05) for _ in range(n)] for _ in names])
    return dict(format=numeric_policy.FORMAT, version=1, kind='logreg', task='ACTION_V1', prompt_revision=2,
                hold_margin=margin, families=families, meta={})


def write(tmp_path, model) -> tuple[str, str]:
    path = tmp_path / 'policy.json'
    path.write_bytes(json.dumps(model).encode())
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_settings(path, sha, margin=0.5):
    return Settings(api_key=KEY, device='cpu', action_policy='numeric', numeric_model_path=path,
                    numeric_model_sha256=sha, action_hold_margin=margin)


def never():
    raise AssertionError('numeric mode must not construct the Qwen engine')


@pytest.mark.parametrize('position', [dict(side='flat', entry_price=None, opened_at=None, last_trade_at=None), LONG])
def test_numeric_action_uses_artifact_only(tmp_path, position):
    path, sha = write(tmp_path, artifact())
    with TestClient(create_app(numeric_settings(path, sha), engine_factory=never)) as client:
        health = client.get('/healthz').json()
        assert health['ready'] and health['model'] == 'jev-numeric/logreg' and health['action']['policy'] == 'numeric'
        response = client.post('/action', headers=HEADERS, json=body(position=position))
        assert response.status_code == 200
        out = response.json()
        assert client.post('/score', headers=HEADERS, json={'jobs': []}).status_code == 503
    job = action_task.build_job(candles=candles(), cutoff=CUTOFF, position=position, market=body()['market'])
    logps = numeric_policy.log_probs(numeric_policy.load(path, sha), job['state'])
    names = list(action_task.options_for(position['side']))
    assert out['policy'] == 'numeric' and out['model'] == 'jev-numeric/logreg'
    assert out['revision'] == f'numeric-logreg:sha256:{sha}' and out['inputTokens'] == 0 and out['holdMargin'] == 0.5
    assert [o['name'] for o in out['options']] == names
    for o in out['options']:
        assert math.isclose(o['probability'], math.exp(logps[o['name']]), rel_tol=1e-12)
    assert out['action'] == decide(names, [logps[n] for n in names], 0.5)


def test_numeric_startup_fails_closed(tmp_path):
    path, sha = write(tmp_path, artifact())
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        create_app(numeric_settings(path, 'f' * 64), engine_factory=never)
    with pytest.raises(ValueError, match='hold_margin'):
        create_app(numeric_settings(path, sha, margin=0.0), engine_factory=never)
    bad = artifact()
    bad['families']['flat']['n_features'] = 10
    path, sha = write(tmp_path, bad)
    with pytest.raises(ValueError, match='feature count'):
        create_app(numeric_settings(path, sha), engine_factory=never)


def test_numeric_settings_validation():
    with pytest.raises(ValueError, match='ACTION_POLICY'):
        Settings(api_key=KEY, action_policy='lora_with_prior')
    with pytest.raises(ValueError, match='NUMERIC_MODEL'):
        Settings(api_key=KEY, action_policy='numeric', numeric_model_path='/x')
    with pytest.raises(ValueError, match='only valid'):
        Settings(api_key=KEY, numeric_model_sha256='a' * 64)


def test_numeric_mode_never_imports_torch(tmp_path):
    path, sha = write(tmp_path, artifact())
    script = f'''
import sys, json
from fastapi.testclient import TestClient
sys.path.insert(0, {str(ROOT / "tests")!r})
from test_action_api import HEADERS, body
from jev_inference.app import create_app
from jev_inference.settings import Settings
s = Settings(api_key={KEY!r}, device="cpu", action_policy="numeric", numeric_model_path={path!r},
             numeric_model_sha256={sha!r}, action_hold_margin=0.5)
with TestClient(create_app(s)) as c:
    assert c.post("/action", headers=HEADERS, json=body()).status_code == 200
heavy = sorted(m for m in ("torch", "transformers", "numpy", "sklearn") if m in sys.modules)
print(json.dumps(heavy))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
