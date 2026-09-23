"""Local numeric replay: same format as the Modal replay, same scoring as live /action, resumable."""
import json
import sys
from pathlib import Path

import pytest

from jev_inference import action_replay, action_task, numeric_policy
from jev_inference.action_api import decide

from test_action_replay import manifest
from test_numeric_policy import artifact, write

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import action_replay_local  # noqa: E402


def run(tmp_path, out, *extra):
    m = manifest(steps=6)
    (tmp_path / 'in.json').write_text(json.dumps(m))
    path, sha = write(tmp_path, artifact(margin=0.3))
    args = ['--input-file', str(tmp_path / 'in.json'), '--out-dir', str(tmp_path / out),
            '--numeric-model', path, '--numeric-sha256', sha, *extra]
    return m, path, sha, args


def test_local_numeric_replay_matches_live_decision_rule(tmp_path):
    m, path, sha, args = run(tmp_path, 'full')
    status = action_replay_local.main(args)
    out = json.loads((tmp_path / 'full' / 'output.json').read_text())
    assert status['completedDecisions'] == 12 and status['policy'] == 'numeric'
    assert (out['model'], out['policy'], out['revision'], out['holdMargin']) == (
        'jev-numeric/logreg', 'numeric', f'numeric-logreg:sha256:{sha}', 0.3)
    model = numeric_policy.load(path, sha)
    checkpoint = json.loads((tmp_path / 'full' / 'checkpoint.json').read_text())
    plan = action_replay.Plan(m, now=10**12)
    positions = {s: action_task.flat_position() for s in plan.symbols}
    for i, record in enumerate(checkpoint['records']):
        cutoff = plan.start + (i // 2) * action_task.STEP
        job = plan.job(record['symbol'], cutoff, positions[record['symbol']])
        logps = numeric_policy.log_probs(model, job['state'])
        names = [o['name'] for o in job['options']]
        assert record['action'] == decide(names, [logps[n] for n in names], 0.3)
        positions[record['symbol']] = action_task.apply_action(positions[record['symbol']], record['action'], record['price'], cutoff)
    assert checkpoint['identity'] != action_replay.identity(m, adapter_id='', adapter_revision='', adapter_sha256='',
                                                            base_revision=out['revision'], hold_margin=0.3)


def test_local_numeric_replay_resumes_to_identical_records(tmp_path):
    _, _, _, full = run(tmp_path, 'full')
    action_replay_local.main(full)
    _, _, _, part = run(tmp_path, 'part')
    action_replay_local.main([*part, '--max-decisions', '4'])
    with pytest.raises(SystemExit, match='--resume'):
        action_replay_local.main(part)
    action_replay_local.main([*part, '--resume'])
    with pytest.raises(SystemExit, match='mismatch'):
        action_replay_local.main([*part, '--resume', '--hold-margin', '0.0'])
    a, b = (json.loads((tmp_path / d / 'checkpoint.json').read_text()) for d in ('full', 'part'))
    assert a == b
