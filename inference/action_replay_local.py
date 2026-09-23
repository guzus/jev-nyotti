"""Local CPU stateful ACTION_V1 replay for ACTION_POLICY=numeric (no Modal, no GPU, no Qwen).

python inference/action_replay_local.py --input-file IN.json --out-dir DIR \
  --numeric-model PATH --numeric-sha256 SHA64 [--hold-margin M] [--max-decisions N] [--resume]

Same input manifest, decision loop, checkpoint/resume checks and output.json/status.json/
checkpoint.json format as modal_action_replay.py (plus "policy": "numeric"). The hold margin
defaults to the artifact's validation-tuned margin; a different value is recorded in the identity.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_inference import action_replay as ar  # noqa: E402
from jev_inference import action_task, numeric_policy  # noqa: E402
from jev_inference.replay import iso  # noqa: E402


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--numeric-model', required=True)
    parser.add_argument('--numeric-sha256', required=True)
    parser.add_argument('--hold-margin', type=lambda v: json.loads(v), help='number or JSON {"flat": m, "position": m}')
    parser.add_argument('--max-decisions', type=int, default=10**9)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    model = numeric_policy.load(args.numeric_model, args.numeric_sha256)
    margin = model['hold_margin'] if args.hold_margin is None else args.hold_margin
    spec = {'rules': model['decision_rules']} if model.get('decision_rules') and args.hold_margin is None else margin
    try:  # decide() would silently favour hold on NaN
        action_task.check_margin(margin)
    except ValueError:
        raise SystemExit('--hold-margin must be finite with |value| <= 20') from None
    manifest = json.loads(args.input_file.read_text())
    plan = ar.Plan(manifest, now=time.time())
    identity = ar.identity(manifest, adapter_id='', adapter_revision='', adapter_sha256='',
                           base_revision=numeric_policy.revision(model), hold_margin=spec,
                           policy='numeric', numeric_sha256=model['sha256'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / 'checkpoint.json'
    if checkpoint.exists() and not args.resume:
        raise SystemExit(f'{checkpoint} exists; pass --resume or use a new --out-dir')
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else ar.new_state(identity)
    if state['identity'] != identity:
        raise SystemExit('immutable run input/model/margin mismatch')
    started = time.monotonic()

    def save() -> dict:
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, allow_nan=False))
        temporary.replace(checkpoint)
        out = ar.output(plan, state, model=numeric_policy.model_name(model), revision=numeric_policy.revision(model),
                        margin=margin, generated_at=time.time(), policy='numeric')
        (args.out_dir / 'output.json').write_text(json.dumps(out, allow_nan=False))
        status = dict(completedDecisions=len(state['records']), completedCutoffs=state['completedCutoffs'],
                      marketThrough=out['to'], elapsedSeconds=time.monotonic() - started, identity=identity,
                      contractSha256=ar.CONTRACT_SHA256, scoring='numeric-local', policy='numeric',
                      numericSha256=model['sha256'], generatedAt=iso(time.time()))
        (args.out_dir / 'status.json').write_text(json.dumps(status, allow_nan=False))
        return status

    calls = ar.run(plan, state, ar.numeric_scorer(model), spec, max_decisions=args.max_decisions)
    status = save()  # one write at the end: CPU decisions take microseconds
    print(json.dumps(dict(calls=calls, **{k: status[k] for k in ('completedDecisions', 'completedCutoffs', 'marketThrough')})))
    return status


if __name__ == '__main__':
    main()
