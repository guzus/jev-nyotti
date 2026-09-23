"""ACTION_V3 (see ACTION_V3.md): fit the pre-registered numeric policy once and score the gate.

Logistic regression per state family on the V2 features, C=0.01, balanced classes, refit on
March train rows + April rows subsampled by the train rule; hold margin frozen at +0.10.
Writes the fitted model (private) and aggregate results (public copy, no per-row data).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import action_baseline as ab  # noqa: E402
import action_eval as ev  # noqa: E402

C = 0.01
MARGIN = 0.10
SEED = 3407
HOLDS_PER_ACTION = 2
THRESHOLDS = {'trade_f1': 0.298050139275766, 'action_macro_f1': 0.10918918291954463}  # ACTION_V2_BASELINE.json


def subsample(rows: list[dict], rng: random.Random) -> list[dict]:
    chosen = [r for r in rows if r['target_action'] != 'hold']
    for side in ('flat', 'long', 'short'):
        holds = [r for r in rows if r['target_action'] == 'hold' and r['side'] == side]
        actions = sum(1 for r in chosen if r['side'] == side)
        chosen += rng.sample(holds, min(len(holds), HOLDS_PER_ACTION * actions))
    return sorted(chosen, key=lambda r: r['cutoff_epoch'])


def rollout_on(model: ab.Baseline, path: Path) -> dict:
    src = json.loads(path.read_text())
    (series,) = src['series']
    start, end = ab_epoch(src['from']), ab_epoch(src['to'])
    candles = [{k: c[k] for k in ('time', 'open', 'high', 'low', 'close', 'volume')} for c in series['candles']]
    result = ev.rollout(candles, start, end,
                        lambda job: ev.decide([o['name'] for o in job['options']], model.logits(job), MARGIN),
                        series['market'])
    summary = {k: v for k, v in result.items() if k != 'decisions'}
    summary.update(window=[src['from'], src['to']], source=f"{src['source']} {series['symbol']}")
    return summary


def ab_epoch(value: str) -> int:
    from datetime import datetime
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--confirm', type=Path, required=True, help='unseen confirmatory action-replay-input.json')
    parser.add_argument('--also', type=Path, help='previously viewed window, PnL report only')
    parser.add_argument('--model-out', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    args = parser.parse_args()
    if args.model_out.exists():
        raise SystemExit('model output exists; V3 is fit once')
    train, val, test = (ab.read(args.dataset / f'{s}.jsonl') for s in ('train', 'validation', 'test'))
    fit_rows = train + subsample(val, random.Random(SEED))
    model = ab.Baseline(fit_rows, C)
    args.model_out.write_bytes(pickle.dumps(model))
    test_metrics = ev.metrics(test, ev.predict(test, [model.logits(r['job']) for r in test], MARGIN))
    confirm = rollout_on(model, args.confirm)
    ratio = test_metrics['predicted_trade_rate'] / test_metrics['teacher_trade_rate']
    criteria = {
        'trade_detection': dict(value=test_metrics['trade_f1'], threshold=THRESHOLDS['trade_f1'],
                                passed=test_metrics['trade_f1'] > THRESHOLDS['trade_f1']),
        'trade_rate_sanity': dict(value=ratio, threshold=[0.5, 2.0], passed=0.5 <= ratio <= 2.0),
        'action_quality': dict(value=test_metrics['action_macro_f1'], threshold=THRESHOLDS['action_macro_f1'],
                               passed=test_metrics['action_macro_f1'] > THRESHOLDS['action_macro_f1']),
        'no_absorbing_state_confirmatory': dict(
            value={k: confirm[k] for k in ('opens', 'closes', 'time_in_position_fraction')},
            threshold={'min_opens': 5, 'min_closes': 5, 'time_in_position': [0.05, 0.95]},
            passed=confirm['opens'] >= 5 and confirm['closes'] >= 5 and 0.05 <= confirm['time_in_position_fraction'] <= 0.95),
    }
    result = dict(task='ACTION_V3', policy='logistic_regression_per_state_family', C=C, margin=MARGIN,
                  fit_rows=len(fit_rows), fit_actions={a: sum(r['target_action'] == a for r in fit_rows)
                                                       for a in ('hold', 'open_long', 'open_short', 'add', 'reduce', 'close')},
                  gate=dict(passed=all(c['passed'] for c in criteria.values()), criteria=criteria),
                  test_metrics={k: test_metrics[k] for k in ('n', 'teacher_trade_rate', 'predicted_trade_rate', 'trade_precision',
                                                              'trade_recall', 'trade_f1', 'action_macro_f1', 'per_class',
                                                              'confusion', 'per_side')},
                  confirmatory_rollout=confirm,
                  previously_viewed_window_rollout=rollout_on(model, args.also) if args.also else None)
    args.public.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(gate=result['gate']['passed'], **{k: v['passed'] for k, v in criteria.items()},
                          f1=test_metrics['trade_f1'], ratio=ratio, macro=test_metrics['action_macro_f1'],
                          confirm={k: confirm[k] for k in ('opens', 'closes', 'time_in_position_fraction', 'net_return_pct',
                                                            'fees_pct', 'max_drawdown_pct')},
                          also={k: result['previously_viewed_window_rollout'][k] for k in ('opens', 'closes', 'net_return_pct')}
                          if args.also else None), indent=1))


if __name__ == '__main__':
    main()
