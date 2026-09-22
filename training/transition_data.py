"""Versioned market-only ablation; historical exposure labels, not buy/sell fills."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from real_data import (HOUR, LOOKBACK, MODEL, REVISION, iso, load_candles, make_example,
                       position_before, read_positions, sha256, side, split_for)

TASK = 'NEXT_HOUR_POSITION_SIDE_MARKET_ONLY_V2'
INSTRUCTIONS = (
    "Predict the supplied historical trader's position side at the end of the next hour "
    "using only this closed-candle market snapshot. The trader's previous position is unavailable. "
    "Long and short describe signed XBTUSD contract exposure; flat means zero contracts. "
    "This is market-only historical exposure imitation, not price direction, a recommendation or an order. "
    "Do not infer missing balances, leverage, news or future executions."
)


def market_only_example(history, previous, target, split, rng):
    row = make_example(history, previous, target, split, rng)
    del row['job']['state']['position_side_before_cutoff']
    row['job']['state']['missing'].append('actual_trader_position')
    row['job']['instructions'] = INSTRUCTIONS
    # previous_action is evaluation metadata only; encode_example reads only job + target_index.
    return row


def select_training(rows, rng, count=2048):
    changed = [r for r in rows if r[1] != r[2]]
    unchanged = [r for r in rows if r[1] == r[2]]
    if len(changed) < count//2 or len(unchanged) < count//2:
        raise ValueError('insufficient unique transition/stable training examples')
    return sorted(rng.sample(changed, count//2) + rng.sample(unchanged, count//2))


def prepare(source, candles_path, output):
    if output.exists():
        raise ValueError('use a new versioned output directory')
    paths = sorted(source.glob('aoa-execution-*.csv'))
    if len(paths) != 4:
        raise ValueError('requires four execution exports')
    times, positions, reconciliation = read_positions(paths)
    candles = load_candles(candles_path)
    groups = {s: [] for s in ('train','validation','test')}
    for i in range(LOOKBACK-1, len(candles)):
        t = candles[i]['end']
        if t < times[0] or t+HOUR > times[-1]:
            continue
        split = split_for(t)
        if split:
            groups[split].append((i, side(position_before(times,positions,t)), side(position_before(times,positions,t+HOUR))))
    rng = random.Random(3411)
    natural = sorted(rng.sample(groups['validation'],256))
    events = [r for r in groups['validation'] if r[1] != r[2]]
    # Predetermined calendar window, never selected for outcomes; continuous hourly sequence.
    rollout = groups['validation'][:96]
    selection = dict(train=select_training(groups['train'],rng),
                     validation=sorted(set(natural+events+rollout)),
                     test=sorted(rng.sample(groups['test'],128)))
    manifest = dict(task=TASK,model=MODEL,revision=REVISION,provenance='user_supplied_aoa_execution_export',
        source_files=[dict(name=p.name,sha256=sha256(p)) for p in paths],
        market_file=dict(sha256=sha256(candles_path),source='BitMEX XBTUSD hourly',candles=len(candles)),
        reconciliation=reconciliation,files={},splits={},
        cohorts={name:[iso(candles[r[0]]['end']) for r in rows] for name,rows in
                 [('natural',natural),('transitions',events),('rollout',rollout)]},
        experiment=dict(previous_position_input=False,train_transition_fraction=.5,max_steps=512,
                        evaluate_test=False,production_promoted=False),
        limitations=['Exposure-side labels discard intrahour trades and size changes.',
                     'No cross-asset, spot-market, four-hour or profitability validation.',
                     'Validation was previously inspected; this is diagnosis, not a final independent test.',
                     'No flat targets in 2021 validation/test; flat-entry and exit generalization unestablished.',
                     'Teacher previous_action is metadata only and absent from model input.'])
    output.mkdir(parents=True)
    for split,rows in selection.items():
        path=output/f'{split}.jsonl'
        with path.open('w') as f:
            for i,prior,target in rows:
                f.write(json.dumps(market_only_example(candles[i-LOOKBACK+1:i+1],prior,target,split,rng),separators=(',',':'))+'\n')
        manifest['files'][path.name]=dict(sha256=sha256(path),bytes=path.stat().st_size)
        manifest['splits'][split]=dict(count=len(rows),eligible_count=len(groups[split]),
            labels=dict(Counter(r[2] for r in rows)),transitions=sum(r[1]!=r[2] for r in rows),
            transition_matrix=dict(Counter(f'{r[1]}->{r[2]}' for r in rows)))
    manifest['dataset_id']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def diagnostics(examples, results, cohorts):
    from run_real import summarize_predictions, classification_metrics
    output={}
    for name,cutoffs in cohorts.items():
        wanted=set(cutoffs)
        pairs=[(r,p) for r,p in zip(examples,results) if r['cutoff'] in wanted]
        if len(pairs)!=len(wanted):
            raise ValueError('incomplete evaluation cohort')
        rows,preds=map(list,zip(*pairs))
        actions=[r['job'].options[p['prediction']].name for r,p in pairs]
        targets=[r['target_action'] for r in rows]
        previous=[r['previous_action'] for r in rows]
        proposed=[i for i,a in enumerate(actions) if a!=previous[i]]
        true_changes=[i for i,t in enumerate(targets) if t!=previous[i]]
        correct=sum(actions[i]==targets[i] for i in proposed)
        m=summarize_predictions(rows,preds)
        stable=[i for i,t in enumerate(targets) if t==previous[i]]
        m.update(predicted_actions=dict(Counter(actions)),proposed_changes=len(proposed),
            correct_transition_precision=correct/len(proposed) if proposed else None,
            correct_transition_recall=correct/len(true_changes) if true_changes else None,
            false_change_rate=sum(actions[i]!=previous[i] for i in stable)/len(stable) if stable else None,
            always_flat=classification_metrics(targets,['flat']*len(targets)),
            training_majority_short=classification_metrics(targets,['short']*len(targets)))
        if name=='rollout':
            stamps=[r['cutoff'] for r in rows]
            from real_data import timestamp
            if any(timestamp(b)-timestamp(a)!=HOUR for a,b in zip(stamps,stamps[1:])):
                raise ValueError('rollout must be contiguous')
            carried=['flat']+actions[:-1]
            m['flat_start_rollout']=dict(hours=len(rows),entries=sum(p=='flat' and a!='flat' for p,a in zip(carried,actions)),
                exits=sum(p!='flat' and a=='flat' for p,a in zip(carried,actions)),
                reversals=sum(p!='flat' and a!='flat' and p!=a for p,a in zip(carried,actions)),
                note='Market-only model has no position input; chronology carries exposure for transition counts, not feedback.')
        output[name]=m
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('source','candles','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    m=prepare(args.source,args.candles,args.output)
    print(json.dumps({k:m[k] for k in ('dataset_id','splits','limitations')},indent=2))


def diagnostic_gate(before, after):
    """Predeclared diagnostic screen, not sufficient for serving or profitability."""
    natural, events = after['natural'], after['transitions']
    checks = {
        'nonconstant_natural_output': len(natural['predicted_actions']) > 1,
        'beats_base_natural_macro_f1': natural['macro_f1'] > before['natural']['macro_f1'],
        'beats_majority_natural_macro_f1': natural['macro_f1'] > natural['training_majority_short']['macro_f1'],
        'natural_accuracy_within_2pp_of_persistence': natural['accuracy'] >= natural['persistence_baseline']['accuracy']-.02,
        'correct_transition_recall_at_least_10pct': (events['correct_transition_recall'] or 0) >= .10,
        'natural_correct_transition_precision_at_least_20pct': (natural['correct_transition_precision'] or 0) >= .20,
        'natural_false_change_rate_at_most_10pct': natural['false_change_rate'] is not None and natural['false_change_rate'] <= .10,
    }
    return dict(passed=all(checks.values()), checks=checks, production_promoted=False,
                note='Diagnostic validation only. Promotion additionally needs independent test, matching serving inputs and cost-aware rollout.')


def validate_cohorts(rows, cohorts):
    from real_data import timestamp
    if set(cohorts) != {'natural','transitions','rollout'}:
        raise ValueError('unexpected diagnostic cohorts')
    lookup={r['cutoff']:r for r in rows}
    for name,cutoffs in cohorts.items():
        if len(cutoffs)!=len(set(cutoffs)) or any(t not in lookup for t in cutoffs):
            raise ValueError('invalid diagnostic cohort membership')
    if len(cohorts['natural'])!=256 or len(cohorts['rollout'])!=96:
        raise ValueError('incorrect fixed cohort sizes')
    expected={r['cutoff'] for r in rows if r['target_action']!=r['previous_action']}
    if set(cohorts['transitions'])!=expected or len(expected)<20:
        raise ValueError('transition challenge does not cover all selected transitions')
    sequence=cohorts['rollout']
    if any(timestamp(b)-timestamp(a)!=HOUR for a,b in zip(sequence,sequence[1:])):
        raise ValueError('noncontiguous diagnostic rollout')
