"""ACTION_V1 bounded LoRA training + evaluation runner (see training/ACTION_V1.md).

Import-safe helpers validate the dataset, encode rows with the serving formatter and hold
the budget constants. Only main() imports the GPU stack. Raw rows, per-row logits and
rollout decisions (absolute prices) stay in the private artifact volume, never report.json.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from jev_inference import action_task as at  # noqa: E402
from jev_inference.settings import MAX_INPUT_TOKENS  # noqa: E402
from run_real import MODEL_ID, MODEL_REVISION, reject_identifier_fields, tensor_hash, valid_id  # noqa: E402

SPLITS = ('train', 'validation', 'test')
MANIFEST_FILES = tuple(f'{s}.jsonl' for s in SPLITS) + ('rollout.json',)
INPUT_FILES = MANIFEST_FILES + ('manifest.json', 'baseline_results.json')
ROW_FIELDS = {'job', 'target_index', 'target_action', 'cutoff', 'cutoff_epoch', 'split', 'side'}
MAX_LENGTH = MAX_INPUT_TOKENS  # every trained/scored row must also be servable
MAX_FILE_BYTES = 100 * 1024 * 1024
LORA_RANK = 16
MAX_STEPS = 600  # ~2 epochs of 1,220 train rows at 4 sequences/step
BATCH_SIZE, GRAD_ACCUM = 2, 2
TRAIN_WALL_SECONDS = 900  # hard cap on the optimizer loop
MIN_STEPS = 10

# Budget: worker timeout + fixed startup/teardown overhead at a conservative H100 rate.
RATE_USD_PER_SECOND = 0.0013
OVERHEAD_SECONDS = 300
MAX_SECONDS = 1800
BUDGET_USD = 3.0


def planned_cost_usd(max_seconds: float = MAX_SECONDS) -> float:
    return RATE_USD_PER_SECOND * (max_seconds + OVERHEAD_SECONDS)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_dataset_id(manifest: dict) -> str:
    """Same content address action_data.build assigns: sha256(manifest minus id, sort_keys)[:16]."""
    body = {k: v for k, v in manifest.items() if k != 'dataset_id'}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _check_row(row: dict, split: str) -> None:
    from jev_inference.schemas import Job
    if set(row) != ROW_FIELDS:
        raise ValueError('row fields')
    reject_identifier_fields(row)
    Job.model_validate(row['job'])
    state = row['job']['state']
    side = row['side']
    names = [o['name'] for o in row['job']['options']]
    if state.get('task') != at.TASK or state['position']['side'] != side:
        raise ValueError('state task/side')
    if sorted(names) != sorted(at.options_for(side)):
        raise ValueError('options for side')
    if type(row['target_index']) is not int or not 0 <= row['target_index'] < len(names):
        raise ValueError('target index')
    if names[row['target_index']] != row['target_action'] or row['split'] != split:
        raise ValueError('target mapping or split')
    t = row['cutoff_epoch']
    if type(t) is not int or t % at.STEP or at.iso(t) != row['cutoff'] or state['data_cutoff'] != row['cutoff']:
        raise ValueError('cutoff')


def load_action_dataset(directory) -> tuple[dict, dict, dict, dict]:
    """Verify identities, schema and chronology before any GPU dispatch.

    Returns (manifest, rows_by_split, rollout, baseline). Rows stay plain dicts (action_eval
    indexes row['job']['options']); errors never echo row content.
    """
    directory = Path(directory)
    for name in INPUT_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'missing file or disallowed symlink: {name}')
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f'input file exceeds 100 MiB: {name}')
    manifest = json.loads((directory / 'manifest.json').read_text())
    valid_id(manifest.get('dataset_id'))
    if manifest['dataset_id'] != content_dataset_id(manifest):
        raise ValueError('content-addressed dataset identity mismatch')
    if manifest.get('task') != at.TASK or not manifest.get('provenance'):
        raise ValueError('dataset task or provenance mismatch')
    if set(manifest['files']) != set(MANIFEST_FILES):
        raise ValueError('manifest file list mismatch')
    for name in MANIFEST_FILES:
        if sha256_file(directory / name) != manifest['files'][name]['sha256']:
            raise ValueError(f'SHA256 mismatch: {name}')
    rows_by_split, previous_last = {}, None
    for split in SPLITS:
        rows = []
        for line_number, line in enumerate((directory / f'{split}.jsonl').read_bytes().splitlines(), 1):
            try:
                row = json.loads(line)
                _check_row(row, split)
                if rows and row['cutoff_epoch'] <= rows[-1]['cutoff_epoch']:
                    raise ValueError('unordered cutoff')
            except Exception:
                raise ValueError(f'invalid row: {split}.jsonl:{line_number}') from None
            rows.append(row)
        written = manifest['splits'][split]['written']
        if not rows or len(rows) != written['total'] or dict(Counter(r['target_action'] for r in rows)) != written['by_action']:
            raise ValueError(f'split counts mismatch: {split}')
        if previous_last is not None and rows[0]['cutoff_epoch'] <= previous_last:
            raise ValueError('splits must be chronological')
        previous_last = rows[-1]['cutoff_epoch']
        rows_by_split[split] = rows
    rollout = _load_rollout(directory, rows_by_split)
    baseline = _load_baseline(directory, manifest)
    return manifest, rows_by_split, rollout, baseline


def _load_rollout(directory: Path, rows_by_split: dict) -> dict:
    rollout = json.loads((directory / 'rollout.json').read_text())
    if set(rollout) != {'candles', 'start', 'end', 'market'}:
        raise ValueError('rollout fields')
    start, end, candles = rollout['start'], rollout['end'], rollout['candles']
    if any(type(x) is not int or x % at.STEP for x in (start, end)) or end <= start:
        raise ValueError('rollout bounds')
    if len(candles) != (end - start) // at.STEP + at.LOOKBACK:
        raise ValueError('rollout candle count')
    first = start - at.LOOKBACK * at.STEP
    for i, c in enumerate(candles):
        values = [c.get(k) for k in ('open', 'high', 'low', 'close', 'volume')]
        if set(c) != {'time', 'open', 'high', 'low', 'close', 'volume'} or c['time'] != first + i * at.STEP:
            raise ValueError('rollout candles must be contiguous 15m opens')
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError('nonfinite rollout candle')
    markets = {r['job']['state']['market'] for rows in rows_by_split.values() for r in rows}
    if markets != {rollout['market']}:
        raise ValueError('rollout market differs from dataset market')
    test = rows_by_split['test']
    if not (start <= test[0]['cutoff_epoch'] and test[-1]['cutoff_epoch'] < end):
        raise ValueError('rollout window must cover the test split')
    return rollout


def _load_baseline(directory: Path, manifest: dict) -> dict:
    baseline = json.loads((directory / 'baseline_results.json').read_text())
    if baseline.get('dataset_id') != manifest['dataset_id'] or baseline.get('task') != at.TASK:
        raise ValueError('baseline results belong to another dataset')
    test = baseline.get('test_metrics')
    if not isinstance(test, dict) or not all(isinstance(test.get(k), (int, float)) for k in ('trade_f1', 'action_macro_f1')):
        raise ValueError('baseline test_metrics missing')
    return baseline


def encode_row(tokenizer, labels, row: dict, max_length: int = MAX_LENGTH) -> dict:
    """Serving formatter (messages_for_job + format_prompt); answer-token-only supervision."""
    from data import encode_example
    from jev_inference.schemas import Job
    return encode_example(tokenizer, labels, {'job': Job.model_validate(row['job']),
                                              'target_index': row['target_index']}, max_length)


def encode_prompt(tokenizer, labels, job_dict: dict, max_length: int = MAX_LENGTH) -> dict:
    """Unlabelled serving prompt; checks EVERY option label boundary like engine.prepare."""
    from jev_inference.prompt import format_prompt, messages_for_job
    from jev_inference.schemas import Job
    job = Job.model_validate(job_dict)
    prompt = format_prompt(tokenizer, messages_for_job(job, labels))
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError('formatted job exceeds serving input limit')
    used = labels[:len(job.options)]
    for label in used:
        if tokenizer.encode(prompt + label.text, add_special_tokens=False) != ids + [label.token_id]:
            raise ValueError('label token boundary changed')
    return {'input_ids': ids, 'candidate_ids': [label.token_id for label in used],
            'names': [o.name for o in job.options]}


def make_batches(lengths: list[int], max_batch: int = 8, max_spread: int = 192) -> list[list[int]]:
    """Length-sorted right-padded batches; a small spread keeps the kept-logit window small."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches, current = [], []
    for i in order:
        if current and (len(current) == max_batch or lengths[i] - lengths[current[0]] > max_spread):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def score_one(model, prompt: dict) -> list[float]:
    """KV-free single forward pass; only the final position's label-token logits."""
    import torch
    ids = torch.tensor([prompt['input_ids']], dtype=torch.long, device='cuda')
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    scores = out.logits[0, -1, prompt['candidate_ids']].float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError('non-finite logits')
    return scores.cpu().tolist()


def score_batch(model, prompts: list[dict], pad_id: int) -> list[list[float]]:
    """Right padding is causal-safe (real tokens never see pads, incl. linear-attention layers).
    Integer logits_to_keep = T - min_len + 1 stays on the same code path as score_one."""
    import torch
    lengths = [len(p['input_ids']) for p in prompts]
    width = max(lengths)
    keep = width - min(lengths) + 1
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device='cuda')
    mask = torch.zeros_like(ids)
    for b, p in enumerate(prompts):
        ids[b, :lengths[b]] = torch.tensor(p['input_ids'], device='cuda')
        mask[b, :lengths[b]] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=keep).logits
    if logits.shape[1] != keep:
        raise RuntimeError('unexpected kept-logit window')
    out = []
    for b, p in enumerate(prompts):
        scores = logits[b, lengths[b] - 1 - (width - keep), p['candidate_ids']].float()
        if not torch.isfinite(scores).all():
            raise FloatingPointError('non-finite logits')
        out.append(scores.cpu().tolist())
    del logits
    return out


def score_all(model, prompts: list[dict], pad_id: int, batched: bool = True) -> list[list[float]]:
    import torch
    results: list = [None] * len(prompts)
    with torch.inference_mode():
        if not batched:
            for i, p in enumerate(prompts):
                results[i] = score_one(model, p)
            return results
        for batch in make_batches([len(p['input_ids']) for p in prompts]):
            for i, scores in zip(batch, score_batch(model, [prompts[i] for i in batch], pad_id)):
                results[i] = scores
    return results


def batched_parity(model, prompts: list[dict], pad_id: int, cases: int = 8, tolerance: float = 0.15) -> dict:
    """Compare batched vs sequential logits on PADDED rows (shorter than their batch maximum)."""
    import time
    import torch
    lengths = [len(p['input_ids']) for p in prompts]
    padded = [(batch, i) for batch in make_batches(lengths) for i in batch if lengths[i] < max(lengths[j] for j in batch)]
    chosen = padded[:: max(1, len(padded) // cases)][:cases] or [(b, b[0]) for b in make_batches(lengths)[:cases]]
    diffs, same = [], True
    tick = time.monotonic()
    with torch.inference_mode():
        sequential = [score_one(model, prompts[i]) for _, i in chosen]
        seq_seconds = (time.monotonic() - tick) / len(chosen)
        for (batch, i), seq in zip(chosen, sequential):
            got = score_batch(model, [prompts[j] for j in batch], pad_id)[batch.index(i)]
            diffs.append(max(abs(a - b) for a, b in zip(got, seq)))
            same &= max(range(len(got)), key=got.__getitem__) == max(range(len(seq)), key=seq.__getitem__)
    # A padding/position bug moves logits by whole units; bf16 batch noise can flip exact ties,
    # so the logit tolerance (same as reload parity) decides and argmax identity is reported.
    return {'cases': len(chosen), 'padded_cases': bool(padded), 'max_logit_difference': max(diffs),
            'argmax_identical': same, 'passed': max(diffs) < tolerance,
            'sequential_seconds_per_row': seq_seconds}


class DeadlineReached(RuntimeError):
    pass


def make_decide_fn(model, tokenizer, labels, margin: float, deadline_epoch: float, sink: list):
    """Rollout policy: serving prompt -> single forward -> action_eval.decide with frozen margin."""
    import time
    from action_eval import decide

    def decide_fn(job: dict) -> str:
        if time.time() >= deadline_epoch:
            raise DeadlineReached('rollout deadline')
        prompt = encode_prompt(tokenizer, labels, job)
        logits = score_one(model, prompt)
        sink.append({'cutoff': job['state']['data_cutoff'], 'names': prompt['names'], 'logits': logits})
        return decide(prompt['names'], logits, margin)
    return decide_fn


def rollout_summary(result: dict) -> dict:
    """Aggregate rollout fields for report.json; per-decision rows (absolute prices) excluded."""
    return {k: v for k, v in result.items() if k != 'decisions'}


def prompts_from_encoded(encoded: list[dict], rows: list[dict], labels) -> list[dict]:
    """Scoring prompt = encoded input minus the target token (boundary already verified)."""
    return [{'input_ids': e['input_ids'][:-1],
             'candidate_ids': [label.token_id for label in labels[:len(r['job']['options'])]],
             'names': [o['name'] for o in r['job']['options']]} for e, r in zip(encoded, rows)]


def train_reserve_seconds(score_seconds: float, rollout_rows: int, sequential_seconds: float) -> float:
    """Time kept after training: LoRA val+test scoring (+25% adapter overhead), sequential
    rollout (+30%), export + reload + parity (240 s) and a 60 s safety margin."""
    return 1.25 * score_seconds + 1.3 * rollout_rows * sequential_seconds + 240 + 60


def train_loop(model, trainable, encoded_train, pad_id, cap_seconds, emit, persist, report):
    """Answer-token-only loss (labels are -100 except the target), bounded steps and wall clock."""
    import random
    import statistics
    import time
    import torch
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=.01)
    losses, durations = [], []
    indices = list(range(len(encoded_train)))
    random.shuffle(indices)
    cursor, started = 0, time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(MAX_STEPS):
        if time.monotonic() - started >= cap_seconds:
            emit('training_time_limit', completed_steps=len(losses))
            break
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            if cursor + BATCH_SIZE > len(indices):
                random.shuffle(indices)
                cursor = 0
            batch = [encoded_train[i] for i in indices[cursor:cursor + BATCH_SIZE]]
            cursor += BATCH_SIZE
            width = max(len(x['input_ids']) for x in batch)
            ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device='cuda')
            attention = torch.zeros_like(ids)
            targets = torch.full_like(ids, -100)
            for i, row in enumerate(batch):
                n = len(row['input_ids'])
                ids[i, :n] = torch.tensor(row['input_ids'], device='cuda')
                attention[i, :n] = 1
                targets[i, :n] = torch.tensor(row['labels'], device='cuda')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(input_ids=ids, attention_mask=attention, labels=targets, use_cache=False)
                loss = result.loss / GRAD_ACCUM
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            loss.backward()
            step_loss += loss.item()
            del result, loss, ids, attention, targets
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError('non-finite gradients')
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(step_loss)
        durations.append(time.monotonic() - tick)
        if step == 0 or (step + 1) % 20 == 0:
            report.update(completed_steps=len(losses))
            persist()
            emit('training', step=step + 1, loss=step_loss, step_seconds=durations[-1])
    report.update(completed_steps=len(losses), losses=[round(x, 5) for x in losses],
                  mean_step_seconds=statistics.mean(durations) if durations else None,
                  training_seconds=time.monotonic() - started, training_cap_seconds=cap_seconds,
                  training_peak_allocated_vram_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                  first_10_mean_loss=statistics.mean(losses[:10]) if losses else None,
                  last_10_mean_loss=statistics.mean(losses[-10:]) if losses else None)
    del optimizer
    return losses


def score_splits(model, prompts: dict, pad_id: int, emit) -> dict:
    """Score every validation+test row; batched only if padded-row parity passes."""
    import time
    from unsloth import FastVisionModel
    FastVisionModel.for_inference(model)
    parity = batched_parity(model, prompts['validation'], pad_id)
    emit('batched_parity', **parity)
    tick = time.monotonic()
    logits = {s: score_all(model, prompts[s], pad_id, batched=parity['passed']) for s in ('validation', 'test')}
    return {'logits': logits, 'parity': parity, 'batched': parity['passed'],
            'seconds': time.monotonic() - tick}


def tuned_metrics(rows: dict, logits: dict) -> dict:
    """Margin tuned on validation only, then frozen and applied once to test."""
    import action_eval
    tuned = action_eval.tune_margin(rows['validation'], logits['validation'])
    margin = tuned['margin']
    test_predictions = action_eval.predict(rows['test'], logits['test'], margin)
    return {'margin': margin, 'margin_rule': tuned['rule'], 'validation_metrics': tuned['val_metrics'],
            'test_metrics': action_eval.metrics(rows['test'], test_predictions),
            'test_metrics_margin_zero': action_eval.metrics(rows['test'], action_eval.predict(rows['test'], logits['test'], 0.0))}


def write_private_scores(path: Path, rows: dict, base: dict, lora: dict | None) -> None:
    """Per-row logits live only in the private artifact volume."""
    with path.open('w') as handle:
        for split in ('validation', 'test'):
            for i, row in enumerate(rows[split]):
                record = {'split': split, 'cutoff': row['cutoff'], 'side': row['side'],
                          'target': row['target_action'], 'names': [o['name'] for o in row['job']['options']],
                          'base_logits': base[split][i], 'lora_logits': lora[split][i] if lora else None}
                handle.write(json.dumps(record) + '\n')


def split_summary(rows: list[dict]) -> dict:
    return {'count': len(rows), 'by_action': dict(sorted(Counter(r['target_action'] for r in rows).items())),
            'by_side': dict(sorted(Counter(r['side'] for r in rows).items()))}


RELOAD_RESERVE_SECONDS = 180  # export already done; fresh base load + 12-row parity


def export_adapter(model, tokenizer, out: Path, report: dict) -> Path:
    adapter = out / 'adapter'
    model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    config = json.loads((adapter / 'adapter_config.json').read_text())
    if config['base_model_name_or_path'] != MODEL_ID:
        raise ValueError('adapter base model mismatch')
    config['revision'] = MODEL_REVISION
    (adapter / 'adapter_config.json').write_text(json.dumps(config, indent=2) + '\n')
    report['adapter_files'] = {p.name: {'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                               for p in sorted(adapter.iterdir()) if p.is_file()}
    report['adapter_sha256'] = report['adapter_files']['adapter_model.safetensors']['sha256']
    return adapter


def run_rollout(ctx: dict, margin: float) -> dict:
    """Closed-loop May rollout from flat with the LoRA; sequential single forwards."""
    import torch
    import action_eval
    rollout, report = ctx['rollout'], ctx['report']
    sink: list = []
    fn = make_decide_fn(ctx['model'], ctx['tokenizer'], ctx['labels'], margin,
                        ctx['deadline_epoch'] - RELOAD_RESERVE_SECONDS, sink)
    try:
        with torch.inference_mode():
            result = action_eval.rollout(rollout['candles'], rollout['start'], rollout['end'], fn, rollout['market'])
    except DeadlineReached:
        report['rollout'] = {'status': 'skipped_deadline', 'decisions_scored': len(sink)}
        return {'opens': 0, 'closes': 0, 'time_in_position_fraction': 0.0}
    (ctx['out'] / 'rollout_decisions.json').write_text(json.dumps(
        [dict(d, names=s['names'], logits=s['logits']) for d, s in zip(result['decisions'], sink)]) + '\n')
    report['rollout'] = {'status': 'completed', 'margin': margin, **rollout_summary(result)}
    return result


def finish(ctx: dict) -> None:
    """Export -> LoRA scoring -> margin -> test -> rollout -> gate -> reload parity."""
    import gc
    import torch
    import action_eval
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from unsloth import FastVisionModel
    from validation import compare_tensors, load_exported_adapter
    report, phase, persist, emit = ctx['report'], ctx['phase'], ctx['persist'], ctx['emit']
    rows, prompts, pad_id = ctx['rows'], ctx['prompts'], ctx['pad_id']
    phase('export')
    adapter = export_adapter(ctx['model'], ctx['tokenizer'], ctx['out'], report)
    FastVisionModel.for_inference(ctx['model'])
    with torch.inference_mode():
        parity_before = [score_one(ctx['model'], p) for p in prompts['validation'][:12]]
    phase('lora_scoring')
    lora = score_splits(ctx['model'], prompts, pad_id, emit)
    report['lora'] = {'batched_parity': lora['parity'], 'scoring_seconds': lora['seconds'],
                      **tuned_metrics(rows, lora['logits'])}
    write_private_scores(ctx['out'] / 'scores.jsonl', rows, ctx['base']['logits'], lora['logits'])
    phase('rollout')
    model_rollout = run_rollout(ctx, report['lora']['margin'])
    phase('gate')
    report['gate'] = action_eval.gate(report['lora']['test_metrics'], ctx['baseline']['test_metrics'], model_rollout)
    report['gate']['baseline_results_model'] = ctx['baseline'].get('model')
    if report['rollout']['status'] != 'completed':
        report['gate'].update(passed=False, reason='rollout not completed before deadline')
    emit('gate', passed=report['gate']['passed'],
         criteria={k: v['passed'] for k, v in report['gate']['criteria'].items()})
    phase('reload')
    report['reload_status'] = 'pending'
    saved = load_file(str(adapter / 'adapter_model.safetensors'))
    in_memory = get_peft_model_state_dict(ctx['model'])
    compare_tensors(in_memory, saved)
    report.update(adapter_tensor_dtypes=sorted({str(t.dtype) for t in saved.values()}),
                  memory_tensor_sha256=tensor_hash(in_memory), saved_tensor_sha256=tensor_hash(saved),
                  saved_tensor_bytes_identical=True)
    persist()
    del in_memory
    ctx.pop('model')
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_exported_adapter(adapter, MAX_LENGTH)
    compare_tensors(saved, get_peft_model_state_dict(reloaded))
    FastVisionModel.for_inference(reloaded)
    with torch.inference_mode():
        parity_after = [score_one(reloaded, p) for p in prompts['validation'][:12]]
    diff = max(abs(x - y) for a, b in zip(parity_before, parity_after) for x, y in zip(a, b))
    same = [action_eval.decide(p['names'], a, report['lora']['margin']) for p, a in zip(prompts['validation'], parity_before)] == \
           [action_eval.decide(p['names'], b, report['lora']['margin']) for p, b in zip(prompts['validation'], parity_after)]
    report.update(reloaded_tensor_bytes_identical=True, reload_cases=12,
                  reload_max_logit_difference=diff, reload_decisions_identical=same)
    if diff >= .15 or not same:
        raise RuntimeError('reload parity failed')
    report.update(reload_status='passed', status='passed')
    phase('complete')
    emit('complete', status='passed', gate_passed=report['gate']['passed'])


def main(run_id: str, dataset_id: str, deadline_epoch: float) -> None:
    import importlib.metadata
    import random
    import time
    start = time.monotonic()
    out = Path('/artifacts') / valid_id(run_id)
    out.mkdir(parents=True, exist_ok=False)
    report = {'run_id': run_id, 'dataset_id': valid_id(dataset_id), 'task': at.TASK, 'model': MODEL_ID,
              'revision': MODEL_REVISION, 'status': 'started', 'phase': 'setup', 'reload_status': 'not_started',
              'production_promoted': False, 'deadline_epoch': deadline_epoch, 'timings': {}}

    def persist():
        report['child_process_elapsed_seconds'] = time.monotonic() - start
        report['gpu_cost_estimate_usd'] = report['child_process_elapsed_seconds'] * RATE_USD_PER_SECOND
        report['planned_cost_ceiling_usd'] = planned_cost_usd()
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def emit(event, **fields):
        record = {'event': event, 'elapsed_seconds': round(time.monotonic() - start, 3), **fields}
        print(json.dumps(record), flush=True)
        with (out / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')

    def phase(name):
        report['timings'][report['phase']] = round(time.monotonic() - phase.tick, 2)
        report['phase'], phase.tick = name, time.monotonic()
        persist()
        emit('phase', name=name)
    phase.tick = start

    persist()
    try:
        from unsloth import FastVisionModel  # Must precede transformers/peft.
        import torch
        from transformers import AutoTokenizer
        from jev_inference.labels import select_labels
        from jev_inference.prompt import SYSTEM_PROMPT, format_prompt
        directory = Path('/inputs') / dataset_id
        manifest, rows, rollout, baseline = load_action_dataset(directory)
        if manifest['dataset_id'] != dataset_id:
            raise ValueError('dataset directory identity mismatch')
        report.update(dataset_files=manifest['files'], baseline_results_sha256=sha256_file(directory / 'baseline_results.json'),
                      splits={s: split_summary(rows[s]) for s in SPLITS},
                      training_method=f'BF16 LoRA rank{LORA_RANK} (FP32 adapter masters); answer-token-only loss',
                      requested_steps=MAX_STEPS, batch_size=BATCH_SIZE, gradient_accumulation=GRAD_ACCUM,
                      cost_note='GPU estimate = child elapsed x $0.0013/s; excludes startup/teardown (300 s budgeted), '
                                'CPU/RAM/storage and image build. Not an invoice.',
                      limitations=['Behavior imitation of one trader over three months; not a profitability claim.',
                                   'Margin tuned on validation only; test and rollout scored once.',
                                   'Reload parity uses the training runtime; production server parity is separate.'])
        phase('load_model')
        torch.manual_seed(3407)
        random.seed(3407)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        model, _ = FastVisionModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, load_in_4bit=False, dtype=torch.bfloat16,
            use_gradient_checkpointing='unsloth', max_seq_length=MAX_LENGTH,
            trust_remote_code=False, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        probe = format_prompt(tokenizer, [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': '{}'}])
        labels = select_labels(tokenizer, probe)  # serving: full label set, sliced per job
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        phase('encode')
        encoded = {s: [encode_row(tokenizer, labels, r) for r in rows[s]] for s in SPLITS}
        prompts = {s: prompts_from_encoded(encoded[s], rows[s], labels) for s in ('validation', 'test')}
        for s in ('validation', 'test'):  # full engine-style boundary check on a sample
            for r, p in zip(rows[s][:32], prompts[s][:32]):
                if encode_prompt(tokenizer, labels, r['job']) != p:
                    raise ValueError('scoring prompt differs from serving prompt')
        lengths = [len(e['input_ids']) for s in SPLITS for e in encoded[s]]
        report.update(input_tokens={'min': min(lengths), 'max': max(lengths), 'mean': sum(lengths) / len(lengths)},
                      labels=[x.text for x in labels[:4]])
        model = FastVisionModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=LORA_RANK, lora_alpha=LORA_RANK, lora_dropout=0, bias='none', random_state=3407)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or not all('lora_' in n and 'visual' not in n for n, _ in trainable):
            raise RuntimeError('unexpected trainable parameters')
        for _, parameter in trainable:  # FP32 masters preserve save/reload precision
            parameter.data = parameter.data.float()
        before = {name: p.detach().cpu().clone() for name, p in trainable}
        report.update(gpu=torch.cuda.get_device_name(0), trainable_parameters=sum(p.numel() for _, p in trainable),
                      packages={p: importlib.metadata.version(p) for p in ['unsloth', 'unsloth_zoo', 'torch', 'transformers', 'peft']})
        phase('base_scoring')  # zero-initialised LoRA B == base model
        base = score_splits(model, prompts, pad_id, emit)
        report['base'] = {'batched_parity': base['parity'], 'scoring_seconds': base['seconds'],
                          **tuned_metrics(rows, base['logits'])}
        write_private_scores(out / 'scores.jsonl', rows, base['logits'], None)
        phase('training')
        reserve = train_reserve_seconds(base['seconds'], len(rollout['candles']) - at.LOOKBACK,
                                        base['parity']['sequential_seconds_per_row'])
        cap = min(TRAIN_WALL_SECONDS, deadline_epoch - time.time() - reserve)
        report.update(train_reserve_seconds=reserve)
        FastVisionModel.for_training(model)
        losses = train_loop(model, trainable, encoded['train'], pad_id, cap, emit, persist, report)
        changed = sum(not torch.equal(before[n], p.detach().cpu()) for n, p in trainable)
        report['adapter_tensors_changed'] = changed
        if len(losses) < MIN_STEPS or not changed:
            raise RuntimeError('training produced too few steps or unchanged adapter')
        del before, trainable
        ctx = dict(model=model, tokenizer=tokenizer, labels=labels, rows=rows, prompts=prompts,
                   rollout=rollout, baseline=baseline, base=base, pad_id=pad_id, deadline_epoch=deadline_epoch,
                   out=out, report=report, phase=phase, persist=persist, emit=emit)
        del model, encoded
        finish(ctx)
    except BaseException as error:
        # Never stringify errors: they could carry example content or tensor values.
        report.update(status='failed', error_type=type(error).__name__)
        if report['reload_status'] == 'pending':
            report['reload_status'] = 'failed'
        persist()
        emit('failed', error_type=type(error).__name__, phase=report['phase'])
        raise


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
