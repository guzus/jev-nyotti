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
from run_real import MODEL_ID, MODEL_REVISION, reject_identifier_fields, valid_id  # noqa: E402

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


def split_summary(rows: list[dict]) -> dict:
    return {'count': len(rows), 'by_action': dict(sorted(Counter(r['target_action'] for r in rows).items())),
            'by_side': dict(sorted(Counter(r['side'] for r in rows).items()))}
