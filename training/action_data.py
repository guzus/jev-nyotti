"""ACTION_V1 dataset builder (see ACTION_V1.md; prompt contract in inference/jev_inference/action_task.py).

Market input: official BitMEX XBTUSD 1m buckets (timestamp = bucket END, open = previous
close) aggregated into 15m candles whose `time` is the candle OPEN. A cutoff's 96 candles
use only buckets ending at or before the cutoff. Teacher state uses fills strictly before
the cutoff; fills at the cutoff belong to the target window [cutoff, cutoff+15m).

Rows carry only the scoring job and the label. No execution identifiers, quantities,
fill prices or balances are written. Jobs are built only through action_task.build_job.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'inference'))
from jev_inference import action_task as at  # noqa: E402

from action_ledger import Inventory, classify_delta  # noqa: E402
from real_data import iso as iso_ms, read_positions, sha256, timestamp  # noqa: E402

STEP = at.STEP
MINUTE = 60
BUCKETS = STEP // MINUTE
EPISODE_GAP_SECONDS = 60
MARKET = at.market_label('BTC')  # neutral; identical in live serving and replay (ACTION_V2.md)
SEED = 3407
HOLDS_PER_ACTION = 2
SPLITS = (('train', '2018-03-02T00:00:00Z', '2018-04-01T00:00:00Z'),
          ('validation', '2018-04-02T00:00:00Z', '2018-05-01T00:00:00Z'),
          ('test', '2018-05-02T00:00:00Z', '2018-06-01T00:00:00Z'))
ROLLOUT = ('2018-05-02T00:00:00Z', '2018-06-01T00:00:00Z')
# Frozen pre-freeze March counts from ACTION_V1.md (all-window, before ambiguity exclusion).
FROZEN_MARCH = {'flat_hold': 1623, 'position_hold': 465, 'open': 140, 'add_reduce_close': 335}
DEFAULT_SOURCE = Path('/Users/guzus/Downloads/aoa_public_2021-12-31_with_letter')
DEFAULT_SNAPSHOT = Path('/Users/guzus/guzus/jev-nyotti-market-action/.runtime/action-market-1m/prefixes/'
                        '2018-06-01T000000Z/bitmex-xbtusd-1m.jsonl')
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / '.runtime' / 'action-v1'
RULES = [
    'Candles: 15 consecutive official BitMEX XBTUSD 1m buckets ending at open+60s..open+900s; open=first bucket open '
    '(BitMEX previous close, not clipped to high/low), high/low=max/min, close=last close, volume=sum; any gap fails.',
    'Market input: 96 closed 15m candles whose last candle closes exactly at the cutoff (bucket end <= cutoff).',
    'Teacher state: action_ledger.Inventory over XBTUSD Trade fills strictly before the cutoff (harmonic entry, '
    'opened_at, last_trade_at); zero initial position checked against every funding quantity in all four files.',
    'Target window [cutoff, cutoff+15m): fills at the cutoff belong to the target. First episode = first fill plus '
    'following same-side fills in the window while each gap <= 60s; an opposite fill or a gap > 60s ends it.',
    "Class = episode's signed quantity applied to the position before the cutoff: open_long/open_short from flat; "
    'add/reduce/close in position (a reversal is close). hold only when the window has no fill.',
    'Ambiguous: any timestamp inside the target window with both Buy and Sell fills; the window is excluded everywhere.',
    'Coverage: cutoffs whose target window ends at or before the first observed execution are not labelled '
    '(unobserved coverage is not deliberate inactivity).',
    'Splits by cutoff with target window inside the month; the first 24h of each month is purged so no 24h lookback '
    'crosses a split. Train = every action window + fixed-seed hold subsample (2 holds per action within each side); '
    'validation/test = every non-ambiguous window.',
    'Train option order is shuffled with a fixed seed; validation/test use canonical order.',
]


@dataclass(frozen=True)
class Fill:
    time: float
    qty: int  # signed contracts, Buy > 0
    price: float


def epoch(value: str) -> int:
    return int(timestamp(value))


def split_for(cutoff: int, splits=None) -> str | None:
    if cutoff % STEP:
        raise ValueError('cutoff must be a 15-minute boundary')
    for name, start, end in (splits or SPLITS):
        if epoch(start) <= cutoff and cutoff + STEP <= epoch(end):
            return name
    return None


# ----------------------------------------------------------------------------- market
def load_buckets(path: Path, metadata_path: Path | None = None) -> tuple[list[dict], dict]:
    digest = sha256(path)
    evidence = {'name': path.name, 'sha256': digest, 'bytes': path.stat().st_size}
    if metadata_path is not None:
        meta = json.loads(metadata_path.read_text())
        if meta.get('sha256') != digest:
            raise ValueError('1m snapshot hash does not match its metadata')
        if meta.get('binSize') != '1m' or meta.get('symbol') != 'XBTUSD' or meta.get('gaps'):
            raise ValueError('unexpected snapshot metadata')
        evidence.update(metadata_sha256=sha256(metadata_path), range_inclusive=meta.get('range_inclusive'),
                        metadata_count=meta.get('count'), source=meta.get('endpoint'))
    rows = []
    for line in path.read_text().splitlines():
        raw = json.loads(line)
        if raw['symbol'] != 'XBTUSD':
            raise ValueError('wrong market symbol')
        end = timestamp(raw['timestamp'])
        if end % MINUTE:
            raise ValueError('bucket end is not a minute boundary')
        row = {'end': int(end), **{k: float(raw[k]) for k in ('open', 'high', 'low', 'close', 'volume')}}
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError('nonfinite bucket')
        if row['low'] <= 0 or row['open'] <= 0 or not row['low'] <= row['close'] <= row['high'] or row['volume'] < 0:
            raise ValueError('invalid bucket')
        rows.append(row)
    rows.sort(key=lambda r: r['end'])
    if any(b['end'] - a['end'] != MINUTE for a, b in zip(rows, rows[1:])):
        raise ValueError('duplicate or missing 1m bucket')
    if metadata_path is not None and len(rows) != evidence['metadata_count']:
        raise ValueError('bucket count does not match metadata')
    evidence['count'] = len(rows)
    return rows, evidence


def aggregate_15m(buckets: list[dict]) -> list[dict]:
    """15m candles with `time` = candle OPEN. Only complete 15-bucket groups; any interior gap fails."""
    if any(b['end'] - a['end'] != MINUTE for a, b in zip(buckets, buckets[1:])):
        raise ValueError('duplicate or missing 1m bucket')
    by_end = {b['end']: b for b in buckets}
    if not buckets:
        return []
    first, last = buckets[0]['end'], buckets[-1]['end']
    t = -(-(first - MINUTE) // STEP) * STEP  # first open whose first bucket (ending t+60) exists
    candles = []
    while t + STEP <= last:
        group = [by_end.get(t + MINUTE * (k + 1)) for k in range(BUCKETS)]
        if any(g is None for g in group):
            raise ValueError(f'missing 1m bucket inside 15m candle {at.iso(t)}')
        candles.append({'time': t, 'open': group[0]['open'], 'high': max(g['high'] for g in group),
                        'low': min(g['low'] for g in group), 'close': group[-1]['close'],
                        'volume': sum(g['volume'] for g in group)})
        t += STEP
    return candles


def history(candles: list[dict], index: dict[int, int], cutoff: int) -> list[dict]:
    """96 closed candles ending exactly at cutoff (last candle open = cutoff - 15m)."""
    last = index.get(cutoff - STEP)
    if last is None or last < at.LOOKBACK - 1:
        raise ValueError(f'insufficient candles for cutoff {at.iso(cutoff)}')
    out = candles[last - at.LOOKBACK + 1:last + 1]
    if out[0]['time'] != cutoff - at.LOOKBACK * STEP:
        raise ValueError('candle gap')
    return out


# ----------------------------------------------------------------------------- fills
def read_fills(paths: list[Path], symbol: str = 'XBTUSD') -> tuple[list[Fill], Counter]:
    """XBTUSD Trade fills (signed contracts, price), chronological and stable within a timestamp."""
    fills, counts = [], Counter()
    for path in sorted(paths):
        with path.open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['symbol'] != symbol:
                    continue
                counts[row['exectype']] += 1
                if row['exectype'] != 'Trade':
                    continue
                qty, price = float(row['lastqty']), float(row['lastpx'])
                if not math.isfinite(qty) or qty <= 0 or not qty.is_integer() or row['side'] not in ('Buy', 'Sell'):
                    raise ValueError('invalid trade fill')
                if not math.isfinite(price) or price <= 0:
                    raise ValueError('invalid fill price')
                fills.append(Fill(timestamp(row['transacttime']), int(qty) if row['side'] == 'Buy' else -int(qty), price))
    fills.sort(key=lambda f: f.time)  # stable: file order kept within one timestamp
    return fills, counts


def check_position_path(fills: list[Fill], times: list[float], positions: list[int], end: float) -> dict:
    """Trade-only net position must equal the funding-reconciled path at every fill time before `end`."""
    position, checked = 0, 0
    fill_times = []
    for i, f in enumerate(fills):
        if f.time >= end:
            break
        position += f.qty
        if i + 1 < len(fills) and fills[i + 1].time == f.time:
            continue
        fill_times.append(f.time)
        k = bisect_right(times, f.time) - 1
        if k < 0 or times[k] != f.time or positions[k] != position:
            raise ValueError(f'trade-only position path diverges from the reconciled path at {iso_ms(f.time)}')
        checked += 1
    reconciled_in_range = [t for t in times if t < end]
    if reconciled_in_range != fill_times:
        raise ValueError('non-trade position change inside the used range')
    return {'distinct_fill_times_checked': checked, 'range_end': iso_ms(end)}


def two_sided_times(fills: list[Fill]) -> set[float]:
    sides: dict[float, set[int]] = {}
    for f in fills:
        sides.setdefault(f.time, set()).add(1 if f.qty > 0 else -1)
    return {t for t, s in sides.items() if len(s) == 2}


def action_class(prior: int, delta: int) -> str:
    kind = classify_delta(prior, delta)
    if kind in ('hold', 'open_long', 'open_short'):
        return kind
    return {'add': 'add', 'reduce': 'reduce', 'close': 'close', 'reverse': 'close'}[kind.split('_')[0]]


def label_window(fills: list[Fill], times: list[float], cutoff: int, prior: int, ambiguous_times: set[float]) -> dict:
    """First same-direction episode in [cutoff, cutoff+15m) applied to the position before cutoff."""
    lo, hi = bisect_left(times, cutoff), bisect_left(times, cutoff + STEP)
    window = fills[lo:hi]
    if not window:
        return {'label': 'hold', 'ambiguous': False, 'truncated': False, 'split_run': False, 'episode_fills': 0}
    if any(f.time in ambiguous_times for f in window):
        return {'label': None, 'ambiguous': True, 'truncated': False, 'split_run': False, 'episode_fills': 0}
    sign = 1 if window[0].qty > 0 else -1
    total, prev, n = window[0].qty, window[0], 1
    for f in window[1:]:
        if (1 if f.qty > 0 else -1) != sign or f.time - prev.time > EPISODE_GAP_SECONDS:
            break
        total, prev, n = total + f.qty, f, n + 1
    truncated = (n == len(window) and hi < len(fills) and (1 if fills[hi].qty > 0 else -1) == sign
                 and fills[hi].time - prev.time <= EPISODE_GAP_SECONDS)
    split_run = (lo > 0 and (1 if fills[lo - 1].qty > 0 else -1) == sign
                 and window[0].time - fills[lo - 1].time <= EPISODE_GAP_SECONDS)
    return {'label': action_class(prior, total), 'ambiguous': False, 'truncated': truncated,
            'split_run': split_run, 'episode_fills': n}


def position_dict(inv: Inventory) -> dict:
    side = 'long' if inv.quantity > 0 else 'short' if inv.quantity < 0 else 'flat'
    return {'side': side, 'entry_price': inv.entry_price if side != 'flat' else None,
            'opened_at': inv.opened_at if side != 'flat' else None, 'last_trade_at': inv.last_trade_at}


def march_bucket(side: str, label: str) -> str:
    if label == 'hold':
        return 'flat_hold' if side == 'flat' else 'position_hold'
    return 'open' if label.startswith('open') else 'add_reduce_close'


# ----------------------------------------------------------------------------- windows
def label_windows(fills: list[Fill], candles: list[dict], start: int, end: int) -> list[dict]:
    """Every 15m cutoff in [start, end) after coverage begins, with teacher state strictly before cutoff."""
    times = [f.time for f in fills]
    ambiguous_times = two_sided_times(fills)
    index = {c['time']: i for i, c in enumerate(candles)}
    inv, k, out = Inventory(), 0, []
    first_exec = fills[0].time if fills else math.inf
    for cutoff in range(start, end, STEP):
        while k < len(fills) and fills[k].time < cutoff:
            f = fills[k]
            inv.apply(f.qty, f.price, f.time)
            k += 1
        if cutoff + STEP <= first_exec:
            continue
        hist = history(candles, index, cutoff)
        inv.snapshot(hist[-1]['close'], cutoff)  # guard: raises if any state fill is at/after cutoff
        label = label_window(fills, times, cutoff, inv.quantity, ambiguous_times)
        out.append({'cutoff': cutoff, 'position': position_dict(inv), **label})
    return out


def make_row(window: dict, candles: list[dict], index: dict[int, int], split: str, rng: random.Random | None) -> dict:
    position, target = window['position'], window['label']
    names = list(at.options_for(position['side']))
    if rng is not None:
        rng.shuffle(names)
    job = at.build_job(candles=history(candles, index, window['cutoff']), cutoff=window['cutoff'],
                       position=position, market=MARKET, order=names)
    if target not in names:
        raise ValueError('label is not an option for the state')
    return {'job': job, 'target_index': names.index(target), 'target_action': target,
            'cutoff': at.iso(window['cutoff']), 'cutoff_epoch': window['cutoff'], 'split': split,
            'side': position['side']}


def select_train(windows: list[dict], rng: random.Random) -> list[dict]:
    chosen = [w for w in windows if w['label'] != 'hold']
    for side in ('flat', 'long', 'short'):
        holds = [w for w in windows if w['label'] == 'hold' and w['position']['side'] == side]
        actions = sum(1 for w in chosen if w['position']['side'] == side)
        chosen += rng.sample(holds, min(len(holds), HOLDS_PER_ACTION * actions))
    return sorted(chosen, key=lambda w: w['cutoff'])


def counts(rows: list[dict]) -> dict:
    return {'total': len(rows), 'by_action': dict(sorted(Counter(r['target_action'] for r in rows).items())),
            'by_side': dict(sorted(Counter(r['side'] for r in rows).items())),
            'by_side_action': dict(sorted(Counter(f"{r['side']}:{r['target_action']}" for r in rows).items()))}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')


def build(source: Path, snapshot: Path, output: Path, fresh: Path | None = None, splits=None) -> dict:
    splits = [tuple(x) for x in (splits or SPLITS)]
    if output.exists():
        raise ValueError(f'{output} already exists; refusing to overwrite')
    paths = sorted(source.glob('aoa-execution-*.csv'))
    if len(paths) != 4:
        raise ValueError('expected the four supplied execution CSV files')
    buckets, market_evidence = load_buckets(snapshot, snapshot.with_name(snapshot.name.replace('.jsonl', '.metadata.json')))
    candles = aggregate_15m(buckets)
    index = {c['time']: i for i, c in enumerate(candles)}
    times, positions, reconciliation = read_positions(paths)
    fills, event_counts = read_fills(paths)
    start, end = epoch(splits[0][1]), epoch(splits[-1][2])
    reconciliation['trade_path_check'] = check_position_path(fills, times, positions, end)
    reconciliation['xbtusd_event_counts'] = dict(event_counts)

    # March all-window check against the frozen pre-freeze counts (pre-ambiguity, pre-purge).
    march = label_windows(fills, candles, epoch('2018-03-01T00:00:00Z') + at.LOOKBACK * STEP, epoch('2018-04-01T00:00:00Z'))
    march_counts = Counter(march_bucket(w['position']['side'], w['label']) for w in march if not w['ambiguous'])
    march_check = {'computed_non_ambiguous': dict(march_counts), 'ambiguous': sum(w['ambiguous'] for w in march),
                   'frozen': FROZEN_MARCH, 'first_labelled_cutoff': at.iso(march[0]['cutoff']) if march else None}

    windows = label_windows(fills, candles, start, end)
    rng = random.Random(SEED)
    groups: dict[str, list[dict]] = {name: [] for name, _, _ in splits}
    diagnostics = Counter()
    for w in windows:
        split = split_for(w['cutoff'], splits)
        if split is None:
            continue
        diagnostics[f'{split}:eligible'] += 1
        if w['ambiguous']:
            diagnostics[f'{split}:ambiguous_excluded'] += 1
            continue
        diagnostics[f'{split}:truncated_at_window_end'] += w['truncated']
        diagnostics[f'{split}:first_episode_continues_pre_cutoff_run'] += w['split_run']
        groups[split].append(w)

    tmp = output.with_name(output.name + f'.tmp-{os.getpid()}')
    tmp.mkdir(parents=True)
    manifest: dict = {'task': at.TASK, 'market': MARKET, 'contract': 'training/ACTION_V1.md',
                      'provenance': 'user_supplied_aoa_execution_export',
                      'source_files': [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in paths],
                      'market_snapshot': market_evidence, 'candles_15m': len(candles), 'rules': RULES,
                      'splits_used': splits, 'seed': SEED, 'holds_per_action': HOLDS_PER_ACTION, 'reconciliation': reconciliation,
                      'march_frozen_count_check': march_check, 'splits': {}, 'files': {}}
    try:
        for name, first, last in splits:
            natural = groups[name]
            chosen = select_train(natural, rng) if name == 'train' else natural
            rows = [make_row(w, candles, index, name, rng if name == 'train' else None) for w in chosen]
            write_jsonl(tmp / f'{name}.jsonl', rows)
            manifest['splits'][name] = {
                'cutoff_range': [first, last], 'first_cutoff': rows[0]['cutoff'] if rows else None,
                'last_cutoff': rows[-1]['cutoff'] if rows else None,
                'eligible_windows': diagnostics[f'{name}:eligible'],
                'excluded_ambiguous': diagnostics[f'{name}:ambiguous_excluded'],
                'truncated_at_window_end': diagnostics[f'{name}:truncated_at_window_end'],
                'first_episode_continues_pre_cutoff_run': diagnostics[f'{name}:first_episode_continues_pre_cutoff_run'],
                'natural': counts([{'target_action': w['label'], 'side': w['position']['side']} for w in natural]),
                'written': counts(rows)}
        if fresh is None:
            r_start, r_end = epoch(ROLLOUT[0]), epoch(ROLLOUT[1])
            roll = [c for c in candles if r_start - at.LOOKBACK * STEP <= c['time'] < r_end]
            roll_source = 'BitMEX XBTUSD 15m aggregated from the official 1m snapshot (May 2018)'
        else:
            # Confirmatory closed-loop window never viewed by any model (ACTION_V2.md): criterion 4
            # needs only candles, not teacher labels.
            src = json.loads(fresh.read_text())
            (series,) = src['series']
            if series['market'] != MARKET:
                raise ValueError('fresh rollout market label differs from training market label')
            r_start, r_end = epoch(src['from']), epoch(src['to'])
            roll = [{k: c[k] for k in ('time', 'open', 'high', 'low', 'close', 'volume')} for c in series['candles']
                    if r_start - at.LOOKBACK * STEP <= c['time'] < r_end]
            roll_source = f"{src['source']} {series['symbol']} ({fresh.name}, sha256 {sha256(fresh)})"
        if len(roll) != (r_end - r_start) // STEP + at.LOOKBACK:
            raise ValueError('rollout candles incomplete')
        (tmp / 'rollout.json').write_text(json.dumps({'candles': roll, 'start': r_start, 'end': r_end, 'market': MARKET},
                                                     separators=(',', ':'), allow_nan=False) + '\n')
        manifest['rollout_source'] = roll_source
        for name in ('train.jsonl', 'validation.jsonl', 'test.jsonl', 'rollout.json'):
            manifest['files'][name] = {'sha256': sha256(tmp / name), 'bytes': (tmp / name).stat().st_size}
        manifest['excluded_ambiguous_total'] = sum(s['excluded_ambiguous'] for s in manifest['splits'].values())
        manifest['dataset_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
        (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        if output.exists():
            raise ValueError(f'{output} appeared during build; refusing to overwrite')
        output.parent.mkdir(parents=True, exist_ok=True)
        os.rename(tmp, output)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--splits', type=json.loads, help='JSON [[name, start, end], ...] (default: ACTION_V1 months)')
    parser.add_argument('--fresh-rollout', type=Path, help='action-replay-input.json with one BTCUSD series')
    args = parser.parse_args()
    manifest = build(args.source, args.snapshot, args.output, args.fresh_rollout, args.splits)
    print(json.dumps({k: manifest[k] for k in ('dataset_id', 'excluded_ambiguous_total', 'march_frozen_count_check')}
                     | {'splits': {k: {'written': v['written'], 'natural': v['natural']['by_action'],
                                       'excluded_ambiguous': v['excluded_ambiguous']} for k, v in manifest['splits'].items()}},
                     indent=2))


if __name__ == '__main__':
    main()
