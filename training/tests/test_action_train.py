"""ACTION_V1 trainer: dataset validation, serving-format encoding and budget arithmetic (no GPU)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from jev_inference import action_task as at
from jev_inference.labels import Label
import run_action as ra

T0 = 1520000000 - 1520000000 % at.STEP  # 15m boundary
MARKET = 'TEST XBTUSD'


def candles(first_open, count):
    out, price = [], 100.0
    for i in range(count):
        close = price * (1.001 if i % 3 else 0.999)
        out.append({'time': first_open + i * at.STEP, 'open': price, 'high': max(price, close) * 1.001,
                    'low': min(price, close) * 0.999, 'close': close, 'volume': 10.0 + i % 7})
        price = close
    return out


SERIES = candles(T0, 400)


def make_row(split, k, side='flat', target=None):
    cutoff = T0 + (at.LOOKBACK + k) * at.STEP
    hist = [c for c in SERIES if cutoff - at.LOOKBACK * at.STEP <= c['time'] < cutoff]
    position = at.flat_position() if side == 'flat' else dict(
        side=side, units=1.0, entry_price=100.0, opened_at=cutoff - 3600, last_trade_at=cutoff - 3600)
    names = list(reversed(at.options_for(side)))
    target = target or names[1]
    job = at.build_job(candles=hist, cutoff=cutoff, position=position, market=MARKET, order=names)
    return {'job': job, 'target_index': names.index(target), 'target_action': target, 'cutoff': at.iso(cutoff),
            'cutoff_epoch': cutoff, 'split': split, 'side': side}


class Fixture:
    def __init__(self, directory: Path):
        self.dir = directory
        self.rows = {'train': [make_row('train', 0), make_row('train', 1, 'long')],
                     'validation': [make_row('validation', 100, 'short', 'hold')],
                     'test': [make_row('test', 200), make_row('test', 201, 'long', 'close')]}
        start = self.rows['test'][0]['cutoff_epoch']
        self.rollout = {'candles': [c for c in SERIES if start - at.LOOKBACK * at.STEP <= c['time'] < start + 4 * at.STEP],
                        'start': start, 'end': start + 4 * at.STEP, 'market': MARKET}
        self.baseline_extra = {}

    def write(self, mutate_manifest=None):
        manifest = {'task': at.TASK, 'provenance': 'test_fixture', 'splits': {}, 'files': {}}
        for split, rows in self.rows.items():
            (self.dir / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            manifest['splits'][split] = {'written': ra.split_summary(rows) | {'total': len(rows)}}
        (self.dir / 'rollout.json').write_text(json.dumps(self.rollout))
        for name in ra.MANIFEST_FILES:
            manifest['files'][name] = {'sha256': hashlib.sha256((self.dir / name).read_bytes()).hexdigest()}
        if mutate_manifest:
            mutate_manifest(manifest)
        manifest['dataset_id'] = ra.content_dataset_id(manifest)
        (self.dir / 'manifest.json').write_text(json.dumps(manifest))
        baseline = {'task': at.TASK, 'dataset_id': manifest['dataset_id'],
                    'test_metrics': {'trade_f1': 0.4, 'action_macro_f1': 0.2}} | self.baseline_extra
        (self.dir / 'baseline_results.json').write_text(json.dumps(baseline))
        return manifest


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def rejects(self, pattern):
        self.fx.write()
        with self.assertRaisesRegex(ValueError, pattern):
            ra.load_action_dataset(self.fx.dir)

    def test_valid_fixture_loads_as_plain_dicts(self):
        manifest = self.fx.write()
        loaded, rows, rollout, baseline = ra.load_action_dataset(self.fx.dir)
        self.assertEqual(loaded['dataset_id'], manifest['dataset_id'])
        self.assertIsInstance(rows['train'][0]['job'], dict)
        self.assertEqual(rollout['market'], MARKET)
        self.assertEqual(baseline['test_metrics']['trade_f1'], 0.4)

    def test_modified_file_rejected_by_hash(self):
        self.fx.write()
        with (self.fx.dir / 'test.jsonl').open('a') as handle:
            handle.write('\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            ra.load_action_dataset(self.fx.dir)

    def test_tampered_manifest_identity_rejected(self):
        self.fx.write()
        manifest = json.loads((self.fx.dir / 'manifest.json').read_text())
        manifest['provenance'] = 'other'
        (self.fx.dir / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'identity'):
            ra.load_action_dataset(self.fx.dir)

    def test_option_names_invalid_for_side_rejected(self):
        row = self.fx.rows['train'][1]
        row['side'] = 'flat'
        row['job']['state']['position']['side'] = 'flat'  # flat state with position options
        self.rejects(r'invalid row: train.jsonl:2')

    def test_unknown_option_name_rejected(self):
        self.fx.rows['test'][0]['job']['options'][0]['name'] = 'buy'
        self.rejects(r'invalid row: test.jsonl:1')

    def test_identifier_field_rejected_without_echo(self):
        self.fx.rows['validation'][0]['job']['state']['account'] = 'SECRET-123'
        self.fx.write()
        with self.assertRaises(ValueError) as caught:
            ra.load_action_dataset(self.fx.dir)
        self.assertNotIn('SECRET', str(caught.exception))

    def test_extra_row_field_rejected(self):
        self.fx.rows['train'][0]['orderID'] = 'x'
        self.rejects('invalid row: train.jsonl:1')

    def test_target_mapping_mismatch_rejected(self):
        self.fx.rows['train'][0]['target_action'] = 'hold' if self.fx.rows['train'][0]['target_action'] != 'hold' else 'open_long'
        self.rejects('invalid row')

    def test_cutoff_string_must_match_epoch(self):
        self.fx.rows['test'][1]['cutoff_epoch'] += at.STEP
        self.rejects('invalid row: test.jsonl:2')

    def test_split_order_and_chronology_enforced(self):
        self.fx.rows['validation'], self.fx.rows['test'] = self.fx.rows['validation'] * 1, self.fx.rows['test']
        self.fx.rows['train'] = list(reversed(self.fx.rows['train']))
        self.rejects('unordered|invalid row')

    def test_split_count_mismatch_rejected(self):
        self.fx.write(lambda m: m['splits']['train']['written'].update(total=99))
        with self.assertRaisesRegex(ValueError, 'counts'):
            ra.load_action_dataset(self.fx.dir)

    def test_baseline_for_other_dataset_rejected(self):
        self.fx.baseline_extra = {'dataset_id': 'other'}
        self.rejects('another dataset')

    def test_rollout_gap_rejected(self):
        del self.fx.rollout['candles'][5]
        self.rejects('rollout')

    def test_symlink_rejected(self):
        self.fx.write()
        target = self.fx.dir / 'baseline_results.json'
        moved = self.fx.dir / 'elsewhere.json'
        target.rename(moved)
        target.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            ra.load_action_dataset(self.fx.dir)


class Tokenizer:
    all_special_ids = []

    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(x['content'] for x in messages) + '\nassistant:'

    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode('utf-8')


class EncodingTests(unittest.TestCase):
    labels = [Label(x, ord(x)) for x in 'ABCD']

    def check(self, row, count):
        enc = ra.encode_row(Tokenizer(), self.labels, row)
        self.assertEqual(enc['labels'][:-1], [-100] * (len(enc['input_ids']) - 1))
        self.assertEqual(enc['labels'][-1], self.labels[row['target_index']].token_id)
        prompt = ra.encode_prompt(Tokenizer(), self.labels, row['job'])
        self.assertEqual(prompt['input_ids'], enc['input_ids'][:-1])  # same serving prompt, no target
        self.assertEqual(len(prompt['candidate_ids']), count)
        self.assertEqual(prompt['names'], [o['name'] for o in row['job']['options']])
        text = bytes(prompt['input_ids']).decode()
        for label, option in zip(self.labels, row['job']['options']):
            self.assertIn(f'"label":"{label.text}","name":"{option["name"]}"', text)
        self.assertNotIn('"label":"D"', text) if count == 3 else self.assertIn('"label":"D"', text)

    def test_flat_state_three_options(self):
        self.check(make_row('train', 0), 3)

    def test_position_state_four_options(self):
        self.check(make_row('train', 1, 'long', 'reduce'), 4)

    def test_never_truncates(self):
        row = make_row('train', 0)
        with self.assertRaisesRegex(ValueError, 'never truncate'):
            ra.encode_row(Tokenizer(), self.labels, row, max_length=100)
        with self.assertRaisesRegex(ValueError, 'serving input limit'):
            ra.encode_prompt(Tokenizer(), self.labels, row['job'], max_length=100)


if __name__ == '__main__':
    unittest.main()
