"""Action-space evaluation and sanitized pilot input safety checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from run_real import (ACTIONS, MODEL_ID, MODEL_REVISION, TASK,
                      classification_metrics, load_dataset, summarize_predictions)


class MetricsTests(unittest.TestCase):
    def test_shuffled_option_indices_are_mapped_back_to_actions(self):
        orderings = [("flat", "long", "short"), ("short", "flat", "long"), ("long", "short", "flat")]
        examples = [{"job": SimpleNamespace(options=[SimpleNamespace(name=a) for a in names]),
                     "target_action": action, "previous_action": "long"}
                    for names, action in zip(orderings, ACTIONS)]
        results = [{"prediction": 1}, {"prediction": 0}, {"prediction": 2}]
        metrics = summarize_predictions(examples, results)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["macro_f1"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertAlmostEqual(metrics["persistence_baseline"]["accuracy"], 1/3)
        self.assertEqual(metrics["position_change_subset"], {"count": 2, "accuracy": 1, "persistence_accuracy": 0})

    def test_absent_class_does_not_inflate_macro_f1(self):
        metrics = classification_metrics(["long", "long"], ["long", "long"])
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], 1/3)
        self.assertEqual(metrics["per_class"]["flat"]["support"], 0)

    def test_zero_change_subset_is_missing_not_zero_accuracy(self):
        rows = [{"job": SimpleNamespace(options=[SimpleNamespace(name=a) for a in ACTIONS]),
                 "target_action": "long", "previous_action": "long"}]
        metrics = summarize_predictions(rows, [{"prediction": 0}])
        self.assertIsNone(metrics["position_change_subset"]["accuracy"])


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.rows = {}
        for split, count, date in [("train", 4, "2018-06-01"), ("validation", 12, "2021-02-01"), ("test", 12, "2021-08-01")]:
            start = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
            self.rows[split] = [
                {"job": {"state": {"position": "flat"}, "instructions": "Predict next-hour position.",
                         "options": [{"name": name} for name in ACTIONS]},
                 "target_index": i % 3, "target_action": ACTIONS[i % 3], "previous_action": "flat",
                 "cutoff": (start + timedelta(hours=i)).isoformat(), "split": split}
                for i in range(count)]
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        manifest = {"dataset_id": "abc123", "task": TASK, "model": MODEL_ID, "revision": MODEL_REVISION,
                    "provenance": "test_fixture", "files": {}, "splits": {}}
        for split, rows in self.rows.items():
            filename = split + ".jsonl"
            content = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            (self.directory / filename).write_bytes(content)
            manifest["files"][filename] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            manifest["splits"][split] = {"count": len(rows)}
        (self.directory / "manifest.json").write_text(json.dumps(manifest))

    def test_verified_dataset_loads(self):
        manifest, rows = load_dataset(self.directory)
        self.assertEqual(manifest["dataset_id"], "abc123")
        self.assertEqual(rows["train"][0]["job"].options[0].name, "long")

    def test_modified_content_rejected(self):
        with (self.directory / 'train.jsonl').open('ab') as handle:
            handle.write(b'\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            load_dataset(self.directory)

    def test_wrong_target_option_rejected_without_record_leak(self):
        self.rows['train'][0]['target_action'] = 'short'
        self.rows['train'][0]['job']['state']['account'] = 'sensitive-example'
        self.write()
        with self.assertRaisesRegex(ValueError, '^invalid sanitized row: train.jsonl:1$'):
            load_dataset(self.directory)

    def test_duplicate_cutoff_across_splits_rejected(self):
        self.rows['validation'][0]['cutoff'] = self.rows['train'][0]['cutoff']
        self.write()
        with self.assertRaises(ValueError):
            load_dataset(self.directory)

    def test_purge_gap_required(self):
        origin = datetime.fromisoformat(self.rows['train'][-1]['cutoff']) + timedelta(hours=1)
        for i, row in enumerate(self.rows['validation']):
            row['cutoff'] = (origin + timedelta(hours=i)).isoformat()
        self.write()
        with self.assertRaisesRegex(ValueError, '96-hour purge'):
            load_dataset(self.directory)

    def test_naive_timestamp_rejected(self):
        self.rows['train'][0]['cutoff'] = '2018-06-01T00:00:00'
        self.write()
        with self.assertRaises(ValueError):
            load_dataset(self.directory)

    def test_unexpected_raw_field_rejected(self):
        self.rows['train'][0]['orderID'] = 'not-allowed'
        self.write()
        with self.assertRaises(ValueError):
            load_dataset(self.directory)

    def test_nested_raw_identifier_rejected(self):
        self.rows['train'][0]['job']['state']['executions'] = [{"execID": "private-value"}]
        self.write()
        with self.assertRaisesRegex(ValueError, '^invalid sanitized row: train.jsonl:1$'):
            load_dataset(self.directory)

    def test_symlink_input_rejected(self):
        path = self.directory / 'train.jsonl'
        target = self.directory / 'outside.jsonl'
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            load_dataset(self.directory)


if __name__ == '__main__':
    unittest.main()
