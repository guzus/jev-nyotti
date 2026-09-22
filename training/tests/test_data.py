import collections
import unittest

from data import ACTIONS, encode_example, synthetic_examples
from jev_inference.labels import Label


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(x['content'] for x in messages) + '\nassistant:'

    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))


class DatasetTests(unittest.TestCase):
    def test_reproducible_balanced_labels_and_option_mapping(self):
        rows = synthetic_examples(192, 3407)
        self.assertEqual(rows, synthetic_examples(192, 3407))
        counts = collections.Counter()
        label_indices = set()
        for row in rows:
            job = row['job']
            change = job.state['features']['changePct']
            expected = 'long' if change > 1 else 'short' if change < -1 else 'hold'
            self.assertEqual(job.options[row['target_index']].name, expected)
            self.assertEqual(set(x.name for x in job.options), set(ACTIONS))
            self.assertEqual(job.state['provenance'], 'SYNTHETIC_REHEARSAL_NOT_TRADER_DATA')
            counts[expected] += 1
            label_indices.add(row['target_index'])
        self.assertEqual(set(counts.values()), {64})
        self.assertEqual(label_indices, {0, 1, 2})

    def test_distinct_evaluation_examples(self):
        train = {x['job'].model_dump_json() for x in synthetic_examples(192, 3407)}
        test = {x['job'].model_dump_json() for x in synthetic_examples(12, 99173)}
        self.assertFalse(train & test)

    def test_only_response_label_is_supervised(self):
        labels = [Label(x, ord(x)) for x in 'ABC']
        example = synthetic_examples(1, 42)[0]
        row = encode_example(Tokenizer(), labels, example, 20000)
        self.assertEqual(len(row['labels']), len(row['input_ids']))
        self.assertEqual(row['labels'][:-1], [-100] * (len(row['input_ids']) - 1))
        self.assertEqual(row['labels'][-1], labels[example['target_index']].token_id)
        self.assertEqual(row['input_ids'][-1], row['labels'][-1])
        with self.assertRaisesRegex(ValueError, 'never truncate'):
            encode_example(Tokenizer(), labels, example, len(row['input_ids']) - 1)

    def test_rejects_changed_token_boundary(self):
        class BrokenTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                ids = super().encode(text, **kwargs)
                return ids[:-2] + [999] if text[-1:] in 'ABC' else ids
        with self.assertRaisesRegex(ValueError, 'continuation'):
            encode_example(BrokenTokenizer(), [Label(x, ord(x)) for x in 'ABC'],
                           synthetic_examples(1, 42)[0], 20000)


if __name__ == '__main__':
    unittest.main()
