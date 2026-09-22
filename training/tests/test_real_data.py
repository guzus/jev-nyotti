import csv
from pathlib import Path
import random
import tempfile
import unittest

from real_data import HOUR, LOOKBACK, iso, make_example, position_before, read_positions, split_for, timestamp


class RealDataTests(unittest.TestCase):
    def test_position_uses_only_events_strictly_before_cutoff(self):
        times, positions = [10, 20, 30], [100, -50, 0]
        self.assertEqual(position_before(times, positions, 10), 0)
        self.assertEqual(position_before(times, positions, 20), 100)
        self.assertEqual(position_before(times, positions, 20.001), -50)
        self.assertEqual(position_before(times, positions, 31), 0)

    def test_target_cannot_change_inputs_and_only_closed_candles_are_supplied(self):
        cutoff = timestamp('2020-03-01T00:00:00Z')
        history = [{'end': cutoff-(LOOKBACK-1-i)*HOUR, 'open': 100+i,
                    'high': 102+i, 'low': 99+i, 'close': 101+i, 'volume': 10+i} for i in range(LOOKBACK)]
        a = make_example(history, 'long', 'short', 'train', random.Random(42))
        b = make_example(history, 'long', 'flat', 'train', random.Random(42))
        self.assertEqual(a['job'], b['job'])
        self.assertNotEqual(a['target_index'], b['target_index'])
        state = a['job']['state']
        self.assertEqual(state['data_cutoff'], iso(cutoff))
        self.assertEqual(max(c['time']+HOUR for c in state['recent_closed_candles']), cutoff)
        self.assertEqual(state['position_side_before_cutoff'], 'long')
        self.assertFalse({'execid', 'orderid', 'account', 'target_action', 'lastqty', 'lastpx'} & state.keys())

    def test_chronological_splits_purge_full_feature_window(self):
        boundary = timestamp('2021-01-01T00:00:00Z')
        self.assertEqual(split_for(boundary-HOUR), 'train')
        self.assertIsNone(split_for(boundary))
        self.assertIsNone(split_for(boundary+95*HOUR))
        self.assertEqual(split_for(boundary+96*HOUR), 'validation')
        boundary = timestamp('2021-07-01T00:00:00Z')
        self.assertEqual(split_for(boundary-HOUR), 'validation')
        self.assertIsNone(split_for(boundary))
        self.assertEqual(split_for(boundary+96*HOUR), 'test')
        self.assertIsNone(split_for(timestamp('2022-01-01T00:00:00Z')))

    def test_unsorted_partial_fills_reconcile_without_exposing_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'execution.csv'
            rows = [dict(symbol='XBTUSD', exectype='Trade', execid='b', lastqty='6', side='Buy', transacttime=iso(20)),
                    dict(symbol='XBTUSD', exectype='Trade', execid='a', lastqty='4', side='Buy', transacttime=iso(10))]
            rows += [dict(symbol='XBTUSD', exectype='Funding', execid=f'f{i}', lastqty='10', side='Sell', transacttime=iso(30+i)) for i in range(100)]
            with path.open('w', newline='') as handle:
                writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(reversed(rows))
            times, positions, report=read_positions([path])
            self.assertEqual(times, [10,20]);self.assertEqual(positions, [4,10])
            self.assertEqual(report['funding_anchors_matched'],100)
            # Funding at a trade timestamp observes the pre-trade position.
            rows += [dict(symbol='XBTUSD', exectype='Trade', execid='tie-buy', lastqty='3', side='Buy', transacttime=iso(30)),
                     dict(symbol='XBTUSD', exectype='Trade', execid='tie-sell', lastqty='3', side='Sell', transacttime=iso(30))]
            with path.open('w', newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(reversed(rows))
            tied_times, tied_positions, _ = read_positions([path])
            self.assertEqual(tied_times, [10,20,30])
            self.assertEqual(tied_positions, [4,10,10])
            rows[0]['lastqty']='7'
            with path.open('w',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaisesRegex(ValueError,'does not reconcile'):
                read_positions([path])


if __name__ == '__main__':
    unittest.main()
