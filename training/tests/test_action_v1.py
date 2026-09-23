import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training'), str(ROOT / 'inference')]

import action_data as ad  # noqa: E402
import action_eval as ev  # noqa: E402
from jev_inference import action_task as at  # noqa: E402

T0 = 1_520_000_100 - 1_520_000_100 % at.STEP  # 15m boundary


def candles(n, start=T0 - 200 * at.STEP, price=10000.0, drift=0.001):
    out = []
    for i in range(n):
        o = price
        price *= 1 + drift * (1 if i % 3 else -1)
        out.append(dict(time=start + i * at.STEP, open=o, high=max(o, price) * 1.001, low=min(o, price) * 0.999,
                        close=price, volume=100.0 + i))
    return out


class LabelTests(unittest.TestCase):
    def label(self, fills, prior=0):
        fills = [ad.Fill(float(t), q, 100.0) for t, q in fills]
        return ad.label_window(fills, [f.time for f in fills], T0, prior, ad.two_sided_times(fills))

    def test_no_fill_is_hold(self):
        self.assertEqual(self.label([(T0 - 1, 5), (T0 + at.STEP, 5)])['label'], 'hold')

    def test_fill_at_cutoff_belongs_to_target(self):
        self.assertEqual(self.label([(T0, 5)])['label'], 'open_long')

    def test_first_episode_gap_and_opposite(self):
        # buy 5, then after 61s gap a sell of 20: only the first episode counts
        self.assertEqual(self.label([(T0 + 1, 5), (T0 + 62, -20)], prior=-10)['label'], 'reduce')
        # contiguous buys fully close a short
        self.assertEqual(self.label([(T0 + 1, 5), (T0 + 30, 5)], prior=-10)['label'], 'close')
        # reversal counts as close
        self.assertEqual(self.label([(T0 + 1, 15)], prior=-10)['label'], 'close')
        self.assertEqual(self.label([(T0 + 1, -3)], prior=-10)['label'], 'add')

    def test_two_sided_timestamp_is_ambiguous(self):
        self.assertTrue(self.label([(T0 + 5, 5), (T0 + 5, -5)])['ambiguous'])

    def test_splits_and_purge(self):
        self.assertIsNone(ad.split_for(ad.epoch('2018-04-01T12:00:00Z')))
        self.assertEqual(ad.split_for(ad.epoch('2018-04-02T00:00:00Z')), 'validation')
        self.assertEqual(ad.split_for(ad.epoch('2018-03-31T23:45:00Z')), 'train')

    def test_aggregate_rejects_gap(self):
        buckets = [dict(end=T0 + 60 * (k + 1), open=1, high=1, low=1, close=1, volume=1) for k in range(30)]
        self.assertEqual(len(ad.aggregate_15m(buckets)), 2)
        del buckets[20]
        with self.assertRaises(ValueError):
            ad.aggregate_15m(buckets)


class EvalTests(unittest.TestCase):
    def test_margin_only_moves_hold(self):
        self.assertEqual(ev.decide(['hold', 'add'], [1.0, 0.5], 0.0), 'hold')
        self.assertEqual(ev.decide(['hold', 'add'], [1.0, 0.5], 0.6), 'add')
        self.assertEqual(ev.decide(['add', 'hold'], [0.5, 1.0], -5), 'hold')

    def test_metrics_trade_f1(self):
        rows = [dict(target_action=a, side='flat') for a in ('hold', 'open_long', 'open_short', 'hold')]
        m = ev.metrics(rows, ['hold', 'open_long', 'hold', 'open_short'])
        self.assertAlmostEqual(m['trade_precision'], 0.5)
        self.assertAlmostEqual(m['trade_recall'], 0.5)
        self.assertEqual(m['teacher_trade_rate'], 0.5)

    def test_rollout_accounting_and_absorption(self):
        cs = candles(300)
        start = cs[at.LOOKBACK]['time']
        end = start + 8 * at.STEP
        script = iter(['open_long', 'add', 'hold', 'reduce', 'close', 'open_short', 'hold', 'close'])
        r = ev.rollout(cs, start, end, lambda job: next(script), 'test')
        self.assertEqual((r['opens'], r['closes'], r['n_decisions']), (2, 2, 8))
        # traded units: 1 + 1 + 1 + 1 + 1 + 1 = 6 -> 6 * 7.5bps
        self.assertAlmostEqual(r['fees_pct'], 100 * 6 * ev.FEE_RATE)
        hold = ev.rollout(cs, start, end, lambda job: 'hold', 'test')
        self.assertEqual(hold['time_in_position_fraction'], 0.0)

    def test_gate_rejects_absorbing(self):
        m = dict(trade_f1=0.5, predicted_trade_rate=0.1, teacher_trade_rate=0.1, action_macro_f1=0.4)
        b = dict(trade_f1=0.4, action_macro_f1=0.3)
        ok = ev.gate(m, b, dict(opens=9, closes=9, time_in_position_fraction=0.5))
        bad = ev.gate(m, b, dict(opens=2, closes=1, time_in_position_fraction=0.999))
        self.assertTrue(ok['passed'])
        self.assertFalse(bad['passed'])


if __name__ == '__main__':
    unittest.main()
