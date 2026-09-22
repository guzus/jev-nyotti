import random
from types import SimpleNamespace
import unittest
from real_data import HOUR, LOOKBACK, timestamp, iso
from transition_data import market_only_example, select_training, diagnostics, diagnostic_gate, validate_cohorts

class TransitionTests(unittest.TestCase):
    def test_prior_and_target_cannot_change_market_only_input(self):
        t=timestamp('2020-03-01T00:00:00Z')
        history=[dict(end=t-(LOOKBACK-1-i)*HOUR,open=100+i,high=102+i,low=99+i,close=101+i,volume=10+i) for i in range(LOOKBACK)]
        a=market_only_example(history,'flat','long','train',random.Random(42))
        b=market_only_example(history,'short','flat','train',random.Random(42))
        self.assertEqual(a['job'],b['job'])
        self.assertNotIn('position_side_before_cutoff',a['job']['state'])
        self.assertEqual(a['previous_action'],'flat')
        self.assertEqual(b['previous_action'],'short')

    def test_sampling_is_balanced_unique_and_deterministic(self):
        rows=[(i,'long','short' if i%3==0 else 'long') for i in range(100)]
        a=select_training(rows,random.Random(1),20)
        self.assertEqual(a,select_training(rows,random.Random(1),20))
        self.assertEqual(len(set(a)),20)
        self.assertEqual(sum(p!=t for _,p,t in a),10)

    def test_natural_false_signals_and_rollout_are_separate_from_event_slice(self):
        t=timestamp('2021-02-01T00:00:00Z')
        actions=['long','short','flat']
        # One correct reversal, one false reversal, one missed true reversal.
        rows=[dict(cutoff=iso(t+i*HOUR),previous_action='long',target_action=a,
                   job=SimpleNamespace(options=[SimpleNamespace(name=x) for x in actions]))
              for i,a in enumerate(['short','long','short'])]
        out=diagnostics(rows,[dict(prediction=i) for i in [1,1,0]],
                        dict(natural=[r['cutoff'] for r in rows],transitions=[rows[0]['cutoff'],rows[2]['cutoff']],rollout=[r['cutoff'] for r in rows]))
        self.assertEqual(out['natural']['correct_transition_precision'],.5)
        self.assertEqual(out['natural']['correct_transition_recall'],.5)
        self.assertEqual(out['natural']['false_change_rate'],1)
        self.assertEqual(out['transitions']['position_change_subset']['count'],2)
        self.assertEqual(out['rollout']['flat_start_rollout']['entries'],1)
        self.assertEqual(out['rollout']['flat_start_rollout']['reversals'],1)
        self.assertFalse(diagnostic_gate(out,out)['passed'])

    def test_cohort_preflight_rejects_missing_duplicate_or_gapped_membership(self):
        t=timestamp('2021-02-01T00:00:00Z')
        rows=[dict(cutoff=iso(t+i*HOUR),previous_action='long',target_action='short' if i<20 else 'long') for i in range(256)]
        cohorts=dict(natural=[r['cutoff'] for r in rows],transitions=[r['cutoff'] for r in rows[:20]],rollout=[r['cutoff'] for r in rows[:96]])
        validate_cohorts(rows,cohorts)
        for key,value in [('transitions',cohorts['transitions'][:-1]),('natural',cohorts['natural'][:-1]+cohorts['natural'][:1]),('rollout',[rows[0]['cutoff']]+[r['cutoff'] for r in rows[2:97]])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cohorts(rows,{**cohorts,key:value})

if __name__=='__main__':unittest.main()
