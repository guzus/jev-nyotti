import unittest
from types import SimpleNamespace
from jev_inference.replay_batch import ReplayBatchScorer,equivalent


def scores(values):
    return [SimpleNamespace(logits=v) for v in values]

class BatchGateTest(unittest.TestCase):
    def test_small_numerical_drift_passes(self):
        self.assertTrue(equivalent(scores([[2.,1.,0.]]),scores([[2.001,1.,0.]])))
    def test_same_argmax_large_drift_fails(self):
        self.assertFalse(equivalent(scores([[2.,1.,0.]]),scores([[4.,1.,0.]])))
    def test_changed_argmax_fails(self):
        self.assertFalse(equivalent(scores([[1.,1.001,0.]]),scores([[1.001,1.,0.]])))
    def test_first_gate_returns_serial_then_batch(self):
        baseline=scores([[2.,1.,0.],[1.,2.,0.]])
        engine=SimpleNamespace(score=lambda _:baseline)
        scorer=ReplayBatchScorer()
        candidate=scores([[2.001,1.,0.],[1.,2.001,0.]])
        scorer._batch=lambda *_:candidate
        self.assertIs(scorer.score(engine,[1,2]),baseline)
        self.assertTrue(scorer.enabled)
        self.assertIs(scorer.score(engine,[1,2]),candidate)
    def test_failed_gate_stays_serial(self):
        baseline=scores([[2.,1.,0.],[1.,2.,0.]])
        engine=SimpleNamespace(score=lambda _:baseline)
        scorer=ReplayBatchScorer()
        scorer._batch=lambda *_:scores([[0.,1.,2.],[1.,2.,0.]])
        self.assertIs(scorer.score(engine,[1,2]),baseline)
        self.assertFalse(scorer.enabled)
        scorer._batch=lambda *_:self.fail('batch must remain disabled')
        self.assertIs(scorer.score(engine,[1,2]),baseline)
