import unittest
from jev_inference.replay import window,index_candles,features,fingerprint

class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.cutoff=3600*100
        self.candles=[dict(time=i*3600,open=100+i,high=101+i,low=99+i,close=100+i,volume=1) for i in range(104)]
    def test_cutoff_excludes_execution_prices(self):
        prior,execution=window(index_candles(self.candles),self.cutoff)
        self.assertEqual(len(prior),96)
        self.assertEqual(prior[-1]['time'],self.cutoff-3600)
        self.assertEqual(execution['time'],self.cutoff)
        self.assertEqual(execution['close'],203)
        self.assertEqual(execution['volume'],4)
    def test_missing_hour_fails(self):
        with self.assertRaisesRegex(ValueError,'missing required'):
            window(index_candles(self.candles[5:]),self.cutoff)
    def test_duplicate_fails(self):
        with self.assertRaises(ValueError): index_candles(self.candles+self.candles[:1])
    def test_features(self):
        f=features(self.candles[:96])
        self.assertEqual(f['rsi14'],100)
        self.assertEqual(f['volumeRatio'],1)
        self.assertEqual(f['lastClose'],195)
    def test_fingerprint_order_stable_content_sensitive(self):
        self.assertEqual(fingerprint({'a':1,'b':2}),fingerprint({'b':2,'a':1}))
        self.assertNotEqual(fingerprint({'a':1}),fingerprint({'a':2}))
