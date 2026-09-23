import unittest
from fetch_binance_replay_market import action_replay_input, normalize

class MarketNormalizationTests(unittest.TestCase):
    def test_archive_microseconds_and_rest_milliseconds_are_same_candle(self):
        fields=['100','105','95','102','3']
        self.assertEqual(normalize([1735689600000000,*fields]),normalize([1735689600000,*fields]))
        self.assertEqual(normalize([1735689600000,*fields])['time'],1735689600)

    def test_bad_ohlc_rejected(self):
        for fields in [['100','99','95','102','3'],['100','105','101','102','3'],['100','105','95','NaN','3'],['100','105','95','102','-1']]:
            with self.assertRaises(ValueError):normalize([1735689600000,*fields])

    def test_off_hour_rejected(self):
        with self.assertRaises(ValueError):normalize([1735689601000,'100','105','95','102','3'])

class FifteenMinuteTests(unittest.TestCase):
    def test_15m_alignment(self):
        self.assertEqual(normalize([1735690500000,'100','105','95','102','3'],900)['time'],1735690500)
        with self.assertRaises(ValueError):normalize([1735690500000,'100','105','95','102','3'])

    def rows(self,first,count):
        return [dict(time=first+i*900,open=1.0,high=2.0,low=0.5,close=1.5,volume=3.0) for i in range(count)]

    def test_replay_input_requires_96_prior_and_execution_candles(self):
        start=1735689600;end=start+4*900
        result={'marketPair':'BTCUSDT','rows':100,'sha256':'x','sources':[],'downloadErrors':[]}
        ordered=self.rows(start-96*900-900,101)  # one extra early candle is trimmed
        out=action_replay_input(start,end,[('BTCUSD',result,ordered)],'2025-01-02T00:00:00Z')
        self.assertEqual((out['from'],out['to']),('2025-01-01T00:00:00Z','2025-01-01T01:00:00Z'))
        candles=out['series'][0]['candles']
        self.assertEqual((len(candles),candles[0]['time'],candles[-1]['time']),(100,start-96*900,end-900))
        self.assertEqual(out['series'][0]['market'],'Binance Spot BTCUSDT')
        for broken in (ordered[:-1],ordered[:50]+ordered[51:]):
            with self.assertRaises(ValueError):action_replay_input(start,end,[('BTCUSD',result,broken)],'t')

if __name__=='__main__': unittest.main()
