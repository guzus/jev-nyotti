import unittest
from fetch_binance_replay_market import normalize

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

if __name__=='__main__': unittest.main()
