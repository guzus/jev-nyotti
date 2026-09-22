import unittest
from action_ledger import Inventory, classify_delta

class InventoryTests(unittest.TestCase):
    def test_inverse_harmonic_entry_and_partial_close(self):
        s=Inventory()
        self.assertEqual(s.apply(100,10000,1),'open_long')
        self.assertEqual(s.apply(100,20000,2),'add_long')
        self.assertAlmostEqual(s.entry_price,200/(.01+.005))
        before=s.entry_price
        self.assertEqual(s.apply(-50,15000,3),'reduce_long')
        self.assertEqual(s.entry_price,before)
        self.assertAlmostEqual(s.realized_xbt,50*(1/before-1/15000))
        self.assertEqual(s.quantity,150)
        self.assertEqual(s.opened_at,1)

    def test_short_buyback_is_reduction_not_long(self):
        s=Inventory();s.apply(-100,20000,1)
        self.assertEqual(s.apply(40,10000,2),'reduce_short')
        self.assertEqual(s.quantity,-60)
        self.assertAlmostEqual(s.realized_xbt,40*(1/10000-1/20000))
        self.assertEqual(classify_delta(-100,40),'reduce_short')

    def test_reverse_closes_then_reopens_at_fill_price(self):
        s=Inventory();s.apply(100,10000,1)
        self.assertEqual(s.apply(-150,12000,2),'reverse_to_short')
        self.assertEqual(s.quantity,-50)
        self.assertEqual(s.entry_price,12000)
        self.assertEqual(s.opened_at,2)
        self.assertAlmostEqual(s.realized_xbt,100*(1/10000-1/12000))
        self.assertEqual(s.apply(50,10000,3),'close_short')
        self.assertIsNone(s.entry_price)
        self.assertIsNone(s.opened_at)

    def test_snapshot_disallows_target_boundary_fill(self):
        s=Inventory();s.apply(100,10000,10)
        with self.assertRaises(ValueError):s.snapshot(11000,10)
        result=s.snapshot(11000,11)
        self.assertAlmostEqual(result['unrealized_xbt'],100*(1/10000-1/11000))
        self.assertEqual(result['position_age_seconds'],1)
        with self.assertRaises(ValueError):s.apply(10,11000,9)

    def test_classification_covers_all_intents(self):
        cases=[(0,0,'hold'),(0,2,'open_long'),(0,-2,'open_short'),(2,1,'add_long'),(-2,-1,'add_short'),(2,-1,'reduce_long'),(-2,1,'reduce_short'),(2,-2,'close_long'),(-2,2,'close_short'),(-2,3,'reverse_to_long'),(2,-3,'reverse_to_short')]
        for prior,delta,expected in cases:self.assertEqual(classify_delta(prior,delta),expected)

if __name__=='__main__':unittest.main()
