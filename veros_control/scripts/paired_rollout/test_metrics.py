import unittest
import numpy as np
from metrics import basin_stats, objective

class MetricsTests(unittest.TestCase):
    def test_area_weights_mask_and_cancellation(self):
        field=np.array([2.,-1.,np.nan]).reshape(3,1,1)
        mean,rms,area=basin_stats(field,np.array([[1.],[2.],[9.]]),np.array([1,1,0],dtype=bool).reshape(3,1,1),np.ones((1,3,1),bool))
        self.assertAlmostEqual(mean[0,0],0.)
        self.assertAlmostEqual(rms[0,0],np.sqrt(2.))
        self.assertEqual(area[0,0],3.)
    def test_fixed_normalization_and_empty_pairs(self):
        bias={'temp':np.array([[2.,99.],[np.nan,99.]]),'salt':np.array([[3.,99.],[np.nan,99.]])}
        terms,total=objective(bias,[0],{'temp':4.,'salt':9.})
        np.testing.assert_allclose(terms,[4.,9.]);self.assertEqual(total,1.)
if __name__=='__main__':unittest.main()
