import unittest
import numpy as np
from veros_control.diagnostics.ocean import threshold_depth, seasonal, overturning, weighted_mean, basin_masks

class DiagnosticsTests(unittest.TestCase):
    def test_mld_interpolated_reference_and_crossing(self):
        z=np.array([5.,15.,30.,60.])
        rho=25.+z[:,None,None]*.01
        self.assertAlmostEqual(threshold_depth(rho,z,np.array([[70.]]))[0,0],13.)
    def test_mld_bottom_and_missing(self):
        z=np.array([5.,15.,30.])
        rho=np.full((3,1,2),25.)
        result=threshold_depth(rho,z,np.array([[40.,8.]]))
        self.assertEqual(result[0,0],40.);self.assertTrue(np.isnan(result[0,1]))
        rho[1,0,0]=np.nan
        self.assertTrue(np.isnan(threshold_depth(rho,z,np.array([[40.,8.]]))[0,0]))
    def test_season_hemispheres(self):
        months=np.broadcast_to(np.arange(1,13)[:,None,None],(12,2,1))
        np.testing.assert_array_equal(seasonal(months,np.array([-30,30]),True)[:,0],[8,2])
        np.testing.assert_array_equal(seasonal(months,np.array([-30,30]),False)[:,0],[2,8])
    def test_overturning_sign_units_interfaces(self):
        # 1 Sv lower southward return, 1 Sv upper northward flow.
        v=np.array([[[-1.]],[[1.]]]);mask=np.ones_like(v,bool)
        psi=overturning(v,np.array([[1e4]]),np.array([100.,100.]),mask,np.ones((1,1,1),bool),np.array([-150.,-50.]))
        np.testing.assert_allclose(psi[0,:,0],[1,0])
    def test_weighting_and_masks(self):
        self.assertAlmostEqual(weighted_mean(np.array([1.,3.,np.nan]),np.array([3.,1.,100.])),1.5)
        masks=basin_masks(np.array([-40,60,180]),np.array([-60,0,45]))
        self.assertTrue(masks[0,0].all());self.assertTrue(masks[1,1,0]);self.assertFalse(masks[2,2,1])

if __name__=='__main__':unittest.main()
