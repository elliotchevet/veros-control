import unittest
from datetime import datetime
from types import SimpleNamespace
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from veros import runtime_settings
runtime_settings.backend='jax'
from veros_control.training.dated_months import model_timestamp,forcing_positions,dated_files
from veros_control.training.stability import loss_test_next_learning_rate
from veros_control.core.preprocessing import encode_date,extract_features
from veros_control.core.model import VerticalCorrectionMLP

class DatedTrainingTests(unittest.TestCase):
    def test_surface_flux_sign_and_restoring_switch(self):
        from veros_control.core.surface_fluxes import surface_tracer_fluxes
        args=(0.,25.,20.,35.,34.,0.,-1e-7,-40.,-6.37e-6,1024.,3991.86795711963)
        heat,salt=surface_tracer_fluxes(*args,disable_restoring=True,correct_evaporation_sign=True)
        self.assertEqual(heat,0.)
        self.assertAlmostEqual(salt,35e-7)
        heat_old,salt_old=surface_tracer_fluxes(*args)
        self.assertLess(heat_old,0.)
        self.assertLess(salt_old,0.)
        wet=list(args);wet[5]=2e-7
        self.assertLess(surface_tracer_fluxes(*wet,disable_restoring=True,correct_evaporation_sign=True)[1],0.)
    def test_january_does_not_require_previous_year(self):
        first,positions=forcing_positions(datetime(1993,1,1),27,3600)
        self.assertEqual(first,0)
        self.assertEqual(positions[0],(1993,0.))
        self.assertEqual({year for year,hour in positions},{1993})
    def test_leap_and_year_boundary(self):
        self.assertEqual(model_timestamp(datetime(1996,3,1))-model_timestamp(datetime(1996,2,1)),30*86400)
        self.assertEqual(model_timestamp(datetime(2003,1,1))-model_timestamp(datetime(2002,12,1)),30*86400)
        _,pos=forcing_positions(datetime(1996,2,15),243,3600)
        self.assertTrue(all(y==1996 for y,h in pos))
        self.assertAlmostEqual(pos[1][1],31*24+np.floor(14*30/29*8)*3*29/30)
        _,pos=forcing_positions(datetime(1999,12,31),81,3600)
        self.assertEqual({y for y,h in pos},{1999,2000})
    def test_two_good_scheduler_and_warmup(self):
        rate,report=loss_test_next_learning_rate(1e-3,1,2,20,start_epoch=21,recovery_interval=2)
        self.assertEqual(rate,1e-3)
        rate,report=loss_test_next_learning_rate(rate,1,2,21,start_epoch=21,recovery_interval=2)
        self.assertEqual(rate,5e-4)
        for epoch in (22,23):
            rate,report=loss_test_next_learning_rate(rate,1,.9,epoch,start_epoch=21,recovery_interval=2,clean_updates=report['lr_clean_updates'])
        self.assertEqual(rate,7.5e-4)
    def test_date_changes_output_and_surface_only_gradient(self):
        model=VerticalCorrectionMLP(nz=4,hidden_dim=8,surface_only=True,use_date_embedding=True,init_eps=1.)
        def features(offset):return jnp.concatenate((jnp.ones((4,4)),jnp.broadcast_to(encode_date(0.,offset),(4,3))),axis=-1)
        x=features(0);params=model.init(jax.random.PRNGKey(4),x)
        a=model.apply(params,x);b=model.apply(params,features(9))
        np.testing.assert_array_equal(a[:-1],0)
        self.assertGreater(float(jnp.linalg.norm(a-b)),0)
        np.testing.assert_allclose(encode_date(360*86400.,0),encode_date(0.,1),atol=1e-6)
    def test_no_december_target_wrap(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            for m in range(1,13):(p/f'GLORYS12_1deg_1993{m:02d}_processed.nc').touch()
            (p/'ERA5_1deg_1993_Flux_processed.nc').touch()
            with self.assertRaises(FileNotFoundError):dated_files(p,1993,1993)
            (p/'GLORYS12_1deg_199401_processed.nc').touch()
            pairs=dated_files(p,1993,1993)
            self.assertTrue(pairs[12][1].endswith('199401_processed.nc'))
if __name__=='__main__':unittest.main()
