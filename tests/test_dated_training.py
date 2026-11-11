import unittest
from datetime import datetime
from types import SimpleNamespace
import tempfile
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from veros_control.data.dated import model_timestamp,forcing_positions,dated_files
from veros_control.training.scheduler import loss_test_next_learning_rate
from veros_control.core.preprocess import encode_date,extract_features
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
        model=VerticalCorrectionMLP(nz=4,hidden_dim=8,surface_only=True,use_date_embedding=True,use_lon_lat=False,zero_output_init=False)
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

class DatedConfigTests(unittest.TestCase):
    def test_120_cases_clock_and_objective_resume_identity(self):
        import h5netcdf,json
        from dataclasses import replace
        from veros_control.training.config import TrainConfig,load_config,objective_signature
        from veros_control.training.batches import case_configs,batch_months
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            for year in range(1993,2004):
                for month in range(1,13):
                    path=p/f'GLORYS12_1deg_{year}{month:02d}_processed.nc'
                    with h5netcdf.File(path,'w') as f:
                        f.dimensions={'time':1};t=f.create_variable('time',('time',),'f8')
                        t[:]=0;t.attrs['units']=f'days since {year}-{month:02d}-01'
                (p/f'ERA5_1deg_{year}_Flux_processed.nc').touch()
            path=p/'config.json';path.write_text(json.dumps(dict(output='out',setup='setup:create',initial='initial.nc',target='target.nc',training_years=[1993,2002],reference_directory='.',forcing_directory='.',batch_size=3,checkpoint_months=[1,53,117],model={'use_date_embedding':True})))
            cfg=load_config(path);self.assertEqual(len(cfg.initial_states),120)
            cases=case_configs(cfg)
            self.assertEqual(cases[53].initial_date,'1997-05-01T00:00:00');self.assertEqual(cases[53].start_month,5)
            self.assertEqual(cases[120].reference_hours,720)
            self.assertTrue(cases[120].target.endswith('200301_processed.nc'))
            self.assertEqual(sorted(i for u in range(40) for i in batch_months(tuple(cases),3,0,u)),list(range(1,121)))
            signature=objective_signature(cfg)
            self.assertNotEqual(signature,objective_signature(replace(cfg,correct_evaporation_sign=True)))
            legacy=objective_signature(TrainConfig('out','setup:create','a','b'))
            self.assertNotIn('initial_date',legacy);self.assertNotIn('use_date_embedding',legacy['model'])
            cfg.initial_states[0]=replace(cfg.initial_states[0],initial_date='bad-date')
            with self.assertRaises(ValueError):cfg.validate_options()

class GlobalDatedForcingTests(unittest.TestCase):
    def test_actual_setup_loader_padding_and_units(self):
        import ast
        from unittest.mock import patch
        # Load the actual method without constructing the asset-dependent global grid.
        path=Path(__file__).parents[1]/'examples/global_ocean/setup.py'
        tree=ast.parse(path.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='prepare_dated_forcing')
        namespace={'npx':jnp}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
        setup=SimpleNamespace();state=SimpleNamespace(settings=SimpleNamespace(dt_tracer=3600.))
        for count in (10,11):
            fields=tuple(np.full((2,2,count),3600.*(i+1)) for i in range(8))
            with patch('veros_control.data.dated.load_window',return_value=(fields,2879)):
                namespace['prepare_dated_forcing'](setup,state,'unused',datetime(1993,12,31),27)
            self.assertEqual([z.shape for z in setup._dated_forcing[:-1]],[(2,2,12)]*6)
            np.testing.assert_allclose(setup._dated_forcing[2],12.)
            np.testing.assert_allclose(setup._dated_forcing[4],7.)
            self.assertEqual(int(setup._dated_forcing[-1]),2879)
