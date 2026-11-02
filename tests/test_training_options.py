import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import h5netcdf
import jax.numpy as jnp
import numpy as np
from veros_control.training.config import TrainConfig, SchedulerConfig, CurriculumStage, load_config
from veros_control.training.scheduler import backtrack_candidate, loss_test_next_learning_rate
from veros_control.training.batches import batch_months
from veros_control.data.reference import monthly_clock
from veros_control.data.forcing import plan_forcing_window, local_forcing_indices


class TrainingOptions(unittest.TestCase):
    def test_monthly_clock_and_december_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for month,day in [('jan',15.),('dec',349.)]:
                p=Path(directory)/(month+'.nc');paths.append(p)
                with h5netcdf.File(p,'w') as nc:
                    nc.dimensions={'time':1};v=nc.create_variable('time',('time',),'f8')
                    v.attrs['units']='days since 1993-01-01';v[:]=day
            shift,interval=monthly_clock(paths[1],paths[0])
            self.assertAlmostEqual(shift,349.*86400*360/365)
            self.assertAlmostEqual(interval,31.*86400*360/365)

    def test_batch_passes_and_config_validation(self):
        months=tuple(range(1,13))
        self.assertEqual(sorted(sum([list(batch_months(months,3,7,u)) for u in range(4)],[])),list(months))
        c=TrainConfig('output','setup:create','initial','target',learning_rate=1e-7)
        c.validate_options()  # fixed rates need not use an adaptive floor
        with self.assertRaises(ValueError):replace(c,curriculum=[CurriculumStage(81,1),CurriculumStage(27,1)]).validate_options()
        for path in (Path(__file__).parents[1]/'examples').glob('train*.json'):
            load_config(path)

    def test_guarded_update_and_scheduler_recovery(self):
        params={'x':jnp.array(1.)};direction={'x':jnp.array(1.)}
        with self.assertRaises(FloatingPointError):
            backtrack_candidate(params,direction,.001,lambda p:(False,{}),max_retries=0)
        self.assertEqual(float(params['x']),1.)
        rate,report=loss_test_next_learning_rate(.001,1.,2.,20)
        self.assertEqual(rate,.0005);self.assertTrue(report['update_kept'])
        streak=0
        for epoch in range(21,31):
            rate,report=loss_test_next_learning_rate(rate,1.,.9,epoch,clean_updates=streak)
            streak=report['lr_clean_updates']
        self.assertEqual(rate,.00075)

    def test_forcing_leading_margin(self):
        window=plan_forcing_window(30.,2,1.,3.,100,leading_margin=1)
        i,j,valid=local_forcing_indices(9,10,window['start'],100,window['length'],np)
        self.assertTrue(valid);self.assertEqual((i,j),(0,1))
