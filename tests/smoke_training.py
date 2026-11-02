"""Real small-grid curriculum, resume and full-depth training smoke test."""
import argparse
from dataclasses import replace
from pathlib import Path
import h5netcdf
import numpy as np
import jax
from veros_control.training.config import TrainConfig, LossConfig, SchedulerConfig, CurriculumStage
from veros_control.core.model import ModelConfig
from veros_control.core.correction import CorrectionConfig
from veros_control.core.checkpoint import load_checkpoint
from veros_control.training.train import train

def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    for label,delta in [('initial',0.),('target',.1)]:
        with h5netcdf.File(output/f'{label}.nc','w') as nc:
            nc.dimensions=dict(time=1,zt=4,lat=8,lon=8)
            depth=np.array([-800.,-450.,-200.,-50.])
            fields={'thetao':(10.+.005*depth+delta)[:,None,None]*np.ones((4,8,8)),
                'so':np.full((4,8,8),35.+delta*.1),'uo':np.zeros((4,8,8)),'vo':np.zeros((4,8,8))}
            for key,value in fields.items():nc.create_variable(key,('time','zt','lat','lon'),'f8')[0]=value
            nc.create_variable('zos',('time','lat','lon'),'f8')[:]=0.
    base=TrainConfig(output=str(output/'surface'),setup=str(Path(__file__).with_name('smoke_setup.py').resolve())+':create_setup',
        initial=str(output/'initial.nc'),target=str(output/'target.nc'),n_steps=27,sample_steps=27,
        updates=1,learning_rate=1e-4,model=ModelConfig(hidden_dim=4,depth=1,surface_only=True,residual_blocks=1),
        correction=CorrectionConfig(momentum=False),loss=LossConfig(levels=1,normalization='no_nn'),
        scheduler=SchedulerConfig(mode='keep_update',start_update=1),
        curriculum=[CurriculumStage(27,1),CurriculumStage(81,1),CurriculumStage(243,1)])
    checkpoint=train(base);saved=load_checkpoint(checkpoint)
    assert saved['update']==1 and saved['config']['n_steps']==243 and np.isfinite(saved['checkpoint_loss'])
    resumed=train(replace(base,curriculum=[],n_steps=243,output=str(output/'resume'),resume=str(checkpoint)))
    assert load_checkpoint(resumed)['update']==2
    jax.clear_caches()
    full=train(replace(base,curriculum=[],output=str(output/'full'),model=replace(base.model,surface_only=False),loss=replace(base.loss,levels=0),n_steps=27))
    assert np.isfinite(load_checkpoint(full)['checkpoint_loss'])
    print('PASS: real Veros 27/81/243 curriculum, optimizer resume and full-depth training',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');run(p.parse_args().output)
