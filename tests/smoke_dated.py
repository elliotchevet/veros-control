"""Actual Veros dated-case gradients, cached reuse, TCP workers."""
from pathlib import Path
from dataclasses import replace
from datetime import datetime
import tempfile,shutil,argparse,calendar
import h5netcdf
import numpy as np
import jax
import jax.numpy as jnp
from veros_control.core.model import ModelConfig,initialize
from veros_control.core.correction import CorrectionConfig,physical_model
from veros_control.data.dataset import create_simulation
from veros_control.training.config import TrainConfig,InitialState,LossConfig
from veros_control.training.batches import BatchObjective
from veros_control.training.workers import ParallelObjective
from veros_control.core.preprocess import fit_statistics,extract_features


def run(data):
    data=Path(data).resolve()
    with tempfile.TemporaryDirectory(prefix='clean_dated_') as tmp:
        root=Path(tmp);cases=[]
        for year,month in [(1993,12),(1994,1),(1996,2),(1996,3)]:
            p=root/f'GLORYS12_1deg_{year}{month:02d}_processed.nc';shutil.copy2(data/'initial.nc',p)
            with h5netcdf.File(p,'a') as f:
                if 'time' not in f.variables: f.create_variable('time',('time',),'f8')
                f['time'][:]=0;f['time'].attrs['units']=f'days since {year}-{month:02d}-01'
                f['thetao'][:]+=(year-1993)*.1
        from veros_control.data.era5_sequence import VARIABLES
        for year in (1993,1994,1996):
            n=(366 if calendar.isleap(year) else 365)*24
            with h5netcdf.File(root/f'ERA5_1deg_{year}_Flux_processed.nc','w') as f:
                f.dimensions={'time':n,'lat':1,'lon':1}
                for name in VARIABLES:f.create_variable(name,('time','lat','lon'),'f8')[:]=np.ones((n,1,1))*(year-1990)
        path=lambda y,m:str(root/f'GLORYS12_1deg_{y}{m:02d}_processed.nc')
        cases=[InitialState(path(1993,12),path(1994,1),1,initial_date='1993-12-01'),InitialState(path(1996,2),path(1996,3),2,initial_date='1996-02-01')]
        cfg=TrainConfig(str(root/'out'),str(Path(__file__).with_name('smoke_setup.py'))+':create_dated_setup',cases[0].initial,cases[0].target,n_steps=3,sample_steps=1,normalization='meanstd',initial_states=cases,forcing_directory=str(root),batch_size=2,model=ModelConfig(hidden_dim=4,depth=1,surface_only=True,use_date_embedding=True),correction=CorrectionConfig(momentum=False),loss=LossConfig(levels=1))
        cfg.validate_options()
        from veros_control.training.batches import case_configs
        sim,_=create_simulation(case_configs(cfg)[1]);sim.set_forcing(sim.state)
        stats=fit_statistics(sim.state,'meanstd',True)
        model,params=initialize(cfg.model,4,0);physical=physical_model(model,cfg.correction,sim.state.variables.zt)
        regular=BatchObjective(cfg,sim,physical,stats);fast=BatchObjective(replace(cfg,reuse_rollouts=True),sim,physical,stats)
        for sample in (1,2,1):
            wanted=regular.mean((sample,),params,jnp.ones(2),gradient=True)
            result=fast.mean((sample,),params,jnp.ones(2),gradient=True)
            for a,b in zip(jax.tree_util.tree_leaves(wanted),jax.tree_util.tree_leaves(result)):
                np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
            features=extract_features(fast.initial,fast.initial.variables.tau,stats,True,scaling_method='meanstd',use_forcings=True)
            assert features.shape[-1]==12
        assert fast.gradient_one._cache_size()==1
        wanted=fast.mean((1,2),params,jnp.ones(2),gradient=True)
        parallel=ParallelObjective(BatchObjective(replace(cfg,reuse_rollouts=True,worker_commands=[[],[]]),sim,physical,stats))
        try:
            result=parallel.mean((1,2),params,jnp.ones(2),gradient=True)
            for a,b in zip(jax.tree_util.tree_leaves(wanted),jax.tree_util.tree_leaves(result)):
                np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
        finally:parallel.close()
        print('PASS: date inputs + yearly forcing, legacy/cached losses and gradients agree; one trace; two TCP workers agree.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');run(p.parse_args().data)
