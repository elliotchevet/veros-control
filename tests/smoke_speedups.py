"""Compare cached and TCP-worker objectives against ordinary Veros rollouts."""
from dataclasses import replace
from pathlib import Path
import argparse
import tempfile
import shutil
import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
from veros_control.training.config import TrainConfig, InitialState, LossConfig
from veros_control.core.model import ModelConfig, initialize
from veros_control.core.correction import CorrectionConfig, physical_model
from veros_control.data.dataset import create_simulation
from veros_control.training.batches import BatchObjective
from veros_control.training.workers import ParallelObjective


def run(data):
    data=Path(data).resolve()
    with tempfile.TemporaryDirectory(prefix='clean_speed_') as temp:
        root=Path(temp)
        other=root/'other.nc';shutil.copy2(data/'target.nc',other)
        with h5netcdf.File(other,'a') as f:f['thetao'][:]+=0.3
        cfg=TrainConfig(output=str(root/'workers'),
            setup=str(Path(__file__).with_name('smoke_setup.py').resolve())+':create_dynamic_setup',
            initial=str(data/'initial.nc'),target=str(data/'target.nc'),n_steps=3,sample_steps=1,
            model=ModelConfig(hidden_dim=4,depth=1,surface_only=True),correction=CorrectionConfig(momentum=False),
            loss=LossConfig(levels=1),initial_states=[
                InitialState(str(data/'initial.nc'),str(data/'target.nc'),1,forcing_time_shift=0.),
                InitialState(str(data/'target.nc'),str(other),2,reference_hours=672.,forcing_time_shift=3600.)],
            batch_size=2,checkpoint_months=[1,2])
        sim,_=create_simulation(cfg);model,params=initialize(cfg.model,4,0)
        physical=physical_model(model,cfg.correction,sim.state.variables.zt)
        original=BatchObjective(cfg,sim,physical,None)
        expected=[]
        for month in (1,2,1):expected.append(original.mean((month,),params,jnp.ones(2),gradient=True))
        fast=BatchObjective(replace(cfg,reuse_rollouts=True),sim,physical,None)
        for month,wanted in zip((1,2,1),expected):
            result=fast.mean((month,),params,jnp.ones(2),gradient=True)
            for a,b in zip(jax.tree_util.tree_leaves(result),jax.tree_util.tree_leaves(wanted)):
                np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
        assert fast.gradient_one._cache_size()==1,fast.gradient_one._cache_size()
        assert getattr(sim,'_training_forcing',None) is None
        print('PASS: different dates/targets/intervals; identical loss/gradients; one gradient compilation',flush=True)
        wanted=fast.mean((1,2),params,jnp.ones(2),gradient=True)
        parallel=ParallelObjective(BatchObjective(replace(cfg,reuse_rollouts=True,worker_commands=[[],[]]),sim,physical,None))
        try:
            parallel.member(1)
            actual=parallel.mean((1,2),params,jnp.ones(2),gradient=True)
            for a,b in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(wanted)):
                np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
            assert parallel.validate((1,2),params)==fast.validate((1,2),params)
            print('PASS: two persistent TCP workers agree with serial objective/gradient and validation',flush=True)
        finally:parallel.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');run(p.parse_args().data)
