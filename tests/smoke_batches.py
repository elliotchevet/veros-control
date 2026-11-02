"""Real monthly batch training must resume with identical parameters and Adam state."""
from pathlib import Path
from dataclasses import replace
import argparse
import jax
import numpy as np
from veros_control.training.config import TrainConfig, InitialState, LossConfig, SchedulerConfig
from veros_control.core.model import ModelConfig
from veros_control.core.correction import CorrectionConfig
from veros_control.core.checkpoint import load_checkpoint
from veros_control.training.train import train


def run(data, output):
    data=Path(data).resolve();output=Path(output).resolve()
    initial=str(data/'initial.nc');target=str(data/'target.nc')
    config=TrainConfig(output=str(output/'continuous'),
        setup=str(Path(__file__).with_name('smoke_setup.py').resolve())+':create_setup',
        initial=initial,target=target,n_steps=3,sample_steps=1,updates=2,seed=4,learning_rate=1e-4,
        model=ModelConfig(hidden_dim=4,depth=1,surface_only=True),correction=CorrectionConfig(momentum=False),
        loss=LossConfig(levels=1,normalization='no_nn'),scheduler=SchedulerConfig(mode='keep_update',start_update=1),
        initial_states=[InitialState(initial,target,month=m,forcing_time_shift=0.) for m in (1,2)],
        batch_size=1,checkpoint_months=[1,2])
    continuous=load_checkpoint(train(config))
    first=train(replace(config,output=str(output/'first'),updates=1))
    resumed=load_checkpoint(train(replace(config,output=str(output/'resumed'),updates=1,resume=str(first))))
    for key in ('params','opt_state'):
        for a,b in zip(jax.tree_util.tree_leaves(continuous[key]),jax.tree_util.tree_leaves(resumed[key])):
            np.testing.assert_allclose(a,b,rtol=0,atol=0)
    assert continuous['learning_rate']==resumed['learning_rate']
    assert [r['batch_months'] for r in continuous['train_history']]==[r['batch_months'] for r in resumed['train_history']]
    assert continuous['loss_term_denominators']==resumed['loss_term_denominators']
    assert continuous['checkpoint_loss']==resumed['checkpoint_loss']
    print('PASS: monthly shuffled batches, calibration, fixed panel and exact Adam/weights/LR resume')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('output');a=p.parse_args();run(a.data,a.output)
