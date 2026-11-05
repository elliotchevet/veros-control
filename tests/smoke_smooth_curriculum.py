"""Real small-grid best-checkpoint transitions with fixed loss balancing."""
import argparse
from pathlib import Path
from dataclasses import replace
import tempfile
import jax
import numpy as np
from veros_control.training.config import TrainConfig, InitialState, LossConfig, SchedulerConfig, CurriculumStage
from veros_control.core.model import ModelConfig
from veros_control.core.correction import CorrectionConfig
from veros_control.core.checkpoint import load_checkpoint
from veros_control.training.train import train
parser=argparse.ArgumentParser();parser.add_argument('data');args=parser.parse_args()
root=Path(__file__).resolve().parents[1];data=Path(args.data).resolve()
with tempfile.TemporaryDirectory(prefix='clean_curriculum_') as temp:
 cfg=TrainConfig(output=temp,setup=str(root/'tests/smoke_setup.py')+':create_setup',initial=str(data/'initial.nc'),target=str(data/'target.nc'),n_steps=3,sample_steps=1,updates=1,learning_rate=1e-3,model=ModelConfig(hidden_dim=4,depth=1,surface_only=True),correction=CorrectionConfig(momentum=False),loss=LossConfig(levels=1,normalization='no_nn'),reuse_rollouts=True,fixed_loss_balance=True,transition_checkpoint='best',scheduler=SchedulerConfig(mode='keep_update',min_rate=1e-5,start_update=1,recovery_interval=2,reject_worsening=True),curriculum=[CurriculumStage(3,1,1e-4),CurriculumStage(6,1,1e-5)])
 last=load_checkpoint(train(cfg));first=load_checkpoint(Path(temp)/'window_3/best.pkl')
 assert last['loss_term_denominators']==first['loss_term_denominators']
 assert last['baseline_calibration']==first['baseline_calibration']
 assert last['config']['transition']==str(Path(temp)/'window_3/best.pkl')
 assert last['train_history'][0]['learning_rate']==1e-5
 assert last['config']['scheduler']['start_update']==1
 print('PASS: real best-checkpoint curriculum, fixed first-stage normalization, no warmup, per-stage LR and rejection mode',flush=True)
