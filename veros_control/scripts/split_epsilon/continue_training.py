"""Controlled split-epsilon continuation using the unchanged nested training loop.

Checkpoints from this experiment must be resumed with this runner so that the
saved momentum scale is applied; the generic benchmark uses a shared epsilon.
"""
import argparse
import importlib.util
import inspect
import math
import os
from pathlib import Path
import pickle
import shutil
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--tracer-epsilon', type=float, default=1e-6)
parser.add_argument('--momentum-epsilon', type=float, default=1e-7)
parser.add_argument('--total-updates', type=int, default=None)
scales, remaining = parser.parse_known_args()
if not math.isfinite(scales.tracer_epsilon) or scales.tracer_epsilon <= 0:
    raise ValueError('tracer epsilon must be finite and positive')
if not math.isfinite(scales.momentum_epsilon) or scales.momentum_epsilon < 0:
    raise ValueError('momentum epsilon must be finite and nonnegative')
source = Path(__file__).resolve().parents[2] / 'training/ad_benchmark_root.py'
sys.argv = [str(source)] + remaining
spec = importlib.util.spec_from_file_location('split_epsilon_benchmark', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original_train = m.run_train_mode
original_payload = m._train_checkpoint_payload
resume_update = None


def run_train(*args, **kwargs):
    global resume_update
    bound = inspect.signature(original_train).bind(*args, **kwargs)
    cfg = bound.arguments['args']
    if cfg.train_epsilon_final is not None:
        raise ValueError('This controlled experiment uses fixed epsilon values')
    target = Path(cfg.train_output_file)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite experiment checkpoint: {target}')
    source_checkpoint = Path(os.environ['SOURCE_CHECKPOINT'])
    with source_checkpoint.open('rb') as f:
        checkpoint = pickle.load(f)
    resume_update = int(checkpoint['update'])
    for name in ('tracer_epsilon', 'momentum_epsilon'):
        if name in checkpoint and checkpoint[name] != getattr(scales, name):
            raise ValueError(f'Saved {name} differs from requested continuation scale')
    if checkpoint.get('opt_state') is None:
        raise ValueError('Continuation requires the saved optimizer state')
    if scales.total_updates is not None:
        if scales.total_updates <= resume_update:
            raise ValueError('Target update must exceed checkpoint update')
        cfg.train_updates = scales.total_updates - resume_update
    shutil.copy2(source_checkpoint, target)
    cfg.epsilon = scales.tracer_epsilon
    base_model = bound.arguments['model_batched_unscaled']
    ratio = m.jnp.asarray([1., 1., scales.momentum_epsilon/scales.tracer_epsilon,
                           scales.momentum_epsilon/scales.tracer_epsilon])
    def scaled_model(params, features, coordinates, dzt):
        predictions = base_model(params, features, coordinates, dzt)
        return m.jnp.where(ratio != 0, predictions * ratio, 0.)
    bound.arguments['model_batched_unscaled'] = scaled_model
    print(f'SPLIT EPSILON: tracer={scales.tracer_epsilon:g}, momentum={scales.momentum_epsilon:g}', flush=True)
    print(f'Resuming checkpoint {resume_update} for {cfg.train_updates} additional updates with its optimizer state and fixed loss denominators; nested checkpointing unchanged.', flush=True)
    return original_train(*bound.args, **bound.kwargs)


def payload(*args, **kwargs):
    result = original_payload(*args, **kwargs)
    result.update(tracer_epsilon=scales.tracer_epsilon,
                  momentum_epsilon=scales.momentum_epsilon,
                  split_epsilon_runner=str(Path(__file__).resolve()),
                  continuation_source=str(Path(os.environ['SOURCE_CHECKPOINT']).resolve()))
    for item in result['train_history']:
        if item['update'] > resume_update:
            item['tracer_epsilon'] = scales.tracer_epsilon
            item['momentum_epsilon'] = scales.momentum_epsilon
    return result

m.run_train_mode = run_train
m._train_checkpoint_payload = payload
m.main()
