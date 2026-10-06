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
parser.add_argument('--fresh-training', action='store_true', help='Start a separate objective with fresh optimizer and loss normalization; optional --nn-checkpoint initializes weights')
parser.add_argument('--bound-corrections', action='store_true')
parser.add_argument('--max-correction-per-hour', type=float, nargs=4, default=[0.05, 0.05, 0.005, 0.005], metavar=('TEMP', 'SALT', 'U', 'V'))
parser.add_argument('--validate-candidates', action='store_true')
parser.add_argument('--candidate-max-retries', type=int, default=8)
parser.add_argument('--max-rollout-cfl', type=float, default=1.0)
parser.add_argument('--correction-depth-taper', type=int, nargs=2, default=None, metavar=('FULL_LEVELS', 'TRANSITION_LEVELS'))
scales, remaining = parser.parse_known_args()
if any(not math.isfinite(x) or x <= 0 for x in scales.max_correction_per_hour):
    raise ValueError('Correction bounds must be finite and positive')
if scales.candidate_max_retries < 0 or not math.isfinite(scales.max_rollout_cfl) or scales.max_rollout_cfl <= 0:
    raise ValueError('Invalid candidate validation settings')
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
active_cfg = None


def run_train(*args, **kwargs):
    global resume_update, active_cfg
    bound = inspect.signature(original_train).bind(*args, **kwargs)
    cfg = bound.arguments['args']
    active_cfg = cfg
    if cfg.train_epsilon_final is not None:
        raise ValueError('This controlled experiment uses fixed epsilon values')
    target = Path(cfg.train_output_file)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite experiment checkpoint: {target}')
    checkpoint = {}
    resume_update = 0
    if not scales.fresh_training:
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
    elif scales.total_updates is not None:
        cfg.train_updates = scales.total_updates
    saved_bounds = checkpoint.get('correction_bounds_per_hour')
    requested_bounds = scales.max_correction_per_hour if scales.bound_corrections else None
    if saved_bounds is not None and saved_bounds != requested_bounds:
        raise ValueError('Saved correction bounds differ; explicitly migrate rather than silently changing them')
    if scales.bound_corrections and cfg.conservative_mixing:
        raise ValueError('Pointwise tanh bounds do not preserve conservative_mixing; this combination is unsupported')
    if checkpoint.get('correction_depth_taper') != scales.correction_depth_taper and checkpoint:
        raise ValueError('Saved correction depth taper differs; start a fresh experiment')
    taper = 1.
    if scales.correction_depth_taper is not None:
        if cfg.conservative_mixing:
            raise ValueError('Depth taper is incompatible with conservative_mixing')
        from veros_control.training.stability import correction_depth_taper
        taper = m.jnp.asarray(correction_depth_taper(
            bound.arguments['simulation'].state.variables.zt, *scales.correction_depth_taper))[:, None]
        print(f'Correction depth taper (model level order): {taper[:, 0]}', flush=True)
    cfg.lr_backoff = checkpoint.get('lr_backoff', 1.0)
    cfg.candidate_max_retries = scales.candidate_max_retries
    if not scales.fresh_training:
        shutil.copy2(source_checkpoint, target)
        if getattr(cfg, 'train_keep_best', False):
            from veros_control.training.best_checkpoint import best_path
            source_best = best_path(source_checkpoint)
            if source_best.exists():
                shutil.copy2(source_best, best_path(target))
    cfg.epsilon = scales.tracer_epsilon
    base_model = bound.arguments['model_batched_unscaled']
    ratio = m.jnp.asarray([1., 1., scales.momentum_epsilon/scales.tracer_epsilon,
                           scales.momentum_epsilon/scales.tracer_epsilon])
    def scaled_model(params, features, coordinates, dzt):
        predictions = base_model(params, features, coordinates, dzt)
        if scales.bound_corrections:
            from veros_control.training.stability import bound_corrections
            epsilon = m.jnp.asarray([scales.tracer_epsilon, scales.tracer_epsilon,
                                    scales.momentum_epsilon, scales.momentum_epsilon])
            # nn_epsilon_step multiplies by tracer epsilon afterward.
            applied = bound_corrections(predictions, epsilon, scales.max_correction_per_hour) / scales.tracer_epsilon
        else:
            applied = m.jnp.where(ratio != 0, predictions * ratio, 0.)
        return m.jnp.where(taper > 0, applied * taper, 0.)
    bound.arguments['model_batched_unscaled'] = scaled_model
    if scales.validate_candidates:
        from veros_control.training.stability import make_rollout_validator
        simulation = bound.arguments['simulation']
        steps = m._resolve_train_n_steps(cfg, bound.arguments['steps_list'])
        simulation.prepare_ERA5_forcing_window(simulation.state, steps)
        cfg.candidate_validator = make_rollout_validator(
            simulation, scaled_model, bound.arguments['nn_stats'], scales.tracer_epsilon,
            m.make_chunk_sizes(steps, base=cfg.chunk_base), m.nn_epsilon_step,
            m._temporary_solver_cache, max_cfl=scales.max_rollout_cfl,
        )
    print(f'Correction bounds per hour: {requested_bounds}; candidate validation: {scales.validate_candidates}', flush=True)
    print(f'SPLIT EPSILON: tracer={scales.tracer_epsilon:g}, momentum={scales.momentum_epsilon:g}', flush=True)
    initialization = 'fresh optimizer and newly calibrated loss denominators' if scales.fresh_training else 'saved optimizer and fixed loss denominators'
    print(f'Starting at update {resume_update} for {cfg.train_updates} additional updates with {initialization}; nested checkpointing unchanged.', flush=True)
    return original_train(*bound.args, **bound.kwargs)


def payload(*args, **kwargs):
    result = original_payload(*args, **kwargs)
    result.update(tracer_epsilon=scales.tracer_epsilon,
                  momentum_epsilon=scales.momentum_epsilon,
                  split_epsilon_runner=str(Path(__file__).resolve()),
                  continuation_source=None if scales.fresh_training else str(Path(os.environ['SOURCE_CHECKPOINT']).resolve()))
    result.update(correction_depth_taper=scales.correction_depth_taper,
                  correction_bounds_per_hour=scales.max_correction_per_hour if scales.bound_corrections else None,
                  validate_candidates=scales.validate_candidates, max_rollout_cfl=scales.max_rollout_cfl,
                  lr_backoff=getattr(active_cfg, 'lr_backoff', 1.0),
                  candidate_max_retries=scales.candidate_max_retries)
    for item in result['train_history']:
        if item['update'] > resume_update:
            item['tracer_epsilon'] = scales.tracer_epsilon
            item['momentum_epsilon'] = scales.momentum_epsilon
    return result

m.run_train_mode = run_train
m._train_checkpoint_payload = payload
m.main()
