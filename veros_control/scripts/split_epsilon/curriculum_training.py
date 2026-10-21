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

parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
parser.add_argument('--monthly-initial-states', default=None, metavar='ASSET_DIR')
parser.add_argument('--initial-state-batch-size', type=int, default=3)
parser.add_argument('--monthly-parallel-workers', type=int, default=0)
parser.add_argument('--monthly-worker-nodes', action='store_true')
parser.add_argument('--monthly-fused-validation', action='store_true')
parser.add_argument('--monthly-baseline-normalization', action='store_true', help='Normalize each term by its no-NN mean over all 12 months at this horizon')
parser.add_argument('--checkpoint-score-every', type=int, default=1)
parser.add_argument('--initial-state-seed', type=int, default=0)
parser.add_argument('--checkpoint-months', type=int, nargs='+', default=[1, 5, 9])
parser.add_argument('--cyclic-glorys-targets', action='store_true')
parser.add_argument('--era5-forcing-file', default=None)
parser.add_argument('--curriculum-source', default=None)
parser.add_argument('--resume-stage', action='store_true')
parser.add_argument('--depth-loss-stats', default=None)
parser.add_argument('--filtered-loss', action='store_true')
parser.add_argument('--loss-backtracking', action='store_true')
parser.add_argument('--train-min-lr', type=float, default=None)
parser.add_argument('--lr-decrease-start-epoch', type=int, default=None)
parser.add_argument('--restart-stage-lr', action='store_true', help='Reset LR and epoch history while preserving source weights and Adam state')
parser.add_argument('--require-normalized-inputs', action='store_true')
parser.add_argument('--curriculum-denominators', required=True)
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
if scales.restart_stage_lr and (scales.resume_stage or not scales.fresh_training or not scales.curriculum_source):
    raise ValueError('--restart-stage-lr requires a fresh stage and curriculum source')
if scales.resume_stage:
    if scales.fresh_training or not scales.curriculum_source:
        raise ValueError('--resume-stage requires --curriculum-source and forbids --fresh-training')
    os.environ['SOURCE_CHECKPOINT'] = str(Path(scales.curriculum_source).resolve())
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
    if scales.require_normalized_inputs:
        if not getattr(cfg, 'normalize_inputs', False) or bound.arguments['nn_stats'] is None or cfg.scaling_method_features != 'meanstd':
            raise ValueError('This experiment requires GLORYS meanstd input normalization')
        print('Verified GLORYS input normalization: enabled, meanstd', flush=True)
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
    if scales.resume_stage:
        cfg.lr_clean_updates = checkpoint.get('lr_clean_updates', 0)
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
    if scales.monthly_initial_states:
        from veros_control.training.monthly_batches import MonthlyRollouts
        cfg.train_batch_factory = lambda single: MonthlyRollouts(
            m, single, cfg, monthly_pairs, scales.initial_state_batch_size,
            scales.initial_state_seed, scales.checkpoint_months, scales.max_rollout_cfl)
    if scales.validate_candidates and not scales.monthly_initial_states:
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
    initialization = 'fresh stage with fixed loss denominators' if scales.fresh_training else 'saved optimizer and fixed loss denominators'
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
# A new objective at each horizon, with continuous optimizer state and fixed scales.
import json
depth_loss = json.loads(Path(scales.depth_loss_stats).read_text()) if scales.depth_loss_stats else None
original_parse = m.parse_args

def parse_args():
    cfg = original_parse()
    if scales.resume_stage and (cfg.train_overwrite or cfg.nn_checkpoint is None or Path(cfg.nn_checkpoint).resolve() != Path(scales.curriculum_source).resolve()):
        raise ValueError('Resume requires the same NN checkpoint and forbids overwrite')
    if stage_source and stage_source.get('monthly_initial_states') is not None and stage_source['monthly_initial_states'] != monthly_config:
        raise ValueError('Monthly curriculum continuation must preserve dataset and sampling settings')
    cfg.monthly_baseline_normalization = scales.monthly_baseline_normalization
    if scales.monthly_baseline_normalization and (not monthly_config or not cfg.surface_only or not scales.monthly_fused_validation):
        raise ValueError('Monthly baseline normalization requires surface monthly fused training')
    if scales.resume_stage and bool(stage_source.get('monthly_baseline_normalization', False)) != scales.monthly_baseline_normalization:
        raise ValueError('Resume must preserve the baseline normalization objective')
    cfg.monthly_initial_states = monthly_config
    cfg.monthly_parallel_workers = scales.monthly_parallel_workers
    cfg.monthly_worker_nodes = scales.monthly_worker_nodes
    cfg.monthly_fused_validation = scales.monthly_fused_validation
    cfg.checkpoint_score_every = scales.checkpoint_score_every
    cfg.monthly_momentum_epsilon = scales.momentum_epsilon
    cfg.monthly_correction_bounds = scales.max_correction_per_hour if scales.bound_corrections else None
    if cfg.monthly_parallel_workers < 0 or cfg.checkpoint_score_every < 1:
        raise ValueError('Invalid worker count or checkpoint scoring frequency')
    if scales.monthly_fused_validation and (not monthly_config or cfg.loss_reduction != 'mean' or not cfg.balance_loss_terms or tuple(cfg.keys_loss) != ('temp', 'salt')):
        raise ValueError('Fused monthly validation requires mean, balanced temperature/salinity loss')
    if scales.monthly_parallel_workers and (not monthly_config or scales.correction_depth_taper is not None):
        raise ValueError('Parallel monthly workers require monthly inputs without a depth taper')
    if monthly_config:
        cfg.glorys_file, cfg.glorys_target_file = monthly_pairs[1]
        cfg.glorys_index = cfg.glorys_target_index = 0
        cfg.glorys_month_index = 0
        cfg.glorys_start_month = 1
        if cfg.restart_file:
            raise ValueError('Monthly GLORYS batches cannot use a restart file')
        if not (cfg.basin_mean_loss and cfg.glorys_sampled_interpolation and scales.validate_candidates):
            raise ValueError('Monthly batches require sampled basin interpolation and candidate validation')
        if cfg.warmup != 0:
            raise ValueError('Monthly batches require zero warmup')
        if scales.resume_stage and stage_source.get('monthly_initial_states') != monthly_config:
            raise ValueError('Resume must preserve monthly dataset, batches, forcing and reference months')
    if cfg.surface_only and (not cfg.basin_mean_loss or cfg.basin_levels != 1):
        raise ValueError("Surface-only curriculum requires --basin-mean-loss --basin-levels 1")
    if stage_source is not None and bool(stage_source.get("surface_only", False)) != cfg.surface_only:
        raise ValueError("Curriculum must preserve surface-only architecture")
    cfg.train_min_lr = scales.train_min_lr if scales.train_min_lr is not None else (1e-5 if cfg.surface_only else 0.)
    cfg.lr_decrease_start_epoch = scales.lr_decrease_start_epoch if scales.lr_decrease_start_epoch is not None else (20 if cfg.surface_only else 1)
    if not math.isfinite(cfg.train_min_lr) or not 0 <= cfg.train_min_lr <= cfg.train_lr or cfg.lr_decrease_start_epoch < 1:
        raise ValueError('Invalid learning-rate floor or decrease start epoch')
    cfg.depth_loss = depth_loss
    cfg.spatial_loss_filter = depth_loss.get('spatial_filter') if depth_loss else None
    if bool(cfg.spatial_loss_filter) != scales.filtered_loss:
        raise ValueError('--filtered-loss must match calibrated depth statistics')
    if scales.filtered_loss:
        if not cfg.glorys_sampled_interpolation or cfg.sample_steps != depth_loss['sample_steps']:
            raise ValueError('Filtered calibration requires matching sampled interpolation')
        print(f'Filtered loss enabled: {cfg.spatial_loss_filter}', flush=True)
    cfg.loss_backtracking = scales.loss_backtracking
    cfg.lr_clean_updates = 0
    if scales.loss_backtracking and (not scales.validate_candidates or not cfg.train_keep_best or cfg.train_lr_final is not None):
        raise ValueError('Loss backtracking requires validation, best/latest saves and constant maximum LR')
    if depth_loss is not None and (not cfg.basin_mean_loss or cfg.basin_pattern_lambda != 1.):
        raise ValueError('Depth calibration requires basin loss with lambda=1')
    return cfg

m.parse_args = parse_args
original_init = m._load_or_init_train_state
stage_source = m.load_checkpoint(scales.curriculum_source) if scales.curriculum_source else None
fixed_denominators = json.loads(Path(scales.curriculum_denominators).read_text())
if depth_loss is not None and any(value != 1. for value in fixed_denominators.values()):
    raise ValueError('Depth normalization replaces scalar normalization')
if stage_source is not None and stage_source.get('depth_loss') != depth_loss:
    raise ValueError('Curriculum must preserve depth normalization')
if not scales.fresh_training and not scales.resume_stage:
    raise ValueError('Curriculum stages require --fresh-training and a new output directory')

def stage_init(cfg, checkpoint_path, params, optimizer):
    if Path(checkpoint_path).exists() and not scales.resume_stage:
        raise FileExistsError(checkpoint_path)
    params, opt_state, history, start = original_init(cfg, checkpoint_path, params, optimizer)
    denominators = stage_source['loss_term_denominators'] if scales.resume_stage and scales.monthly_baseline_normalization else fixed_denominators
    cfg.loss_term_denominators = m._validate_loss_denominators(denominators, cfg.keys_loss)
    if scales.resume_stage and scales.monthly_baseline_normalization:
        cfg.monthly_baseline_calibration = stage_source['monthly_baseline_calibration']
    if depth_loss is not None:
        if cfg.surface_only:
            surface_index = int(m.np.argmin(depth_loss['depths_m']))
            surface_scales = {key: values[surface_index] for key, values in depth_loss['denominators'].items()}
            print(f"Surface loss normalization: depth={depth_loss['depths_m'][surface_index]:.3f}m; denominators={surface_scales}; outer denominators={cfg.loss_term_denominators}", flush=True)
        else:
            print(f"Depth normalization enabled; band boundaries (m): {depth_loss.get('band_boundaries_m')}; outer scalar denominators are unity", flush=True)
    if stage_source is not None:
        for key, expected in dict(residual_blocks=cfg.residual_blocks, hidden_dim_nn=cfg.hidden_dim,
                depth_nn=cfg.depth, sample_steps=cfg.sample_steps, basin_levels=cfg.basin_levels,
                basin_pattern_lambda=cfg.basin_pattern_lambda,
                glorys_sampled_interpolation=cfg.glorys_sampled_interpolation,
                tracer_epsilon=scales.tracer_epsilon, momentum_epsilon=scales.momentum_epsilon,
                correction_bounds_per_hour=scales.max_correction_per_hour).items():
            if stage_source.get(key) != expected:
                raise ValueError(f'Curriculum checkpoint differs in {key}')
        if bool(stage_source.get('loss_backtracking', False)) != scales.loss_backtracking:
            raise ValueError('Curriculum must preserve loss backtracking mode')
        if stage_source.get('loss_term_denominators') != cfg.loss_term_denominators and not scales.monthly_baseline_normalization:
            raise ValueError('Curriculum must preserve loss denominators')
        if scales.resume_stage:
            if start != stage_source['update'] or cfg.train_n_steps != stage_source['n_steps']:
                raise ValueError('Resume must preserve stage horizon and update numbering')
            expected = dict(rtol=1e-8, atol=1e-10, recovery_factor=1.5, recovery_interval=5, max_lr=cfg.train_lr)
            if cfg.surface_only or cfg.train_min_lr or cfg.lr_decrease_start_epoch != 1:
                expected.update(min_lr=cfg.train_min_lr, decrease_start_epoch=cfg.lr_decrease_start_epoch)
            if stage_source.get('loss_backtracking_settings') != expected:
                raise ValueError('Resume must preserve LR recovery settings')
            cfg.lr_clean_updates = stage_source.get('lr_clean_updates', 0)
        params, opt_state = stage_source['params'], stage_source['opt_state']
        cfg.lr_backoff = 1. if scales.restart_stage_lr else max(stage_source.get('lr_backoff', 1.), cfg.train_min_lr / cfg.train_lr)
        if scales.restart_stage_lr:
            cfg.lr_clean_updates = 0
            print('Restarting stage at epoch 1 with source weights/Adam and reset learning rate', flush=True)
        if opt_state is None:
            raise ValueError('Missing curriculum optimizer state')
    print(f'Curriculum stage: {cfg.train_n_steps} steps; optimizer carried={stage_source is not None}; fixed denominators={cfg.loss_term_denominators}', flush=True)
    return (params, opt_state, history, start) if scales.resume_stage else (params, opt_state, [], 0)

previous_payload = m._train_checkpoint_payload

def stage_payload(*args, **kwargs):
    result = previous_payload(*args, **kwargs)
    result['loss_backtracking'] = scales.loss_backtracking
    result['lr_clean_updates'] = getattr(active_cfg, 'lr_clean_updates', 0)
    result['loss_backtracking_settings'] = dict(rtol=1e-8, atol=1e-10, recovery_factor=1.5, recovery_interval=5, max_lr=active_cfg.train_lr) if scales.loss_backtracking else None
    if result['loss_backtracking_settings'] is not None and (active_cfg.surface_only or active_cfg.train_min_lr or active_cfg.lr_decrease_start_epoch != 1):
        result['loss_backtracking_settings'].update(min_lr=active_cfg.train_min_lr, decrease_start_epoch=active_cfg.lr_decrease_start_epoch)
    result['monthly_execution'] = dict(workers=scales.monthly_parallel_workers, worker_nodes=scales.monthly_worker_nodes, fused_validation=scales.monthly_fused_validation, checkpoint_score_every=scales.checkpoint_score_every)
    result['monthly_baseline_normalization'] = scales.monthly_baseline_normalization
    result['monthly_baseline_calibration'] = getattr(active_cfg, 'monthly_baseline_calibration', None)
    result['restart_stage_lr'] = scales.restart_stage_lr
    result['spatial_loss_filter'] = getattr(active_cfg, 'spatial_loss_filter', None)
    result['depth_loss'] = depth_loss
    result['curriculum_source'] = scales.curriculum_source
    offset = stage_source.get('curriculum_total_update', stage_source['update']) if stage_source else 0
    if scales.resume_stage:
        offset -= stage_source['update']
    result['curriculum_total_update'] = offset + result['update']
    return result

m._load_or_init_train_state = stage_init
m._train_checkpoint_payload = stage_payload
monthly_pairs = monthly_config = None
if scales.monthly_initial_states:
    from veros_control.training.monthly_batches import monthly_files, batch_months
    monthly_pairs = monthly_files(scales.monthly_initial_states, scales.cyclic_glorys_targets)
    batch_months(tuple(monthly_pairs), scales.initial_state_batch_size, scales.initial_state_seed, 0)
    if not set(scales.checkpoint_months) <= set(monthly_pairs) or len(set(scales.checkpoint_months)) != len(scales.checkpoint_months):
        raise ValueError('Invalid fixed checkpoint months')
    forcing = Path(scales.era5_forcing_file or Path(scales.monthly_initial_states) / 'ERA5_1deg_1993_Flux_processed.nc').resolve()
    if not forcing.is_file():
        raise FileNotFoundError(forcing)
    monthly_config = dict(files=monthly_pairs, batch_size=scales.initial_state_batch_size,
        seed=scales.initial_state_seed, checkpoint_months=scales.checkpoint_months,
        forcing_file=str(forcing), cyclic_glorys_targets=scales.cyclic_glorys_targets,
        calendar='GLORYS timestamps scaled to the Veros model year')
    original_simulation_init = m.init_simulation
    def monthly_simulation_init(*args, **kwargs):
        kwargs.update(glorys_file=monthly_pairs[1][0], glorys_month_index=0, glorys_start_month=1,
            era5_forcing_file=str(forcing), setup_forcing_window=True)
        simulation = original_simulation_init(*args, **kwargs)
        simulation.era5_forcing_file = str(forcing)
        return simulation
    m.init_simulation = monthly_simulation_init
m.main()
