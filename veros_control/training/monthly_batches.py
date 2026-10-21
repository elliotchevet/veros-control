"""Sequential monthly rollouts: one optimizer update from a mean batch gradient."""
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def monthly_files(directory, cyclic=False):
    files = {}
    for path in sorted(Path(directory).glob('*4*0_1993*.nc')):
        match = re.search(r'_1993(\d{2})\.nc$', path.name)
        if match and 1 <= int(match[1]) <= 12:
            month = int(match[1])
            if month in files:
                raise ValueError(f'Ambiguous GLORYS files for month {month}')
            files[month] = str(path.resolve())
    if set(files) != set(range(1, 13)):
        raise ValueError('Expected exactly twelve 40-level monthly GLORYS files for 1993')
    return {month: (files[month], files[month % 12 + 1])
            for month in range(1, 13 if cyclic else 12)}


def batch_months(months, size, seed, update):
    """Deterministic shuffled passes; resume depends only on the update number."""
    if not 1 <= size <= len(months) or update < 0:
        raise ValueError('Invalid monthly batch size or update')
    # Keep batches within a shuffled pass (the final batch can be smaller).
    batches = (len(months) + size - 1) // size
    epoch, position = divmod(update, batches)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(months)
    return tuple(map(int, order[position * size:(position + 1) * size]))


def snapshot_date(path):
    import h5netcdf
    with h5netcdf.File(path, 'r') as f:
        v = f.variables['time']
        units = v.attrs['units']
        unit, origin = units.split(' since ')
        if v.attrs.get('calendar', 'standard') not in ('standard', 'gregorian', 'proleptic_gregorian'):
            raise ValueError('Expected Gregorian GLORYS timestamps')
        factor = {'seconds': 1., 'hours': 3600., 'days': 86400.}[unit]
        return datetime.fromisoformat(origin) + timedelta(seconds=float(v[0]) * factor)


def monthly_clock(initial, target, model_year_seconds, cyclic=False):
    """Use the files' timestamps, including the monthly-mean midmonth offset."""
    start, end = snapshot_date(initial), snapshot_date(target)
    if end <= start and cyclic:
        end = end.replace(year=end.year + 1)
    if start.year != 1993 or end <= start:
        raise ValueError('Invalid monthly target dates')
    scale = model_year_seconds / (365. * 86400.)
    return ((start - datetime(1993, 1, 1)).total_seconds() * scale,
            (end - start).total_seconds() * scale)


class MonthlyRollouts:
    def __init__(self, module, single, cfg, files, batch_size, seed, score_months, max_cfl):
        self.m, self.single, self.cfg = module, single, cfg
        self.simulation = single.simulation
        self.base_state = self.simulation.state.copy()
        self.files, self.batch_size, self.seed = files, batch_size, seed
        self.score_months, self.max_cfl = tuple(score_months), max_cfl
        self.steps = cfg.train_n_steps
        if not self.score_months or not set(self.score_months) <= set(files):
            raise ValueError('Checkpoint months must belong to the available initial states')
        if getattr(cfg, 'loss_term_denominators', None) is not None:
            self.single.loss_term_denominators = module.jnp.asarray([cfg.loss_term_denominators[k] for k in ('temp', 'salt')])
        self.select(0)
        self.call_options = None
        self.cached_candidate = None
        self.workers = None
        if getattr(cfg, 'monthly_parallel_workers', 0):
            from veros_control.training.monthly_parallel import MonthlyWorkers
            self.workers = MonthlyWorkers(cfg, files, batch_size, seed, score_months, max_cfl)
        self.rollout_counts = dict(gradient=0, candidate=0, panel=0)

    @property
    def loss_term_denominators(self):
        return self.single.loss_term_denominators

    @loss_term_denominators.setter
    def loss_term_denominators(self, value):
        self.single.loss_term_denominators = value

    def select(self, update):
        self.months = batch_months(tuple(self.files), self.batch_size, self.seed, update)
        self.cached_candidate = None
        print(f'Monthly training batch at update {update+1}: {self.months}', flush=True)

    @contextmanager
    def member(self, month):
        from veros import time
        from veros_control.training.train import load_glorys_state
        sim = self.simulation
        original_state = sim.state
        window = getattr(sim, '_era5_forcing_window', None)
        cache = getattr(sim, '_era5_forcing_cache', {})
        initial, target = self.files[month]
        try:
            sim.state = self.base_state
            sim.state = load_glorys_state(sim, initial, config=SimpleNamespace(
                glorys_month_lengths=MONTH_DAYS, glorys_start_month=month))
            shift, interval = monthly_clock(initial, target, time.convert_time(1., 'years', 'seconds'),
                self.cfg.monthly_initial_states['cyclic_glorys_targets'])
            if self.steps * sim.state.settings.dt_tracer > interval:
                raise ValueError('Monthly interpolation supports rollouts only up to the next monthly target')
            with sim.state.variables.unlock():
                sim.state.variables.forcing_time_shift = self.m.jnp.asarray(shift)
            # Division and floor-division can disagree at exact floating-point record boundaries.
            sim.prepare_ERA5_forcing_window(sim.state, self.steps, leading_margin=1)
            cfg = SimpleNamespace(**vars(self.cfg))
            cfg.glorys_reference_seconds = interval
            cfg.glorys_interpolation_alpha = self.steps * sim.state.settings.dt_tracer / interval
            loss, names, terms = self.m.make_root_loss_functions(cfg, sim, target, initial)
            ad = self.m.NNParamsTrainChunkedVJP(sim, self.single.model_batched_unscaled,
                self.single.stats, loss, freeze_vars=cfg.freeze_vars,
                loss_reduction=cfg.loss_reduction, loss_terms_function=terms, loss_term_count=len(names))
            ad.loss_term_denominators = self.loss_term_denominators
            yield ad
        finally:
            sim.state = original_state
            sim._era5_forcing_window, sim._era5_forcing_cache = window, cache

    def _mean(self, months, params, options):
        started = time.monotonic()
        kind = 'panel' if options.get('forward_only') else 'gradient'
        if hasattr(self, 'rollout_counts'):
            self.rollout_counts[kind] += len(months)
        if getattr(self, 'workers', None) is not None:
            values = self.workers.map('g', months, params, options, self.loss_term_denominators)
            result = self.m.jax.tree_util.tree_map(lambda *xs: self.m.jnp.asarray(np.mean(xs, axis=0)), *values)
            print(f'BATCH {kind}: months={months} rollouts={len(months)} wall={time.monotonic()-started:.2f}s', flush=True)
            return result
        total = None
        for month in months:
            with self.member(month) as ad:
                value = ad.g(self.simulation.state, params, **options)
                self.m.jax.block_until_ready(value)
                total = value if total is None else self.m.jax.tree_util.tree_map(
                    lambda a, b: a + b, total, value)
                del value
        return self.m.jax.tree_util.tree_map(lambda x: x / len(months), total)

    def g(self, state, params, **options):
        self.call_options = dict(options)
        cached = getattr(self, 'cached_candidate', None)
        if options.get('forward_only') and cached is not None and cached[0] is params:
            return cached[1]
        return self._mean(self.months, params, options)

    def calibrate_no_nn(self, params, **options):
        """One forward rollout per month; preserve training params and Adam state."""
        import json
        months = tuple(sorted(self.files))
        if months != tuple(range(1, 13)):
            raise ValueError('Baseline normalization requires all twelve months')
        zero_params = self.m.jax.tree_util.tree_map(self.m.jnp.zeros_like, params)
        options = dict(options, forward_only=True, return_aux=True)
        started = time.monotonic()
        print(f'BASELINE calibration: horizon={self.steps} no-NN months={months}; 12 forward rollouts', flush=True)
        if self.workers is not None:
            values = self.workers.map('g', months, zero_params, options, self.loss_term_denominators)
        else:
            values = []
            for month in months:
                with self.member(month) as ad:
                    values.append(self.m.jax.device_get(ad.g(self.simulation.state, zero_params, **options)))
        terms = np.asarray([value[1] for value in values], dtype=float)
        if terms.shape != (12, 2) or not np.all(np.isfinite(terms)) or np.any(terms <= 0):
            raise ValueError(f'Invalid no-NN baseline terms: {terms}')
        denominators = dict(zip(('temp', 'salt'), map(float, terms.mean(axis=0))))
        calibration = dict(horizon=self.steps, months=list(months), no_nn=True,
            term_units='original filtered surface loss terms before outer normalization',
            denominators=denominators, sample_steps=self.cfg.sample_steps,
            spatial_loss_filter=getattr(self.cfg, 'spatial_loss_filter', None),
            per_month={str(month): dict(zip(('temp', 'salt'), map(float, row))) for month, row in zip(months, terms)},
            wall_seconds=time.monotonic()-started)
        path = Path(self.cfg.train_output_file).parent/'no_nn_baseline.json'
        temporary = path.with_suffix('.pending')
        temporary.write_text(json.dumps(calibration, indent=2)+'\n')
        temporary.replace(path)
        self.cfg.loss_term_denominators = denominators
        self.cfg.monthly_baseline_calibration = calibration
        self.loss_term_denominators = self.m.jnp.asarray(list(denominators.values()))
        self.cached_candidate = None
        self.rollout_counts['calibration'] = 12
        print(f'BASELINE fixed denominators: {denominators}; no-NN mean relative terms: temp=1 salt=1; '
              f'wall={calibration["wall_seconds"]:.2f}s; saved={path}', flush=True)
        return calibration

    def checkpoint_score(self, params):
        options = dict(self.call_options, forward_only=True, return_aux=True)
        return self._mean(self.score_months, params, options)

    def validate(self, params):
        from veros_control.training.stability import make_rollout_validator
        started = time.monotonic()
        self.rollout_counts['candidate'] += len(self.months)
        reports = {}
        if self.workers is not None:
            results = self.workers.map('validate', self.months, params, {}, self.loss_term_denominators)
        else:
            results = [self.validate_member(month, params) for month in self.months]
        for month, (valid, report) in zip(self.months, results):
            reports[str(month)] = report
            if not valid:
                return False, {'months': reports}
        if getattr(self.cfg, 'monthly_fused_validation', False):
            terms = self.m.jnp.asarray(np.mean([r['loss_terms'] for _, r in results], axis=0))
            self.cached_candidate = (params, (self.m.jnp.mean(terms / self.loss_term_denominators), terms))
        print(f'BATCH candidate: months={self.months} rollouts={len(self.months)} wall={time.monotonic()-started:.2f}s', flush=True)
        return True, {'months': reports}

    def validate_member(self, month, params):
        from veros_control.training.stability import make_rollout_validator
        for month in (month,):
            with self.member(month) as ad:
                validate = make_rollout_validator(self.simulation, ad.model_batched_unscaled,
                    ad.stats, self.cfg.epsilon, self.m.make_chunk_sizes(self.steps, base=self.cfg.chunk_base),
                    self.m.nn_epsilon_step, self.m._temporary_solver_cache, max_cfl=self.max_cfl,
                    score_terms=ad.loss_terms_function if getattr(self.cfg, 'monthly_fused_validation', False) else None,
                    sample_steps=self.cfg.sample_steps, score_denominators=self.loss_term_denominators)
                valid, report = validate(params)
                return valid, report
