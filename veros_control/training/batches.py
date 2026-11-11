"""Monthly batches average independent rollout gradients before one Adam update."""
from dataclasses import replace
import jax
import numpy as np
from veros_control.core.rollout import Rollout, solver_cache_scope
from veros_control.data.dataset import create_simulation
from veros_control.training.loss import build_loss
from veros_control.training.stability import make_validator


def batch_months(months, size, seed, update):
    """Same deterministic shuffled passes as main; resumable from update number."""
    if not 1 <= size <= len(months) or update < 0:
        raise ValueError('Invalid monthly batch size or update')
    batches = (len(months) + size - 1) // size
    epoch, position = divmod(update, batches)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(months)
    return tuple(map(int, order[position * size:(position + 1) * size]))


def case_configs(config):
    if not config.initial_states:
        return {config.start_month: config}
    from veros_control.data.reference import monthly_clock
    from veros import time
    cases = {}
    for case in config.initial_states:
        shift, interval = case.forcing_time_shift, case.reference_hours*3600.
        if case.initial_date:
            from datetime import datetime
            from veros_control.data.dated import model_timestamp
            from veros_control.data.reference import snapshot_date
            start = datetime.fromisoformat(case.initial_date)
            interval = model_timestamp(snapshot_date(case.target))-model_timestamp(start)
        elif shift is None:
            shift, interval = monthly_clock(case.initial, case.target, time.convert_time(1., 'years', 'seconds'))
        cases[case.month] = replace(config, initial=case.initial, target=case.target,
            start_month=start.month if case.initial_date else case.month, initial_date=case.initial_date,
            reference_hours=interval/3600., forcing_time_shift=shift)
    return cases


class BatchObjective:
    """Reuse the setup; optionally keep one dynamic objective across monthly cases."""
    def __init__(self, config, simulation, model, stats):
        self.config, self.simulation = config, simulation
        self.model, self.stats = model, stats
        self.cases = case_configs(config)
        self.months = tuple(self.cases)
        self.panel = tuple(config.checkpoint_months) or self.months
        self.active = None
        self.metadata = {}

    def member(self, month):
        if self.active != month:
            if self.active is not None and not self.config.reuse_rollouts:
                del self.evaluate_one, self.gradient_one, self.validate_one, self.rollout
                jax.clear_caches()
            cfg = self.cases[month]
            sim, reference = create_simulation(cfg, self.simulation)
            sim.set_forcing(sim.state)
            self.initial = sim.state.copy()
            self.reference = (reference.initial, reference.target,
                              np.asarray(reference.initial_time), np.asarray(reference.duration_seconds))
            self.forcing = sim.training_forcing() if hasattr(sim, 'training_forcing') else None
            names, terms, metadata = build_loss(sim.state, reference, cfg.loss, cfg.n_steps,
                                                dynamic=self.config.reuse_rollouts)
            self.metadata[str(month)] = metadata
            if self.active is None or not self.config.reuse_rollouts:
                self.rollout = Rollout(sim, self.model, self.stats, cfg.normalization, cfg.model.use_forcings)
                self.names = names
                factory = self.rollout.reusable_objective if self.config.reuse_rollouts else self.rollout.objective
                self.evaluate_one, self.gradient_one = factory(terms, cfg.n_steps, cfg.sample_steps, cfg.chunk_base)
                self.validate_one = make_validator(self.rollout, cfg.n_steps, cfg.max_cfl) if cfg.validate else lambda *a:(True, {})
            self.active = month
        return self

    def mean(self, months, params, scale, gradient=False):
        total = None
        for month in months:
            self.member(month)
            with solver_cache_scope():
                value = (self.gradient_one if gradient else self.evaluate_one)(params, scale, *self.dynamic_inputs())
                jax.block_until_ready(value)
            total = value if total is None else jax.tree_util.tree_map(lambda a,b:a+b, total, value)
        return jax.tree_util.tree_map(lambda x:x/len(months), total)

    def validate(self, months, params):
        reports = {}
        for month in months:
            self.member(month)
            valid, report = self.validate_one(params, self.initial, self.forcing) if self.config.reuse_rollouts else self.validate_one(params)
            reports[str(month)] = report
            if not valid:
                return False, reports
        return True, reports

    def dynamic_inputs(self):
        return (self.initial, self.reference, self.forcing) if self.config.reuse_rollouts else ()

    def close(self):
        pass
