"""Continuous paired training diagnostics with hourly checks and monthly archives.

These are train_diags: their initial state comes from the training set. No weights
are updated. A 30-year run uses 259200 hourly steps on the 360-day model calendar.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import jax
import jax.numpy as jnp
import numpy as np
from veros_control.core.checkpoint import load_checkpoint, model_from_checkpoint
from veros_control.core.correction import physical_model
from veros_control.core.rollout import Rollout, solver_cache_scope
from veros_control.data.dataset import create_simulation
from veros_control.evaluation.config import EvaluationConfig
from veros_control.training.config import load_config
from veros_control.training.stability import state_metrics
from veros_control.diagnostics.monthly import MonthlyOutput, sample


@dataclass
class PairedConfig(EvaluationConfig):
    years: int = 30
    months: int | None = None  # short smoke runs; otherwise years * 12
    start_year: int = 1993
    stop_on_invalid: bool = True
    forcing_note: str = 'Configured forcing repeated continuously; no annual state reset'
    diagnostic_kind: str = 'train_diags'

    def validate_options(self):
        super().validate_options()
        if self.years < 1 or (self.months is not None and self.months < 1) or self.start_month != 1:
            raise ValueError('Monthly archive requires positive years and a January initialization')
        if self.diagnostic_kind != 'train_diags':
            raise ValueError('This runner uses a training initial state and is classified as train_diags')


def write_json(path, values):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(values, indent=2)+'\n')
    temporary.replace(path)


def run(config):
    config.validate_options()
    output = Path(config.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty paired-rollout output directory')
    output.mkdir(parents=True, exist_ok=True)
    n_steps = (config.months if config.months is not None else config.years*12)*720
    source = load_checkpoint(config.checkpoint)
    _, config.model, config.correction, stats, config.normalization = model_from_checkpoint(source, 1)
    # Only initialization needs a reference pair. No extrapolated target is used in the long run.
    initialization = replace(config, n_steps=1)
    sim, _ = create_simulation(initialization)
    if float(sim.state.settings.dt_tracer) != 3600.:
        raise ValueError('Monthly averaging currently requires hourly model timesteps')
    if hasattr(sim, 'prepare_ERA5_forcing_window'):
        sim.prepare_ERA5_forcing_window(sim.state, n_steps)
    model, _, _, _, _ = model_from_checkpoint(source, sim.state.settings.nz)
    physical = physical_model(model, config.correction, sim.state.variables.zt)
    fingerprint = hashlib.sha256(Path(config.checkpoint).read_bytes()).hexdigest()
    shutil.copyfile(config.checkpoint, output/'checkpoint.pkl')
    write_json(output/'config.json', asdict(config))
    write_json(output/'metadata.json', dict(diagnostic_kind='train_diags', sha256=fingerprint,
        checkpoint_update=source['update'], forcing_note=config.forcing_note, calendar='360_day'))
    summary = {}
    cases = [('baseline', lambda p,x,c,d:jnp.zeros((*x.shape[:2],4), dtype=x.dtype)), ('trained', physical)]
    for label, apply in cases:
        rollout = Rollout(sim, apply, stats, config.normalization, config.model.use_forcings)
        state = rollout.initial
        writer = MonthlyOutput(output/f'{label}.averages.nc', state,
            dict(trajectory=label, checkpoint_sha256=fingerprint, forcing_note=config.forcing_note,
                 diagnostic_kind='train_diags'), config.start_year)
        sums = jax.tree_util.tree_map(jnp.zeros_like, sample(state))

        def step(carry, _):
            current, accumulated = carry
            current = rollout._step(current, source['params'])
            accumulated = jax.tree_util.tree_map(lambda a,b:a+b, accumulated, sample(current))
            return (current, accumulated), state_metrics(current)

        advance = jax.jit(lambda carry:jax.lax.scan(step, carry, None, length=24))
        report = dict(status='running', checked_steps=0, max_cfl=0.)
        summary[label] = report
        stability_file = output/f'{label}_stability.csv'
        columns = ['step', 'nonfinite_count', 'min_salinity', 'max_cfl', 'min_temperature', 'max_temperature']
        with stability_file.open('w', newline='') as stream:
            csv.writer(stream).writerow(columns)
        for offset in range(0, n_steps, 24):
            try:
                with solver_cache_scope():
                    (state, sums), metrics = advance((state, sums))
                    metrics = np.asarray(jax.device_get(metrics))
            except Exception as exc:
                report.update(status='runtime_exception', error=str(exc))
                write_json(output/'summary.json', summary)
                raise
            daily = [offset+24, float(np.max(metrics[:,0])), float(np.min(metrics[:,1])),
                     float(np.max(metrics[:,2])), float(np.min(metrics[:,3])), float(np.max(metrics[:,4]))]
            with stability_file.open('a', newline='') as stream:
                csv.writer(stream).writerow(daily)
            invalid = (~np.isfinite(metrics).all(axis=1)) | (metrics[:,0] != 0) | (metrics[:,1] <= 0) | (metrics[:,2] >= config.max_cfl)
            nonfinite = (~np.isfinite(metrics).all(axis=1)) | (metrics[:,0] != 0)
            if invalid.any() and 'first_invalid_step' not in report:
                report['first_invalid_step'] = offset+int(np.flatnonzero(invalid)[0])+1
            if nonfinite.any() and 'first_nonfinite_step' not in report:
                report['first_nonfinite_step'] = offset+int(np.flatnonzero(nonfinite)[0])+1
            report.update(checked_steps=offset+24, max_cfl=max(report['max_cfl'],float(np.nanmax(metrics[:,2]))))
            if invalid.any() and config.stop_on_invalid:
                report['status'] = 'failed_stability'
                write_json(output/'summary.json', summary)
                break
            if (offset+24) % 720 == 0:
                writer.write(state, jax.device_get(sums), (offset+24)//720-1)
                sums = jax.tree_util.tree_map(jnp.zeros_like, sums)
                print(f'{label}: month={(offset+24)//720} max_cfl={report["max_cfl"]:.5g}', flush=True)
                write_json(output/'summary.json', summary)
        else:
            report['status'] = 'completed_with_nonfinite_fields' if 'first_nonfinite_step' in report else 'complete'
        write_json(output/'summary.json', summary)
        del advance, rollout, state, sums
        jax.clear_caches()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    print(run(load_config(parser.parse_args().config, PairedConfig)))


if __name__ == '__main__':
    main()
