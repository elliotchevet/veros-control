"""Forward-only ablations of NN output channels, with per-step stability data."""
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

source = Path(__file__).resolve().parents[2] / 'training/ad_benchmark_root.py'
sys.argv[0] = str(source)
spec = importlib.util.spec_from_file_location('loss_benchmark', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
jax, jnp, np = m.jax, m.jnp, m.np
out = Path(os.environ['LOSS_DEBUG_OUT'])
original_train = m.run_train_mode

def inspect_rollouts(*args, **kwargs):
    values = inspect.signature(original_train).bind(*args, **kwargs).arguments
    sim, cfg = values['simulation'], values['args']
    steps = m._resolve_train_n_steps(cfg, values['steps_list'])
    sim.prepare_ERA5_forcing_window(sim.state, steps)
    params, model, stats = values['nn_params'], values['model_batched_unscaled'], values['nn_stats']
    loss_terms_function = values['loss_terms_function']
    with sim.state.settings.unlock():
        sim.state.settings.baseline_model = False
    base = sim.state
    keys = ('temp', 'salt', 'u', 'v', 'w', 'tke', 'eke')

    def rollout(params_arg, scales):
        def scaled_model(p, features, coordinates, dzt):
            predictions = model(p, features, coordinates, dzt)
            return jnp.where(scales != 0, predictions * scales, 0.)
        def step(state, index):
            next_state = m.nn_epsilon_step(state, sim, params_arg, scaled_model, stats,
                                           jnp.asarray(cfg.epsilon))
            bad, maximum = [], []
            for key in keys:
                field = m._field_at_tau(getattr(next_state.variables, key), next_state.variables.tau)
                bad.append(jnp.sum(~jnp.isfinite(field)))
                maximum.append(jnp.max(jnp.abs(field)))
            return next_state, (loss_terms_function(next_state), jnp.stack(bad), jnp.stack(maximum))
        _, metrics = jax.lax.scan(step, base.copy(), jnp.arange(steps))
        return metrics
    compiled = jax.jit(rollout)
    results = {}
    variants = [('all_corrections', [1.,1.,1.,1.]),
                ('temp_salt_only', [1.,1.,0.,0.]),
                ('no_nn_corrections', [0.,0.,0.,0.]),
                ('momentum_01', [1.,1.,0.1,0.1])]
    for label, scales in variants:
        start = time.monotonic()
        with m._temporary_solver_cache():
            metrics = compiled(params, jnp.asarray(scales))
            jax.block_until_ready(metrics)
        terms, bad, maximum = [np.asarray(x) for x in jax.device_get(metrics)]
        np.savez(out / f'{label}.npz', loss_terms=terms, nonfinite_count=bad, max_abs=maximum)
        failed = np.flatnonzero(np.any(bad > 0, axis=1) | ~np.all(np.isfinite(terms), axis=1))
        result = dict(first_nonfinite_step=int(failed[0]+1) if len(failed) else None,
                      final_loss_terms=terms[-1].tolist(),
                      max_abs_by_field=dict(zip(keys, maximum[-1].tolist())),
                      elapsed_seconds=time.monotonic()-start)
        results[label] = result
        print('ABLATION', label, json.dumps(result), flush=True)
        for index in (0,26,53,80):
            if index < steps:
                print('STEP', label, index+1, 'loss_terms', terms[index].tolist(),
                      'bad', dict(zip(keys,bad[index].tolist())),
                      'max_abs', dict(zip(keys,maximum[index].tolist())), flush=True)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        del metrics, terms, bad, maximum
    print('Forward ablations complete; no optimizer updates or training checkpoints written.')

m.run_train_mode = inspect_rollouts
m.main()
