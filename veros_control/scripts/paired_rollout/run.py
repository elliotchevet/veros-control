"""Replay a trusted training checkpoint twice; never update weights or optimizer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
p = argparse.ArgumentParser()
p.add_argument('--checkpoint', required=True)
p.add_argument('--output', required=True)
opts = p.parse_args()
out = Path(opts.output).resolve()
if not out.is_relative_to(ROOT):
    raise ValueError('All outputs must stay in veros_control')
out.mkdir(parents=True, exist_ok=True)
if (out / 'metadata.json').exists():
    raise FileExistsError('Use a new output directory')
source = Path(opts.checkpoint).resolve()
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
spec = importlib.util.spec_from_file_location('paired_benchmark', ROOT / 'training/ad_benchmark_root.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from metrics import basin_stats, objective
from veros_control.training.basin_loss import basin_masks, BASIN_NAMES
from veros_control.training.stability import bound_corrections, correction_depth_taper
np, jnp, jax = m.np, m.jnp, m.jax
with source.open('rb') as f:
    c = pickle.load(f)
assert c['n_steps'] == 27 and c['basin_mean_loss'] and c['glorys_linear_interpolation']
assert c['sample_steps'] in (1, 27) and c['loss_reduction'] == 'mean'
assert c['correction_bounds_per_hour'] is not None and not c['conservative_mixing']
sys.argv = [str(spec.origin), '--diskless-mode']
a = m.parse_args()
for key, value in c.items():
    if hasattr(a, key):
        setattr(a, key, value)
a.nn_checkpoint = str(source)
a.feature_stats_file = None  # restore the checkpoint's exact saved feature statistics
sim = m.init_simulation(0, glorys_file=c['glorys_file'],
    glorys_time_index=c['glorys_time_index'], glorys_month_index=c['glorys_month_index'],
    glorys_start_month=c['glorys_start_month'])
params, model, stats, kwargs = m.init_nn_benchmark(sim, a)
sim.prepare_ERA5_forcing_window(sim.state, 27)
base = sim.state.copy()
v = base.variables
dt = float(base.settings.dt_tracer)
assert dt == 3600., dt
alpha = float(c['glorys_interpolation_alpha'])
assert np.isclose(alpha, 27 / 720)
target0 = m.load_glorys_snapshot(sim, c['glorys_file'], ('temp','salt'), time_index=c['glorys_time_index'])
target1 = m.load_glorys_snapshot(sim, c['glorys_target_file'], ('temp','salt'), time_index=c['glorys_time_index'])
target = {k: np.asarray((1-alpha)*target0[k]+alpha*target1[k]) for k in target0}
keys = ('temp','salt','u','v')
mask_names = ('maskT','maskT','maskU','maskV')
area_names = ('area_t','area_t','area_u','area_v')
coords = [(v.xt,v.yt),(v.xt,v.yt),(v.xu,v.yt),(v.xt,v.yu)]
wet, area, basins = {}, {}, {}
for key, mn, an, (x,y) in zip(keys,mask_names,area_names,coords):
    wet[key] = np.array(getattr(v,mn),dtype=bool)
    wet[key][:2] = False; wet[key][-2:] = False
    wet[key][:,:2] = False; wet[key][:,-2:] = False
    area[key] = np.asarray(getattr(v,an))
    basins[key] = basin_masks(np.asarray(x),np.asarray(y))
indices = np.asarray(c['basin_loss_metadata']['level_indices'])
denominators = c['loss_term_denominators']
bounds = np.asarray(c['correction_bounds_per_hour'])
eps = jnp.asarray([c['tracer_epsilon']]*2+[c['momentum_epsilon']]*2)

pattern_lambda = c.get('basin_pattern_lambda', 0.)
taper_config = c.get('correction_depth_taper')
taper = jnp.asarray(correction_depth_taper(v.zt, *taper_config))[:, None] if taper_config else 1.

def physical_model(p, features, coordinates, dzt):
    applied = bound_corrections(model(p,features,coordinates,dzt),eps,bounds) / c['tracer_epsilon']
    return jnp.where(taper > 0, applied * taper, 0.)

def disabled_model(p, features, coordinates, dzt):
    return jnp.zeros((*features.shape[:2],4), dtype=features.dtype)

initial = {k: np.asarray(getattr(v,k)[...,v.tau]) for k in keys}
metadata = dict(checkpoint=str(source),sha256=source_hash,update=int(c['update']),
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,env=dict(os.environ, GIT_CONFIG_GLOBAL='/dev/null', GIT_CONFIG_NOSYSTEM='1')).strip(),
    basin_pattern_lambda=pattern_lambda,correction_depth_taper=taper_config,sample_steps=c['sample_steps'],
    n_steps=27,dt_seconds=dt,alpha=alpha,basins=list(BASIN_NAMES),
    depths_m=np.abs(np.asarray(v.zt)).tolist(),trained_level_indices=indices.tolist(),
    denominators=denominators,bounds_per_hour=bounds.tolist(),
    tracer_epsilon=c['tracer_epsilon'],momentum_epsilon=c['momentum_epsilon'],
    glorys_initial=c['glorys_file'],glorys_next=c['glorys_target_file'],
    reference='Fixed interpolated hour-27 GLORYS target at every diagnostic time',
    correction_timing='Sources saved after step, read at entering tau; means in field units/hour',
    weighting='Wet-cell area within basin/level; momentum on native staggered grids; halos excluded',
    baseline='Same stepping path and forcing, identically zero NN output',
    notes='Final checkpoint weights are post-update; logged training loss is pre-update. Integrated source residual is not a closed physical budget.')
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
np.savez_compressed(out/'grid_target.npz',lon=np.asarray(v.xt),lat=np.asarray(v.yt),
    lon_u=np.asarray(v.xu),lat_v=np.asarray(v.yu),depth=np.abs(np.asarray(v.zt)),dzt=np.asarray(v.dzt),
    **{'wet_'+k:wet[k] for k in keys},**{'area_'+k:area[k] for k in keys},
    **{'target_'+k:target[k] for k in target},**{'initial_'+k:initial[k] for k in keys})
summary = {}
for label, apply_model in [('baseline',disabled_model),('trained',physical_model)]:
    def step(state):
        old_tau = state.variables.tau
        new = m.nn_epsilon_step(state,sim,params,apply_model,stats,jnp.asarray(c['tracer_epsilon']))
        nv = new.variables
        fields = tuple(getattr(nv,k)[...,nv.tau] for k in (*keys,'w'))
        correction = tuple(getattr(nv,k)[...,old_tau] for k in ('dtemp_nn','dsalt_nn','du_nn','dv_nn'))
        cx = jnp.abs(fields[2])*nv.maskU*dt/(nv.dxt[:,None,None]*nv.cost[None,:,None])
        cy = jnp.abs(fields[3])*nv.maskV*dt/nv.dyt[None,:,None]
        cz = jnp.abs(fields[4])*nv.maskW*dt/nv.dzt[None,None,:]
        cfl = jnp.stack([jnp.max(x[2:-2,2:-2]) for x in (cx,cy,cz)])
        return new, (fields, correction, cfl)
    compiled = jax.jit(step)
    state = base.copy()
    histories = {}; integrated = {k:np.zeros_like(initial[k]) for k in keys}
    sum_sq = {k:np.zeros_like(initial[k]) for k in keys}
    sum_corr = {k:np.zeros_like(initial[k]) for k in keys}
    def append(name,value): histories.setdefault(name,[]).append(value)
    bias = {}; rmse = {}
    for k in ('temp','salt'):
        b,r,_ = basin_stats(initial[k]-target[k],area[k],wet[k],basins[k])
        append('bias_'+k,b);append('rmse_'+k,r);bias[k]=b;rmse[k]=r
    terms, loss = objective(bias,indices,denominators,rmse,pattern_lambda)
    append('loss_terms',terms);append('normalized_loss',loss)
    start=time.monotonic()
    for n in range(27):
        with m._temporary_solver_cache():
            state,data = compiled(state)
            fields,corr,cfl = jax.device_get(data)
        if not np.isfinite(cfl).all(): raise FloatingPointError('Nonfinite CFL')
        append('cfl',cfl)
        bias = {}; rmse = {}
        for j,k in enumerate(keys):
            field, correction = np.asarray(fields[j]),np.asarray(corr[j])
            if not np.isfinite(field[wet[k]]).all() or not np.isfinite(correction[wet[k]]).all():
                raise FloatingPointError(f'{label}: nonfinite {k} at step {n+1}')
            if label=='baseline' and np.any(correction != 0): raise AssertionError('Nonzero baseline source')
            if np.max(np.abs(correction[wet[k]]))*3600 > bounds[j]*(1+1e-10):
                raise AssertionError('Applied correction exceeds checkpoint bound')
            integrated[k] += dt*correction;sum_corr[k] += correction;sum_sq[k] += correction**2
            cm,cr,_ = basin_stats(correction*3600,area[k],wet[k],basins[k])
            sat,_,_ = basin_stats((np.abs(correction)*3600 >= .95*bounds[j]).astype(float),area[k],wet[k],basins[k])
            append('corr_mean_'+k,cm);append('corr_rms_'+k,cr);append('saturation_'+k,sat)
            if k in target:
                b,r,_=basin_stats(field-target[k],area[k],wet[k],basins[k]);bias[k]=b;rmse[k]=r
                append('bias_'+k,b);append('rmse_'+k,r)
        terms,loss=objective(bias,indices,denominators,rmse,pattern_lambda)
        append('loss_terms',terms);append('normalized_loss',loss)
        append('salt_min',float(np.min(fields[1][wet['salt']])))
        if n in (0,8,26):
            np.savez_compressed(out/f'{label}_step{n+1:02d}.npz',**dict(zip((*keys,'w'),fields)),
                **{'correction_'+k:np.asarray(corr[j]) for j,k in enumerate(keys)})
        print(f'{label} step={n+1}/27 loss={loss:.9g} cfl={np.max(cfl):.6g} elapsed={time.monotonic()-start:.1f}s',flush=True)
    np.savez_compressed(out/f'{label}_history.npz',**{k:np.asarray(x) for k,x in histories.items()})
    np.savez_compressed(out/f'{label}_integrated.npz',**{'integral_'+k:integrated[k] for k in keys},
        **{'mean_'+k:sum_corr[k]/27 for k in keys},**{'rms_'+k:np.sqrt(sum_sq[k]/27) for k in keys})
    summary[label]=dict(final_normalized_loss=loss,trajectory_mean_normalized_loss=float(np.mean(histories['normalized_loss'][1:])),
        final_loss_terms=terms.tolist(),max_cfl=float(np.max(histories['cfl'])),min_salinity=float(np.min(histories['salt_min'])))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    del state, compiled
    jax.clear_caches()
assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash,'Checkpoint changed'
print('Paired replay complete; source checkpoint unchanged.',flush=True)
