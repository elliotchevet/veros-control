"""Forward diagnostics only: local CFL and salinity at fixed cells, checkpoint 24."""
import importlib.util
import inspect
import json
import os
import pickle
from pathlib import Path
import sys

source = Path(__file__).resolve().parents[2] / 'training/ad_benchmark_root.py'
sys.argv[0] = str(source)
spec = importlib.util.spec_from_file_location('cfl_benchmark', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
jax, jnp, np = m.jax, m.jnp, m.np
out = Path(os.environ['LOSS_DEBUG_OUT'])
original_train = m.run_train_mode


def inspect_rollout(*args, **kwargs):
    values = inspect.signature(original_train).bind(*args, **kwargs).arguments
    sim, cfg = values['simulation'], values['args']
    with open(os.environ['SOURCE_CHECKPOINT'], 'rb') as f:
        saved = pickle.load(f)
    assert saved['update'] == 24, saved['update']
    params = saved['params']
    steps = m._resolve_train_n_steps(cfg, values['steps_list'])
    sim.prepare_ERA5_forcing_window(sim.state, steps)
    with sim.state.settings.unlock():
        sim.state.settings.baseline_model = False
    base, model, stats = sim.state, values['model_batched_unscaled'], values['nn_stats']
    dt = base.settings.dt_tracer
    wet = base.variables.maskT.astype(bool)
    interior = jnp.zeros_like(wet).at[2:-2, 2:-2, :].set(True)
    wet = wet & interior
    fields = ('salt', 'u', 'v', 'w', 'rho', 'tke')
    summary_names = ['salt_min', 'salt_negative_count', 'salt_min_i', 'salt_min_j', 'salt_min_k',
                     'cx_max', 'cy_max', 'cz_max', 'face_sum_max', 'cells_face_cfl_gt1',
                     'cfl_max_i', 'cfl_max_j', 'cfl_max_k'] + ['bad_' + f for f in fields]
    probe_names = ['salt_before','salt_after','cx_face_max','cy_face_max','cz_face_max','face_sum',
                   'nn_salt_dt','adv_salt_dt','vmix_salt_dt','hmix_salt_dt',
                   'u_before','v_before','w_before','salt_west','salt_east','salt_south','salt_north']

    def cfl(state):
        vs = state.variables
        cx = jnp.abs(vs.u[..., vs.tau]) * vs.maskU * dt / (vs.dxt[:,None,None]*vs.cost[None,:,None])
        cy = jnp.abs(vs.v[..., vs.tau]) * vs.maskV * dt / vs.dyt[None,:,None]
        cz = jnp.abs(vs.w[..., vs.tau]) * vs.maskW * dt / vs.dzt[None,None,:]
        # Both bounding faces for each tracer cell; sum is a diagnostic indicator,
        # not a claimed exact stability criterion for the full Veros scheme.
        x = jnp.maximum(cx, jnp.roll(cx, 1, axis=0))
        y = jnp.maximum(cy, jnp.roll(cy, 1, axis=1))
        z = jnp.maximum(cz, jnp.concatenate((jnp.zeros_like(cz[:,:,:1]),cz[:,:,:-1]),axis=2))
        return cx, cy, cz, x, y, z

    def rollout(p, scales, probes):
        i,j,k = probes[:,0],probes[:,1],probes[:,2]
        def model_scaled(p, features, coords, dzt):
            prediction = model(p, features, coords, dzt)
            return jnp.where(scales != 0, prediction * scales, 0.)
        def step(state, index):
            vs = state.variables
            salt_before = vs.salt[...,vs.tau]
            cx,cy,cz,x,y,z = cfl(state)
            summed = x+y+z
            next_state = m.nn_epsilon_step(state,sim,p,model_scaled,stats,jnp.asarray(cfg.epsilon))
            nv = next_state.variables
            salt = nv.salt[...,nv.tau]
            minloc = jnp.stack(jnp.unravel_index(jnp.argmin(jnp.where(wet,salt,jnp.inf)),salt.shape))
            maxloc = jnp.stack(jnp.unravel_index(jnp.argmax(jnp.where(wet,summed,-jnp.inf)),salt.shape))
            maxima = [jnp.max(jnp.where(interior,a,0.)) for a in (cx,cy,cz)]
            summary = jnp.concatenate((jnp.asarray([jnp.min(jnp.where(wet,salt,jnp.inf)),jnp.sum(wet & (salt<0))]), minloc,
                jnp.stack(maxima),jnp.asarray([jnp.max(jnp.where(wet,summed,0.)),jnp.sum(wet & (jnp.maximum(jnp.maximum(x,y),z)>1))]),maxloc,
                jnp.stack([jnp.sum(wet & ~jnp.isfinite(m._field_at_tau(getattr(nv,f),nv.tau))) for f in fields])))
            probe = jnp.stack([salt_before[i,j,k],salt[i,j,k],x[i,j,k],y[i,j,k],z[i,j,k],summed[i,j,k],
                dt*nv.salt_source[i,j,k],dt*nv.dsalt[i,j,k,vs.tau],dt*nv.dsalt_vmix[i,j,k],dt*nv.dsalt_hmix[i,j,k],
                vs.u[i,j,k,vs.tau],vs.v[i,j,k,vs.tau],vs.w[i,j,k,vs.tau],
                salt_before[i-1,j,k],salt_before[i+1,j,k],salt_before[i,j-1,k],salt_before[i,j+1,k]],axis=-1)
            return next_state,(summary,probe,values['loss_terms_function'](next_state))
        end, metrics = jax.lax.scan(step,base.copy(),jnp.arange(steps))
        return end.variables.salt[...,end.variables.tau], metrics

    compiled = jax.jit(rollout)
    probes = np.tile(np.array([[2,2,0]],dtype=np.int32),(16,1))
    with m._temporary_solver_cache():
        final_salt, metrics = compiled(params,jnp.ones(4),jnp.asarray(probes))
        final_salt,metrics = jax.device_get((final_salt,metrics))
    wet_host = np.asarray(wet)
    negative = np.argwhere(wet_host & (final_salt<0))
    summary = np.asarray(metrics[0])
    candidates = list(negative) + [row[2:5].astype(int) for row in summary] + [row[10:13].astype(int) for row in summary]
    unique=[]
    for index in candidates:
        item=tuple(map(int,index))
        if item not in unique: unique.append(item)
    unique=unique[:16]
    probes[:len(unique)] = np.array(unique)
    vs=base.variables
    metadata=dict(checkpoint_update=24,dt=float(dt),summary_names=summary_names,probe_names=probe_names,
                  probes=[dict(index=list(q),lon=float(vs.xt[q[0]]),lat=float(vs.yt[q[1]]),z=float(vs.zt[q[2]])) for q in unique],
                  cfl_timing='CFL from state entering step; salt and nonfinite counts from state leaving step',
                  notes='CFL components follow Veros CFLMonitor. Probe components take maximum of adjacent faces. face_sum is an indicator, not an exact scheme stability criterion. Source and tendency increments are diagnostics, not a complete additive budget (Adams-Bashforth and implicit processes apply).')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('PROBES',json.dumps(metadata['probes']),flush=True)
    results={}
    for name,scale in [('all',[1.,1.,1.,1.]),('momentum_01',[1.,1.,.1,.1]),('no_momentum',[1.,1.,0.,0.]),('no_salt_nn',[1.,0.,1.,1.]),('no_nn',[0.,0.,0.,0.])]:
        with m._temporary_solver_cache():
            fs,data=compiled(params,jnp.asarray(scale),jnp.asarray(probes))
            fs,data=jax.device_get((fs,data))
        summary,history,loss=map(np.asarray,data)
        np.savez_compressed(out/(name+'.npz'),summary=summary,probes=history[:,:len(unique)],loss_terms=loss)
        results[name]=dict(final_loss_terms=loss[-1].tolist(),final_min_salt=float(summary[-1,0]),negative_steps=(np.flatnonzero(summary[:,1]>0)+1).tolist(),
                           max_cfl_xyz=np.max(summary[:,5:8],axis=0).tolist(),first_component_cfl_gt1=next((int(i+1) for i,row in enumerate(summary) if np.max(row[5:8])>1),None))
        print('RESULT',name,json.dumps(results[name]),flush=True)
        if name=='all':
            for i,row in enumerate(summary): print('STEP',i+1,json.dumps(dict(zip(summary_names,row.tolist()))),flush=True)
            for n,q in enumerate(unique): print('PROBE_HISTORY',q,json.dumps(history[:,n,:].tolist()),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('Done. No optimizer updates; original checkpoint unchanged.',flush=True)

m.run_train_mode=inspect_rollout
m.main()
