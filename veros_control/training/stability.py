"""Feasibility checks and conservative step-size backtracking, not loss descent."""
import jax
import jax.numpy as jnp
import numpy as np
from veros_control.core.rollout import solver_cache_scope


def state_metrics(state):
    v, settings = state.variables, state.settings
    interior = jnp.zeros_like(v.maskT,dtype=bool).at[2:-2,2:-2].set(True)
    masks = {'u':v.maskU,'v':v.maskV,'w':v.maskW,'tke':v.maskW,'eke':v.maskW,
             'Nsqr':v.maskW,'kappaM':v.maskW,'kappaH':v.maskW}
    bad = jnp.asarray(0.)
    for name in ('temp','salt','u','v','w','rho','prho','Nsqr','tke','eke','Hd','kappaM','kappaH'):
        if name not in state.var_meta or not state.var_meta[name].active:
            continue
        field = getattr(v,name)
        if field.ndim == 4: field = field[...,v.tau]
        bad += jnp.sum(interior & (masks.get(name,v.maskT)!=0) & ~jnp.isfinite(field))
    wet = interior & (v.maskT!=0)
    salt = jnp.min(jnp.where(wet,v.salt[...,v.tau],jnp.inf))
    dt = settings.dt_tracer
    cfl = [jnp.abs(v.u[...,v.tau])*v.maskU*dt/(v.dxt[:,None,None]*v.cost[None,:,None]),
           jnp.abs(v.v[...,v.tau])*v.maskV*dt/v.dyt[None,:,None],
           jnp.abs(v.w[...,v.tau])*v.maskW*dt/v.dzt[None,None,:]]
    max_cfl = jnp.max(jnp.stack([jnp.max(jnp.where(interior,x,0.)) for x in cfl]))
    temp_min = jnp.min(jnp.where(wet,v.temp[...,v.tau],jnp.inf))
    temp_max = jnp.max(jnp.where(wet,v.temp[...,v.tau],-jnp.inf))
    return jnp.stack((bad,salt,max_cfl,temp_min,temp_max))


def report_metrics(values):
    return dict(zip(('nonfinite_count','min_salinity','max_cfl','min_temperature','max_temperature'),map(float,values)))


def valid_metrics(values,max_cfl):
    return bool(np.isfinite(values).all() and values[0]==0 and values[1]>0 and values[2]<=max_cfl)


def make_validator(rollout,n_steps,max_cfl):
    def evaluate(params):
        def step(carry,_):
            state, previous = carry
            state = rollout._step(state,params)
            now = state_metrics(state)
            accumulated = jnp.stack((jnp.maximum(previous[0],now[0]),jnp.minimum(previous[1],now[1]),
                jnp.maximum(previous[2],now[2]),jnp.minimum(previous[3],now[3]),jnp.maximum(previous[4],now[4])))
            return (state,accumulated),None
        (_, values),_ = jax.lax.scan(step,(rollout.initial,state_metrics(rollout.initial)),None,length=n_steps)
        return values
    compiled = jax.jit(evaluate)
    def validate(params):
        with solver_cache_scope():
            values = np.asarray(jax.device_get(compiled(params)))
        return valid_metrics(values,max_cfl),report_metrics(values)
    return validate


def backtrack(params,direction,rate,validate,retries):
    for attempt in range(retries+1):
        candidate = jax.tree_util.tree_map(lambda p,d:p+rate*d,params,direction)
        finite = all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(candidate))
        valid,report = validate(candidate) if finite else (False,{'nonfinite_parameters':True})
        print(f'candidate lr={rate:g} valid={valid}: {report}',flush=True)
        if valid:return candidate,rate,report
        rate *= .5
    raise FloatingPointError('No feasible candidate; saved checkpoints preserved')
