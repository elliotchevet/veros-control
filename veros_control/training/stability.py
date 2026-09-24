"""Opt-in correction bounds and transactional optimizer-step backtracking."""
import jax
import jax.numpy as jnp
import numpy as np


def bound_corrections(raw, epsilon, max_change_per_hour):
    """Return bounded physical tendencies; limits have field-units/hour.

    Temperature: deg C/hour; salinity: g/kg/hour; u,v: (m/s)/hour.
    The limit is a rate, independent of the simulation timestep.
    """
    limit = jnp.asarray(max_change_per_hour) / 3600.0
    return limit * jnp.tanh(jnp.asarray(epsilon) * raw / limit)


def backtrack_candidate(params, direction, learning_rate, validate, max_retries=8, factor=0.5):
    """Try one Adam direction at smaller rates; never mutate optimizer state.

    validate returns (valid, diagnostic). Finite parameters are checked before
    simulation. Exhaustion raises, leaving the caller's parameters/state intact.
    """
    rate = learning_rate
    for retry in range(max_retries + 1):
        updates = jax.tree_util.tree_map(lambda x: rate * x, direction)
        candidate = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(candidate))
        valid, diagnostic = validate(candidate) if finite else (False, 'nonfinite candidate parameters')
        print(f'  candidate lr={rate:.6e} valid={valid}: {diagnostic}', flush=True)
        if valid:
            return candidate, updates, rate, retry, diagnostic
        rate *= factor
    raise FloatingPointError(f'Candidate rollout rejected after {max_retries + 1} attempts; previous parameters and optimizer state preserved')


def make_rollout_validator(simulation, params_model, stats, epsilon, chunk_sizes,
                           step_function, cache_scope, max_cfl=1.0, min_salinity=0.0):
    """Check every step of a forward rollout, retaining nested scan structure.

    This is a feasibility check, not a line search requiring loss reduction.
    CFL follows Veros CFLMonitor; it is not a proof of full-scheme stability.
    """
    state0 = simulation.state.copy()
    vs0 = state0.variables
    interior = jnp.zeros_like(vs0.maskT, dtype=bool).at[2:-2, 2:-2, :].set(True)
    wet = (vs0.maskT != 0) & interior
    dt = state0.settings.dt_tracer
    names = ('temp', 'salt', 'u', 'v', 'w', 'rho', 'prho', 'Nsqr', 'tke', 'eke', 'Hd', 'kappaM', 'kappaH')
    masks = {'u': vs0.maskU, 'v': vs0.maskV, 'w': vs0.maskW, 'tke': vs0.maskW,
             'eke': vs0.maskW, 'Nsqr': vs0.maskW, 'kappaM': vs0.maskW, 'kappaH': vs0.maskW}

    def metrics(state):
        vs = state.variables
        bad = jnp.asarray(0)
        for name in names:
            field = getattr(vs, name)
            if field.ndim == 4:
                field = field[..., vs.tau]
            mask = (masks.get(name, vs0.maskT) != 0) & interior
            bad += jnp.sum(mask & ~jnp.isfinite(field))
        salt_min = jnp.min(jnp.where(wet, vs.salt[..., vs.tau], jnp.inf))
        cx = jnp.abs(vs.u[..., vs.tau]) * vs.maskU * dt / (vs.dxt[:, None, None] * vs.cost[None, :, None])
        cy = jnp.abs(vs.v[..., vs.tau]) * vs.maskV * dt / vs.dyt[None, :, None]
        cz = jnp.abs(vs.w[..., vs.tau]) * vs.maskW * dt / vs.dzt[None, None, :]
        cfl = jnp.max(jnp.stack([jnp.max(jnp.where(interior, a, 0.)) for a in (cx, cy, cz)]))
        return jnp.stack((bad, salt_min, cfl))

    def rollout(params):
        def step(carry, _):
            state, previous = carry
            state = step_function(state, simulation, params, params_model, stats, jnp.asarray(epsilon))
            now = metrics(state)
            accumulated = jnp.stack((jnp.maximum(previous[0], now[0]),
                                     jnp.minimum(previous[1], now[1]),
                                     jnp.maximum(previous[2], now[2])))
            return (state, accumulated), None
        current = step
        for size in reversed(chunk_sizes[1:]):
            inner = current
            def chunk(carry, _, inner=inner, size=size):
                return jax.lax.scan(inner, carry, xs=None, length=size)
            current = chunk
        (_, result), _ = jax.lax.scan(current, (state0.copy(), metrics(state0)), xs=None, length=chunk_sizes[0])
        return result

    compiled = jax.jit(rollout)
    def validate(params):
        with cache_scope():
            result = np.asarray(jax.device_get(compiled(params)))
        report = dict(nonfinite_count=int(result[0]), min_salinity=float(result[1]), max_cfl=float(result[2]))
        valid = bool(np.all(np.isfinite(result)) and result[0] == 0 and result[1] > min_salinity and result[2] <= max_cfl)
        return valid, report
    return validate
