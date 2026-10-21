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


def backtrack_candidate(params, direction, learning_rate, validate, max_retries=8, factor=0.5,
                        score=None, reference_loss=None, loss_rtol=1e-8, loss_atol=1e-10,
                        min_learning_rate=0., enforce_loss_decrease=True):
    """Try one Adam direction at smaller rates; never mutate optimizer state.

    validate returns (valid, diagnostic). Finite parameters are checked before
    simulation. Exhaustion raises, leaving the caller's parameters/state intact.
    """
    if score is not None and (reference_loss is None or not np.isfinite(reference_loss)):
        raise ValueError('Loss backtracking requires a finite reference loss')
    if not np.isfinite(min_learning_rate) or min_learning_rate < 0:
        raise ValueError("Minimum learning rate must be finite and nonnegative")
    rate = max(learning_rate, min_learning_rate)
    for retry in range(max_retries + 1):
        updates = jax.tree_util.tree_map(lambda x: rate * x, direction)
        candidate = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(candidate))
        valid, diagnostic = validate(candidate) if finite else (False, 'nonfinite candidate parameters')
        if valid and score is not None:
            value, terms = score(candidate)
            value = float(value)
            terms = np.asarray(terms)
            limit = reference_loss + loss_atol + loss_rtol * abs(reference_loss)
            require_decrease = enforce_loss_decrease and (min_learning_rate == 0 or rate > min_learning_rate)
            valid = bool(np.isfinite(value) and np.isfinite(terms).all() and (not require_decrease or value <= limit))
            diagnostic = dict(diagnostic, loss=value, loss_limit=limit, loss_terms=terms.tolist())
        print(f'  candidate lr={rate:.6e} valid={valid}: {diagnostic}', flush=True)
        if valid:
            return candidate, updates, rate, retry, diagnostic
        if min_learning_rate > 0 and rate <= min_learning_rate:
            break
        rate = max(min_learning_rate, rate * factor)
    raise FloatingPointError(f'Candidate rollout rejected after {max_retries + 1} attempts; previous parameters and optimizer state preserved')


def recover_learning_rate(backoff, retries, clean_updates):
    """Persist reductions; recover 50% after five consecutive first-try acceptances."""
    backoff *= 0.5 ** retries
    clean_updates = 0 if retries else clean_updates + 1
    if clean_updates >= 5:
        backoff = min(1., backoff * 1.5)
        clean_updates = 0
    return backoff, clean_updates


def make_rollout_validator(simulation, params_model, stats, epsilon, chunk_sizes,
                           step_function, cache_scope, max_cfl=1.0, min_salinity=0.0,
                           score_terms=None, sample_steps=1, score_denominators=None):
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
            state, previous, total = carry
            state = step_function(state, simulation, params, params_model, stats, jnp.asarray(epsilon))
            now = metrics(state)
            accumulated = jnp.stack((jnp.maximum(previous[0], now[0]),
                                     jnp.minimum(previous[1], now[1]),
                                     jnp.maximum(previous[2], now[2])))
            if score_terms is not None:
                sampled = ((state.variables.itt - state0.variables.itt) % sample_steps) == 0
                total = total + jax.lax.cond(sampled, score_terms, lambda s: jnp.zeros(2), state)
            return (state, accumulated, total), None
        current = step
        for size in reversed(chunk_sizes[1:]):
            inner = current
            def chunk(carry, _, inner=inner, size=size):
                return jax.lax.scan(inner, carry, xs=None, length=size)
            current = chunk
        (_, result, total), _ = jax.lax.scan(current, (state0.copy(), metrics(state0), jnp.zeros(2)), xs=None, length=chunk_sizes[0])
        return result, total

    compiled = jax.jit(rollout)
    def validate(params):
        with cache_scope():
            result, total = jax.device_get(compiled(params))
            result = np.asarray(result)
        report = dict(nonfinite_count=int(result[0]), min_salinity=float(result[1]), max_cfl=float(result[2]))
        valid = bool(np.all(np.isfinite(result)) and result[0] == 0 and result[1] > min_salinity and result[2] <= max_cfl)
        if score_terms is not None:
            terms = np.asarray(total) / (int(np.prod(chunk_sizes)) // sample_steps)
            loss = float(np.mean(terms / np.asarray(score_denominators)))
            report.update(loss=loss, loss_terms=terms.tolist())
            valid = valid and np.isfinite(loss) and np.isfinite(terms).all()
        return bool(valid), report
    return validate


def correction_depth_taper(zt, full_levels=5, transition_levels=5):
    """Raised cosine in physical depth: full strength through k, zero at k+n."""
    import numpy as np
    depths = np.abs(np.asarray(zt))
    if full_levels < 1 or transition_levels < 1 or full_levels + transition_levels > len(depths):
        raise ValueError('Depth taper requires positive level counts within the vertical grid')
    ordered = np.sort(depths)
    start, stop = ordered[full_levels - 1], ordered[full_levels + transition_levels - 1]
    if stop <= start:
        raise ValueError('Depth taper needs distinct transition depths')
    fraction = np.clip((depths - start) / (stop - start), 0., 1.)
    return 0.5 * (1. + np.cos(np.pi * fraction))
