"""One stepping path for training and inference, with nested rematerialization."""
from contextlib import contextmanager
import math
import jax
import jax.numpy as jnp
from veros.state import (veros_state_pytree_flatten, veros_variables_pytree_flatten,
                         veros_state_pytree_unflatten, veros_variables_pytree_unflatten)
@contextmanager
def solver_cache_scope():
    """Release solver entries for temporary diagnostic / AD state copies only.

    Existing entries (including a solver for the persistent simulation state) are
    preserved. Removing a cache entry does not change or invalidate the solver.
    """
    from veros.core.external.solvers import get_linear_solver

    cache = getattr(get_linear_solver, "cache", None)
    if not isinstance(cache, dict):
        # Other Veros versions may manage solver lifetime without this cache.
        yield
        return
    existing_keys = set(cache)
    try:
        yield
    finally:
        for key in list(cache):
            if key not in existing_keys:
                del cache[key]
def zero_tendency_sources(state):
    with state.variables.unlock():
        for source_name in ("temp_source", "salt_source", "u_source", "v_source"):
            if source_name not in state.var_meta:
                continue
            try:
                source_value = getattr(state.variables, source_name)
            except RuntimeError as exc:
                if "not active" in str(exc):
                    continue
                raise
            setattr(state.variables, source_name, jnp.zeros_like(source_value))

def nn_set_forcing_step(
    state,
    params,
    simulation,
    model_batched,
    stats=None,
    normalize=False,
    scaling_method=None,
    use_forcings=False,
):
    next_state = state.copy()

    # Traced parameters must not escape through persistent simulation attributes.
    nn_fields = ("_nn_correction_initialized", "_nn_params", "_nn_model_batched", "_nn_stats")
    previous_nn = tuple(getattr(simulation, name, None) for name in nn_fields)
    try:
        simulation._nn_correction_initialized = True
        simulation._nn_params = params
        simulation._nn_model_batched = model_batched
        simulation._nn_stats = stats

        with next_state.settings.unlock():
            next_state.settings.baseline_model = False
            next_state.settings.normalize_inputs = normalize
            next_state.settings.use_forcings = use_forcings
            if scaling_method is not None:
                next_state.settings.scaling_method_features = scaling_method

        zero_tendency_sources(next_state)

        simulation.step(next_state)
    finally:
        for name, value in zip(nn_fields, previous_nn):
            setattr(simulation, name, value)
    return next_state

def make_chunk_sizes(iterations, base=3):
    n = iterations
    chunks = []
    while n > 1:
        if n % base == 0:
            chunks.append(base)
            n //= base
        else:
            chunks.append(n)
            n = 1
    if not chunks:
        chunks = [iterations]
    return tuple(chunks)

def get_prognostic_keys(state):
    prognostic_keys = ["tau", "taum1", "taup1", "itt", "time"]
    for key, meta in state.var_meta.items():
        if meta.active and meta.dims is not None:
            if "timesteps" in meta.dims or meta.write_to_restart:
                if key not in prognostic_keys:
                    prognostic_keys.append(key)
    return prognostic_keys

def inject_dynamic_vars(base_state, dynamic_updates):
    var_leaves, var_aux = veros_variables_pytree_flatten(base_state.variables)
    var_keys = var_aux[0]
    new_leaves = [
        dynamic_updates[key] if key in dynamic_updates else leaf
        for key, leaf in zip(var_keys, var_leaves)
    ]
    new_vars = veros_variables_pytree_unflatten(var_aux, new_leaves)

    _, state_aux = veros_state_pytree_flatten(base_state)
    return veros_state_pytree_unflatten(state_aux, [new_vars])


def nested_scan(step, carry, length, base=3):
    """Checkpoint individual steps and nested chunks; keep one dynamic carry."""
    sizes = make_chunk_sizes(length, base)
    current = jax.checkpoint(step)
    for size in reversed(sizes[1:]):
        inner = current
        def chunk(state, indices, inner=inner, size=size):
            return jax.lax.scan(inner, state, indices, length=size)
        current = jax.checkpoint(chunk)
    return jax.lax.scan(current, carry, jnp.arange(1, length+1).reshape(sizes), length=sizes[0])


class Rollout:
    """Compiled single step plus a differentiable sampled training objective."""
    def __init__(self, simulation, model, stats, normalization='none', use_forcings=True):
        self.simulation = simulation
        self.initial = simulation.state.copy()
        self.keys = get_prognostic_keys(self.initial)
        self.model, self.stats = model, stats
        self.normalization, self.use_forcings = normalization, use_forcings
        self.step = jax.jit(self._step)

    def _step(self, state, params):
        return nn_set_forcing_step(state, params, self.simulation, self.model, self.stats,
            normalize=self.normalization != 'none', scaling_method=self.normalization if self.normalization != 'none' else None,
            use_forcings=self.use_forcings)

    def objective(self, terms, n_steps, sample_steps, chunk_base=3):
        initial, keys = self.initial, self.keys
        carry = {key:getattr(initial.variables,key) for key in keys}
        n_terms = int(terms(initial).shape[0])
        def evaluate(params, denominators):
            def step(dynamic, index):
                state = inject_dynamic_vars(initial, dynamic)
                next_state = self._step(state, params)
                values = jax.lax.cond(index % sample_steps == 0, terms,
                    lambda _:jnp.zeros((n_terms,)), next_state)
                return {key:getattr(next_state.variables,key) for key in keys}, values
            _, sampled = nested_scan(step, carry, n_steps, chunk_base)
            raw = sampled.reshape(-1,n_terms).sum(axis=0)/(n_steps//sample_steps)
            return jnp.mean(raw/denominators), raw
        return jax.jit(evaluate), jax.jit(jax.value_and_grad(evaluate, has_aux=True))
