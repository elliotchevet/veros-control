import jax.numpy as jnp


def _target_at_tau(target, key, tau):
    value = target[key]
    if value.ndim == 4:
        return value[..., tau]
    return value


def _mask_for_key(state, key):
    if key in ("temp", "salt"):
        return state.variables.maskT
    if key == "u":
        return state.variables.maskU
    if key == "v":
        return state.variables.maskV
    if key == "ssh":
        return state.variables.maskT[..., -1]
    return None


def compute_loss(state, target, keys=("temp", "salt"), tau=None):
    if tau is None:
        tau = state.variables.tau

    loss = 0.0
    n_terms = 0

    for key in keys:
        pred = getattr(state.variables, key)
        if pred.ndim == 4:
            pred = pred[..., tau]

        pred = jnp.asarray(pred)
        true = jnp.asarray(_target_at_tau(target, key, tau))
        mask = _mask_for_key(state, key)

        if mask is not None:
            mask = jnp.asarray(mask, dtype=pred.dtype)
            squared_error = ((pred - true) ** 2) * mask
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            loss += jnp.sum(squared_error) / denom
        else:
            loss += jnp.mean((pred - true) ** 2)
        n_terms += 1

    return loss / max(n_terms, 1)
