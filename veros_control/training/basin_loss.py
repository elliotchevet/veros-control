"""Area-weighted tracer means over the six agreed rectangular domains."""
import numpy as np
import jax.numpy as jnp

BASIN_NAMES = ('north_atlantic', 'south_atlantic', 'north_pacific', 'south_pacific', 'indian', 'southern')
DEFINITION_VERSION = 'rectangles_equator_split_v1'


def basin_masks(lon, lat):
    lon = (np.asarray(lon) + 180.) % 360. - 180.
    lat = np.asarray(lat)
    x, y = lon[:, None], lat[None, :]
    atl = (x >= -70.) & (x < 20.)
    pac = (x >= 120.) | (x < -70.)
    north = (y >= 0.) & (y <= 90.)
    south = (y >= -60.) & (y < 0.)
    return np.stack(np.broadcast_arrays(atl & north, atl & south, pac & north, pac & south,
        (x >= 20.) & (x < 120.) & (y >= -60.) & (y < 30.),
        np.ones_like(x, dtype=bool) & (y >= -90.) & (y < -60.)))


def make_basin_loss(variables, target, keys=('temp', 'salt'), levels=5, pattern_lambda=0.,
                    initial_target=None, target_alpha=None):
    if not np.isfinite(pattern_lambda) or pattern_lambda < 0:
        raise ValueError('Pattern lambda must be finite and nonnegative')
    if not isinstance(levels, (int, np.integer)) or levels < 0:
        raise ValueError('Basin levels must be a nonnegative integer (0 selects all)')
    if not keys or any(k not in ('temp', 'salt') for k in keys):
        raise ValueError('Basin loss supports temperature and salinity only')
    z = np.asarray(variables.zt)
    if levels == 0:
        levels = len(z)
    if len(z) < levels:
        raise ValueError('Requested more basin levels than the model has')
    # Select by physical depth, not array order (Veros stores surface last).
    indices = np.argsort(np.abs(z), kind='stable')[:levels]
    masks = basin_masks(variables.xt, variables.yt)
    masks[:, :2, :] = False; masks[:, -2:, :] = False
    masks[:, :, :2] = False; masks[:, :, -2:] = False
    wet = np.asarray(variables.maskT)[..., indices] != 0
    area = np.asarray(variables.area_t)
    weights = masks[..., None] * wet[None, ...] * area[None, ..., None]
    denom = weights.sum(axis=(1, 2))
    active = denom > 0
    if not active.any():
        raise ValueError('Basin loss has no wet basin-level pairs')
    normalized = jnp.asarray(weights / np.where(active, denom, 1.)[:, None, None, :])
    selected = jnp.asarray(indices)
    active_jax = jnp.asarray(active)

    def means(field):
        value = jnp.take(field, selected, axis=2)
        # Mask before multiplication: dry and unselected cells cannot pollute means.
        return jnp.sum(jnp.where(normalized > 0, value[None, ...], 0.) * normalized, axis=(1, 2))

    target_means = {}
    target_fields = {}
    for key in keys:
        truth = np.asarray(target[key])
        if truth.ndim != 3:
            raise ValueError('Basin target must be a three-dimensional GLORYS snapshot')
        if not np.isfinite(truth[..., indices][np.any(weights > 0, axis=0)]).all():
            raise ValueError(f'Nonfinite GLORYS {key} in a selected wet basin cell')
        target_means[key] = means(jnp.asarray(truth))
        target_fields[key] = jnp.take(jnp.asarray(truth), selected, axis=2)

    initial_fields = {}
    initial_means = {}
    if initial_target is not None:
        if target_alpha is None:
            raise ValueError('Time-varying targets require an alpha function')
        for key in keys:
            truth = np.asarray(initial_target[key])
            if truth.shape != np.asarray(target[key]).shape or not np.isfinite(truth[..., indices][np.any(weights > 0, axis=0)]).all():
                raise ValueError('Invalid initial GLORYS target')
            initial_fields[key] = jnp.take(jnp.asarray(truth), selected, axis=2)
            initial_means[key] = means(jnp.asarray(truth))

    def terms(state):
        result = []
        for key in keys:
            field = getattr(state.variables, key)
            if field.ndim == 4:
                field = field[..., state.variables.tau]
            truth_mean, truth_field = target_means[key], target_fields[key]
            if initial_target is not None:
                alpha = target_alpha(state)
                truth_mean = initial_means[key] * (1-alpha) + truth_mean * alpha
                truth_field = initial_fields[key] * (1-alpha) + truth_field * alpha
            squared = (means(field) - truth_mean) ** 2
            if pattern_lambda > 0:
                error = jnp.take(field, selected, axis=2) - truth_field
                masked_error = jnp.where(normalized > 0, error[None, ...], 0.)
                if pattern_lambda == 1.:
                    squared = jnp.sum(normalized * masked_error ** 2, axis=(1, 2))
                else:
                    bias = jnp.sum(normalized * masked_error, axis=(1, 2))
                    centered = jnp.where(normalized > 0, masked_error - bias[:, None, None, :], 0.)
                    squared = bias ** 2 + pattern_lambda * jnp.sum(normalized * centered ** 2, axis=(1, 2))
            result.append(jnp.sum(jnp.where(active_jax, squared, 0.)) / jnp.sum(active_jax))
        return jnp.stack(result)

    metadata = dict(version=DEFINITION_VERSION, pattern_lambda=float(pattern_lambda), names=list(BASIN_NAMES),
                    level_indices=indices.tolist(), depths_m=np.abs(z[indices]).tolist(),
                    active_pairs=active.tolist(), target_means={k:np.asarray(v).tolist() for k,v in target_means.items()})
    return lambda state: jnp.mean(terms(state)), tuple(keys), terms, metadata
