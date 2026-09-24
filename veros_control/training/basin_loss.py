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


def make_basin_loss(variables, target, keys=('temp', 'salt'), levels=5):
    if levels not in (5, 10):
        raise ValueError('Basin loss requires 5 or 10 shallowest levels')
    if not keys or any(k not in ('temp', 'salt') for k in keys):
        raise ValueError('Basin loss supports temperature and salinity only')
    z = np.asarray(variables.zt)
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
    for key in keys:
        truth = np.asarray(target[key])
        if truth.ndim != 3:
            raise ValueError('Basin target must be a three-dimensional GLORYS snapshot')
        if not np.isfinite(truth[..., indices][np.any(weights > 0, axis=0)]).all():
            raise ValueError(f'Nonfinite GLORYS {key} in a selected wet basin cell')
        target_means[key] = means(jnp.asarray(truth))

    def terms(state):
        result = []
        for key in keys:
            field = getattr(state.variables, key)
            if field.ndim == 4:
                field = field[..., state.variables.tau]
            squared = (means(field) - target_means[key]) ** 2
            result.append(jnp.sum(jnp.where(active_jax, squared, 0.)) / jnp.sum(active_jax))
        return jnp.stack(result)

    metadata = dict(version=DEFINITION_VERSION, names=list(BASIN_NAMES),
                    level_indices=indices.tolist(), depths_m=np.abs(z[indices]).tolist(),
                    active_pairs=active.tolist(), target_means={k:np.asarray(v).tolist() for k,v in target_means.items()})
    return lambda state: jnp.mean(terms(state)), tuple(keys), terms, metadata
