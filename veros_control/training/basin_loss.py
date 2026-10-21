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
                    initial_target=None, target_alpha=None, depth_loss=None, spatial_filter=None, lead_hours=None,
                    return_depth_errors=False):
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

    filter_error = None
    if spatial_filter is not None:
        if pattern_lambda != 1. or lead_hours is None:
            raise ValueError('Filtered loss requires lambda=1 and a lead-time function')
        from veros_control.training.spatial_filter import make_spatial_filter
        filter_error = make_spatial_filter(variables, indices, spatial_filter)
    depth_scales = {}
    if depth_loss is not None:
        if not depth_loss['thickness_weighting']:
            raise ValueError('Depth-normalized loss requires thickness weighting')
        if not np.allclose(np.abs(z), depth_loss['depths_m']):
            raise ValueError('Depth normalization grid differs from model')
        h = np.asarray(variables.dzt)[indices]
        if not np.isfinite(h).all() or np.any(h <= 0):
            raise ValueError('Layer thickness must be finite and positive')
        if not np.allclose(np.asarray(variables.dzt), depth_loss['thickness_m']):
            raise ValueError('Depth normalization thickness differs from model')
        # Equal basin weights; thickness average within each wet basin column.
        thickness = active * h[None, :]
        basin_active = active.any(axis=1)
        boundaries = depth_loss.get('band_boundaries_m')
        if boundaries is None:
            depth_weights = thickness / np.maximum(thickness.sum(axis=1, keepdims=True), 1e-30)
        else:
            boundaries = np.asarray(boundaries, dtype=float)
            if boundaries.ndim != 1 or not np.isfinite(boundaries).all() or np.any(boundaries <= 0) or np.any(np.diff(boundaries) <= 0):
                raise ValueError('Depth band boundaries must be positive and increasing')
            # Assign each model level by its centre; normalize within each wet band.
            band = np.digitize(np.abs(z[indices]), boundaries)
            depth_weights = np.zeros_like(thickness)
            band_count = np.zeros((len(active), 1))
            for group in range(len(boundaries) + 1):
                w = thickness * (band == group)[None, :]
                total = w.sum(axis=1, keepdims=True)
                depth_weights += w / np.maximum(total, 1e-30)
                band_count += total > 0
            depth_weights /= np.maximum(band_count, 1)
        depth_weights = jnp.asarray(depth_weights / basin_active.sum())
        for key in keys:
            scale = np.asarray(depth_loss['denominators'][key])
            if scale.shape != z.shape or not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError('Invalid fixed depth denominators')
            depth_scales[key] = jnp.asarray(scale[indices])

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
                    if filter_error is not None:
                        smooth = filter_error(error, lead_hours(state))
                        smooth = jnp.where(normalized > 0, smooth[None, ...], 0.)
                        mix = spatial_filter['raw_fraction']
                        squared = mix*squared+(1-mix)*jnp.sum(normalized*smooth**2,axis=(1,2))
                else:
                    bias = jnp.sum(normalized * masked_error, axis=(1, 2))
                    centered = jnp.where(normalized > 0, masked_error - bias[:, None, None, :], 0.)
                    squared = bias ** 2 + pattern_lambda * jnp.sum(normalized * centered ** 2, axis=(1, 2))
            if return_depth_errors:
                result.append(jnp.sum(jnp.where(active_jax,squared,0.),axis=0)/jnp.maximum(jnp.sum(active_jax,axis=0),1))
            elif depth_loss is None:
                result.append(jnp.sum(jnp.where(active_jax, squared, 0.)) / jnp.sum(active_jax))
            else:
                result.append(jnp.sum(jnp.where(active_jax, squared, 0.) * depth_weights / depth_scales[key]))
        return jnp.stack(result)

    metadata = dict(version=DEFINITION_VERSION, pattern_lambda=float(pattern_lambda), names=list(BASIN_NAMES),
                    level_indices=indices.tolist(), depths_m=np.abs(z[indices]).tolist(),
                    active_pairs=active.tolist(), target_means={k:np.asarray(v).tolist() for k,v in target_means.items()})
    if spatial_filter is not None:
        metadata['spatial_filter'] = spatial_filter
    if depth_loss is not None:
        metadata['depth_loss'] = depth_loss
    return lambda state: jnp.mean(terms(state)), tuple(keys), terms, metadata
