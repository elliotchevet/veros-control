import jax.numpy as npx


STATE_FEATURE_NAMES = ("temp", "salt", "u_center", "v_center")
FORCING_FEATURE_NAMES = (
    "surface_taux",
    "surface_tauy",
    "forc_temp_surface",
    "forc_salt_surface",
    "temp_source",
)

def stack_coords(vs):
    """
    Return horizontal coordinates (lon, lat) for UVTS variables on their respective grids.
    
    Args:
        vs: Veros state.variables
        settings: simulation settings (for coord units)
    
    Returns:
        coords: npx array of shape (nx, ny, 2, 4)
            - 3rd dimension: [lon, lat]
            - 4th dimension: [T, S, U, V] coordinates for each grid point
    """
    
    # Get grid dimensions
    nx, ny = len(vs.xt), len(vs.yt)
    
    # Create coordinate grids for each variable type
    # T grid: uses xt (longitude) and yt (latitude)
    lon_t, lat_t = npx.meshgrid(vs.xt, vs.yt, indexing='ij')  # (nx, ny)
    
    # S grid: same as T grid (collocated)
    lon_s, lat_s = lon_t, lat_t
    
    # U grid: uses xu (longitude) and yt (latitude) - staggered in x
    lon_u, lat_u = npx.meshgrid(vs.xu, vs.yt, indexing='ij')  # (nx, ny)
    
    # V grid: uses xt (longitude) and yu (latitude) - staggered in y
    lon_v, lat_v = npx.meshgrid(vs.xt, vs.yu, indexing='ij')  # (nx, ny)
    
    # Stack longitudes and latitudes for all variables
    # Shape: (nx, ny, 4) for each coordinate type
    lons = npx.stack([lon_t, lon_s, lon_u, lon_v], axis=-1)  # (nx, ny, 4)
    lats = npx.stack([lat_t, lat_s, lat_u, lat_v], axis=-1)  # (nx, ny, 4)
    
    # Stack coordinates: [lon, lat] in 3rd dimension, [T, S, U, V] in 4th dimension
    coords = npx.stack([lons, lats], axis=2)  # shape (nx, ny, 2, 4)
    
    return coords

def stack_input(vs, t_idx, mask = False):
    if mask == True:
        return npx.stack([ vs.maskT[:, :, :],
        vs.maskT[:, :, :],
        vs.maskU[:, :, :],
        vs.maskV[:, :, :],
        ], axis=-1)
    else:
        return npx.stack([
            vs.temp[:, :, :, t_idx],
            vs.salt[:, :, :, t_idx],
            vs.u[:, :, :, t_idx],
            vs.v[:, :, :, t_idx],
        ], axis=-1)


def _maybe_field(vs, name, like):
    try:
        return getattr(vs, name)
    except (AttributeError, RuntimeError):
        pass
    return npx.zeros_like(like)


def _broadcast_surface(field, like_3d):
    return npx.broadcast_to(field[:, :, npx.newaxis], like_3d.shape)


def stack_forcing_input(vs, mask=False):
    if mask:
        return npx.stack([vs.maskT] * len(FORCING_FEATURE_NAMES), axis=-1)

    surface_like = vs.maskT[:, :, -1]
    volume_like = vs.maskT
    temp_source = _maybe_field(vs, "temp_source", volume_like)

    return npx.stack(
        [
            _broadcast_surface(_maybe_field(vs, "surface_taux", surface_like), volume_like),
            _broadcast_surface(_maybe_field(vs, "surface_tauy", surface_like), volume_like),
            _broadcast_surface(_maybe_field(vs, "forc_temp_surface", surface_like), volume_like),
            _broadcast_surface(_maybe_field(vs, "forc_salt_surface", surface_like), volume_like),
            temp_source,
        ],
        axis=-1,
    )


def destagger_uv(u, v):
    """
    Interpolates U and V from Arakawa-C faces to cell centers.
    u shape: (nx, ny, nz)
    v shape: (nx, ny, nz)
    """
    # Shift u by +1 along x-axis (axis 0) to get u[i-1]
    # Average current u[i] (right face) and u[i-1] (left face)
    u_center = 0.5 * (u + npx.roll(u, shift=1, axis=0))
    
    # Shift v by +1 along y-axis (axis 1) to get v[j-1]
    # Average current v[j] (top face) and v[j-1] (bottom face)
    v_center = 0.5 * (v + npx.roll(v, shift=1, axis=1))
    
    return u_center, v_center
    
def restagger_uv(u_center, v_center):
    """
    Interpolates U and V from Cell Centers back to Arakawa-C faces.
    u_face[i] is between center[i] and center[i+1].
    """
    # Shift center by -1 to get center[i+1] at position i
    # Average center[i] and center[i+1]
    u_face = 0.5 * (u_center + npx.roll(u_center, shift=-1, axis=0))
    
    # Average center[j] and center[j+1]
    v_face = 0.5 * (v_center + npx.roll(v_center, shift=-1, axis=1))
    
    return u_face, v_face

def _normalize_features(features, features_mask, stats, scaling_method):
    stat_a, stat_b = stats
    n_feature_channels = features.shape[-1]
    n_stat_channels = stat_a.shape[-1]

    if n_stat_channels not in (4, n_feature_channels):
        raise ValueError(
            "Feature stats channel count is incompatible with NN inputs: "
            f"stats has {n_stat_channels}, features has {n_feature_channels}"
        )

    n_normalized_channels = n_stat_channels
    features_to_normalize = features[..., :n_normalized_channels]
    mask_to_normalize = features_mask[..., :n_normalized_channels]

    if scaling_method == 'meanstd':
        mean, std = stats
        denom = npx.where(npx.abs(std) < 1e-6, 1.0, std)
        normalized = (
            features_to_normalize - mean[None, None, :, :n_normalized_channels]
        ) / denom[None, None, :, :n_normalized_channels]

    elif scaling_method == 'minmax':
        min_val, max_val = stats
        denom = npx.where((max_val - min_val) < 1e-6, 1.0, max_val - min_val)
        normalized = (
            features_to_normalize - min_val[None, None, :, :n_normalized_channels]
        ) / denom[None, None, :, :n_normalized_channels]

    else:
        raise ValueError(f"Unsupported scaling_method: {scaling_method}")

    normalized = npx.where(mask_to_normalize, normalized, 0.0)

    if n_normalized_channels == n_feature_channels:
        return normalized

    remaining = npx.where(
        features_mask[..., n_normalized_channels:],
        features[..., n_normalized_channels:],
        0.0,
    )
    return npx.concatenate([normalized, remaining], axis=-1)


def extract_features(state, tau, stats=None, normalize=False, mask=None, scaling_method=None, use_forcings=False):
    vs = state.variables
    u_raw = vs.u[:, :, :, tau]
    v_raw = vs.v[:, :, :, tau]

    u_center, v_center = destagger_uv(u_raw, v_raw)

    features = npx.stack([
        vs.temp[:, :, :, tau],
        vs.salt[:, :, :, tau],
        u_center,
        v_center
    ], axis=-1)

    features_mask = stack_input(vs, tau, mask=True)

    if use_forcings:
        forcing_features = stack_forcing_input(vs, mask=False)
        forcing_mask = stack_forcing_input(vs, mask=True)
        features = npx.concatenate([features, forcing_features], axis=-1)
        features_mask = npx.concatenate([features_mask, forcing_mask], axis=-1)

    if normalize and stats is not None:
        return _normalize_features(features, features_mask, stats, scaling_method)
    else:
        return features

