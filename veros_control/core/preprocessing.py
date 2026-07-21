import jax.numpy as npx

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

def extract_features(state, tau, stats=None, normalize=False, mask=None, scaling_method=None):
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

    if normalize and stats is not None:
        if scaling_method == 'meanstd':
            mean, std = stats
            # Broadcasting over (nx, ny)
            features_normalized = (features - mean[None, None, :, :]) / std[None, None, :, :]
            features_normalized = npx.where(features_mask, features_normalized, 0.0)
            return features_normalized

        elif scaling_method == 'minmax':
            min_val, max_val = stats
            denom = npx.where((max_val - min_val) < 1e-6, 1.0, max_val - min_val)
            features_normalized = (features - min_val) / denom
            features_normalized = npx.where(features_mask, features_normalized, 0.0)
            return features_normalized

        else:
            raise ValueError(f"Unsupported scaling_method: {scaling_method}")
    else:
        return features
