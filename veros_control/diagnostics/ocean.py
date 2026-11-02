"""Offline evaluation kernels; arrays use depth, latitude, longitude order."""
import numpy as np
import gsw

BASINS = ('Southern', 'Atlantic', 'Indian', 'Pacific')
LAT_LIMITS = ((-90, -30), (-30, 90), (-30, 30), (-30, 60))

def basin_masks(lon, lat):
    x = (np.asarray(lon)[None, :] + 180) % 360 - 180
    y = np.asarray(lat)[:, None]
    return np.stack(np.broadcast_arrays(
        (y < -30) & (y >= -90) & np.ones_like(x, bool),
        (y >= -30) & (y <= 90) & (x >= -70) & (x < 20),
        (y >= -30) & (y <= 30) & (x >= 20) & (x < 120),
        (y >= -30) & (y <= 60) & ((x >= 120) | (x < -70))))

def weighted_mean(values, weights, axis=None):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    numerator = np.sum(np.where(valid, values * weights, 0), axis=axis)
    denominator = np.sum(np.where(valid, weights, 0), axis=axis)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)

def thermodynamics(temp, salt, depth, lon, lat, convention):
    """Return Conservative Temperature and Absolute Salinity for TEOS-10."""
    if convention == 'ct-sa':
        return temp, salt
    if convention != 'pt-sp':
        raise ValueError(convention)
    pressure = gsw.p_from_z(-np.abs(depth)[:, None, None], lat[None, :, None])
    sa = gsw.SA_from_SP(salt, pressure, lon[None, None, :] % 360, lat[None, :, None])
    return gsw.CT_from_pt(sa, temp), sa

def mld_density(temp, salt, depth, bottom, lon, lat, convention):
    """0.03 kg/m3 relative to interpolated sigma0 at 10 m; first crossing.

    If no crossing exists in a complete wet profile, return bottom depth. A
    missing internal level invalidates the column; land/shallow (<10m) is NaN.
    """
    order = np.argsort(np.abs(depth)); z = np.abs(depth)[order]
    ct, sa = thermodynamics(temp, salt, depth, lon, lat, convention)
    sigma = gsw.sigma0(sa, ct)[order]
    return threshold_depth(sigma, z, bottom)

def threshold_depth(sigma, z, bottom, reference=10., threshold=.03):
    z = np.asarray(z)
    if np.any(np.diff(z) <= 0) or not z[0] <= reference <= z[-1]:
        raise ValueError('Ascending depths must bracket the 10 m reference')
    upper = np.searchsorted(z, reference, side='right')
    lower = upper - 1
    ref = sigma[lower] + (sigma[upper] - sigma[lower]) * (reference-z[lower])/(z[upper]-z[lower])
    target = ref + threshold
    result = np.full_like(ref, np.nan)
    valid = np.isfinite(ref) & (bottom >= reference)
    prev_rho, prev_z = ref, reference
    found = np.zeros(ref.shape, bool)
    for k in range(upper, len(z)):
        wet = z[k] <= bottom
        valid &= ~wet | np.isfinite(sigma[k])
        crossing = valid & wet & ~found & (sigma[k] >= target)
        denom = sigma[k] - prev_rho
        fraction = np.divide(target-prev_rho, denom, out=np.zeros_like(ref), where=denom != 0)
        result = np.where(crossing, prev_z + fraction * (z[k]-prev_z), result)
        found |= crossing
        prev_rho = np.where(wet, sigma[k], prev_rho)
        prev_z = np.where(wet, z[k], prev_z)
    result = np.where(valid & ~found, bottom, result)
    return np.where(valid, result, np.nan)

def seasonal(monthly, lat, winter):
    # Complete JFM/JAS means; missing months remain missing.
    north = np.mean(monthly[[0,1,2] if winter else [6,7,8]], axis=0)
    south = np.mean(monthly[[6,7,8] if winter else [0,1,2]], axis=0)
    return np.where(lat[:, None] >= 0, north, south)

def overturning(v, width, thickness, mask, basins, depth):
    """Resolved volume streamfunction, Sv, bottom integral; positive clockwise.

    Output is on the upper interface of each layer, in original depth order.
    Positive northward transport in the upper limb gives positive streamfunction.
    """
    transport = np.where(mask, v, 0) * width[None] * thickness[:, None, None]
    order = np.argsort(np.abs(depth))[::-1]  # bottom to surface
    result = []
    for basin in basins:
        layers = np.sum(np.where(basin[None], transport, 0), axis=-1)
        psi = -np.cumsum(layers[order], axis=0) / 1e6
        psi = psi[np.argsort(order)]
        active = np.any(mask & basin[None], axis=-1)
        result.append(np.where(active, psi, np.nan))
    return np.asarray(result)
