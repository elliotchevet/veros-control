"""Host-side, area-weighted basin/level diagnostics; no model or checkpoint imports."""
import numpy as np


def basin_stats(field, area, wet, basins):
    """Return signed mean, RMS and active area, shape (basin, depth)."""
    mean, rms, areas = [], [], []
    for basin in basins:
        weights = np.where(wet & basin[..., None], area[..., None], 0.)
        denom = weights.sum(axis=(0, 1))
        values = np.where(weights > 0, field, 0.)
        total = np.maximum(denom, 1.)
        mean.append(np.where(denom > 0, (values * weights).sum(axis=(0, 1)) / total, np.nan))
        rms.append(np.where(denom > 0, np.sqrt((values**2 * weights).sum(axis=(0, 1)) / total), np.nan))
        areas.append(denom)
    return np.asarray(mean), np.asarray(rms), np.asarray(areas)


def objective(bias, indices, denominators, rmse=None, pattern_lambda=0.):
    """Equal active basin-level weights, then equal normalized tracer weights."""
    if not np.isfinite(pattern_lambda) or pattern_lambda < 0:
        raise ValueError('Invalid pattern weight')
    def squared_error(k):
        bias2 = bias[k][:, indices] ** 2
        if pattern_lambda == 0:
            return bias2
        if rmse is None:
            raise ValueError('Spatial RMSE is required for the pattern objective')
        mse = rmse[k][:, indices] ** 2
        return mse if pattern_lambda == 1 else bias2 + pattern_lambda * np.maximum(mse - bias2, 0.)
    terms = np.array([np.nanmean(squared_error(k)) for k in ('temp', 'salt')])
    normalized = terms / np.array([denominators[k] for k in ('temp', 'salt')])
    return terms, float(normalized.mean())
