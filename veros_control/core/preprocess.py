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
    """Coordinates of T/S/U/V on their native horizontal grids: (x,y,2,4)."""
    grids = [npx.meshgrid(x,y,indexing='ij') for x,y in
             ((vs.xt,vs.yt),(vs.xt,vs.yt),(vs.xu,vs.yt),(vs.xt,vs.yu))]
    return npx.stack((npx.stack([g[0] for g in grids],axis=-1),
                      npx.stack([g[1] for g in grids],axis=-1)),axis=2)


def stack_input(vs, t_idx, mask=False):
    fields = (vs.maskT,vs.maskT,vs.maskU,vs.maskV) if mask else tuple(getattr(vs,k)[...,t_idx] for k in ('temp','salt','u','v'))
    return npx.stack(fields,axis=-1)


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
    """Face velocities to cell centers."""
    return .5*(u+npx.roll(u,1,axis=0)), .5*(v+npx.roll(v,1,axis=1))


def restagger_uv(u, v):
    """Cell-center corrections to faces; apply native masks afterward."""
    return .5*(u+npx.roll(u,-1,axis=0)), .5*(v+npx.roll(v,-1,axis=1))


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



def fit_statistics(state, method, use_forcings):
    """Fit depth/channel statistics once on the training initial state."""
    import numpy as np
    features = np.asarray(extract_features(state,state.variables.tau,use_forcings=use_forcings))
    masks = np.asarray(stack_input(state.variables,state.variables.tau,mask=True),dtype=bool)
    if use_forcings:
        masks = np.concatenate((masks,np.asarray(stack_forcing_input(state.variables,mask=True),dtype=bool)),axis=-1)
    masks[:2] = False; masks[-2:] = False; masks[:,:2] = False; masks[:,-2:] = False
    if not np.isfinite(features[masks]).all():
        raise ValueError('Nonfinite wet-cell training features')
    counts = masks.sum(axis=(0,1))
    safe = np.maximum(counts,1)
    values = np.where(masks,features,0.)
    if method == 'meanstd':
        mean = values.sum(axis=(0,1))/safe
        variance = np.where(masks,(features-mean)**2,0.).sum(axis=(0,1))/safe
        return npx.asarray(mean),npx.asarray(np.sqrt(variance))
    if method == 'minmax':
        lower = np.where(masks,features,np.inf).min(axis=(0,1))
        upper = np.where(masks,features,-np.inf).max(axis=(0,1))
        return npx.asarray(np.where(counts,lower,0.)),npx.asarray(np.where(counts,upper,0.))
    raise ValueError('Statistics require meanstd or minmax')
