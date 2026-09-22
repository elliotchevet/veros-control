import jax
import jax.numpy as jnp
from veros.core.operators import numpy as npx, update, at

from veros_control.core.preprocessing import extract_features, restagger_uv, stack_coords


def predict_nn_forcing(
    state,
    params,
    model_batched,
    stats=None,
    normalize=False,
    scaling_method=None,
    use_forcings=False,
):
    features = extract_features(
        state,
        state.variables.tau,
        stats=stats,
        normalize=normalize,
        scaling_method=scaling_method,
        use_forcings=use_forcings,
    )

    nx, ny, nz, _ = features.shape
    features = features.reshape(-1, nz, features.shape[-1])

    coordinates = stack_coords(state.variables)
    coordinates = coordinates.reshape(-1, 2, 4)

    tendencies = model_batched(
        params,
        features,
        coordinates,
        state.variables.dzt,
    )
    tendencies = tendencies.reshape(nx, ny, nz, 4)

    mask_ghosts = npx.zeros((nx, ny, 1))
    mask_ghosts = mask_ghosts.at[2:-2, 2:-2, :].set(1.0)

    dtemp = tendencies[..., 0] * mask_ghosts * state.variables.maskT
    dsalt = tendencies[..., 1] * mask_ghosts * state.variables.maskT

    du, dv = restagger_uv(tendencies[..., 2], tendencies[..., 3])
    du = du * mask_ghosts * state.variables.maskU
    dv = dv * mask_ghosts * state.variables.maskV

    return dtemp, dsalt, du, dv


def make_batched_model(model, chunk_size=None):
    model_vmap = jax.vmap(model.apply, in_axes=(None, 0, 0, None))

    if chunk_size is None or chunk_size <= 0:
        return model_vmap

    def apply_chunked(params, features, coordinates, dzt):
        n_profiles = features.shape[0]
        pad_profiles = (-n_profiles) % chunk_size
        n_chunks = (n_profiles + pad_profiles) // chunk_size

        features_padded = jnp.pad(
            features,
            ((0, pad_profiles), (0, 0), (0, 0)),
        )
        coordinates_padded = jnp.pad(
            coordinates,
            ((0, pad_profiles), (0, 0), (0, 0)),
        )

        features_chunks = features_padded.reshape(n_chunks, chunk_size, *features.shape[1:])
        coordinate_chunks = coordinates_padded.reshape(
            n_chunks,
            chunk_size,
            *coordinates.shape[1:],
        )

        def apply_one_chunk(inputs):
            features_chunk, coordinates_chunk = inputs
            return model_vmap(params, features_chunk, coordinates_chunk, dzt)

        tendencies = jax.lax.map(apply_one_chunk, (features_chunks, coordinate_chunks))
        tendencies = tendencies.reshape(n_chunks * chunk_size, *tendencies.shape[2:])
        return tendencies[:n_profiles]

    return apply_chunked


def write_nn_tendency_diagnostics(state, dtemp, dsalt, du, dv):
    vs = state.variables
    tau = vs.tau

    vs.dtemp_nn = update(vs.dtemp_nn, at[..., tau], dtemp)
    vs.dsalt_nn = update(vs.dsalt_nn, at[..., tau], dsalt)
    vs.du_nn = update(vs.du_nn, at[..., tau], du)
    vs.dv_nn = update(vs.dv_nn, at[..., tau], dv)


def reset_nn_momentum_sources(state):
    """Clear the previous step's sources before rebuilding forcing in these setups."""
    if state.settings.enable_momentum_sources:
        vs = state.variables
        vs.u_source = update(vs.u_source, at[...], npx.zeros_like(vs.u_source))
        vs.v_source = update(vs.v_source, at[...], npx.zeros_like(vs.v_source))


def add_nn_forcing_to_state(state, dtemp, dsalt, du, dv):
    vs = state.variables
    settings = state.settings

    write_nn_tendency_diagnostics(state, dtemp, dsalt, du, dv)

    if settings.enable_tempsalt_sources:
        vs.temp_source = update(vs.temp_source, at[...], vs.temp_source + dtemp)
        if "salt_source" in state.var_meta:
            vs.salt_source = update(vs.salt_source, at[...], vs.salt_source + dsalt)

    if settings.enable_momentum_sources:
        vs.u_source = update(vs.u_source, at[...], vs.u_source + du)
        vs.v_source = update(vs.v_source, at[...], vs.v_source + dv)

    return state
