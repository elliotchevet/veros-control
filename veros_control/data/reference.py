"""GLORYS fields already remapped to the model grid; no implicit regridding."""
import h5netcdf
import numpy as np
import jax.numpy as jnp
from veros.core.operators import update, at
GLORYS_VAR_MAP = {'temp':'thetao','salt':'so','u':'uo','v':'vo','ssh':'zos'}
def _as_dtype(value, reference):
    return jnp.asarray(value, dtype=getattr(reference, "dtype", None))

def _read_glorys_field(filename, key, time_index=0):
    if key not in GLORYS_VAR_MAP:
        raise KeyError(f"No GLORYS variable mapping for loss/state key {key!r}")

    var_name = GLORYS_VAR_MAP[key]
    with h5netcdf.File(filename, "r") as infile:
        if var_name not in infile.variables:
            raise KeyError(f"{filename} has no variable {var_name!r}")
        data = np.asarray(infile.variables[var_name]).T

    if key == "ssh":
        if data.ndim == 3:
            data = data[..., time_index]
        if data.ndim != 2:
            raise ValueError(f"Expected 2D SSH field for {filename}:{var_name}, got {data.shape}")
    else:
        if data.ndim == 4:
            data = data[..., time_index]
        if data.ndim != 3:
            raise ValueError(f"Expected 3D field for {filename}:{var_name}, got {data.shape}")

    return jnp.asarray(np.nan_to_num(data, nan=0.0))

def _mask_for_state_key(state, key):
    if key in ("temp", "salt"):
        return state.variables.maskT
    if key == "u":
        return state.variables.maskU
    if key == "v":
        return state.variables.maskV
    if key == "ssh":
        return state.variables.maskT[..., -1]
    raise KeyError(f"No mask defined for GLORYS key {key!r}")

def _glorys_field_on_veros_grid(state, filename, key, time_index=0):
    raw = _read_glorys_field(filename, key, time_index=time_index)
    mask = _mask_for_state_key(state, key)
    field = jnp.zeros_like(mask, dtype=raw.dtype)
    field = field.at[2:-2, 2:-2, ...].set(raw)
    return field * mask

def load_glorys_snapshot(simulation, filename, keys, time_index=0):
    return {
        key: _glorys_field_on_veros_grid(simulation.state, filename, key, time_index=time_index)
        for key in keys
    }

def _set_scalar_like(variables, name, value):
    reference = getattr(variables, name)
    setattr(variables, name, _as_dtype(value, reference))

def _month_start_seconds(month_index, month_lengths, start_month):
    month0 = start_month - 1
    target_month = (month0 + month_index) % len(month_lengths)
    days = sum(month_lengths[:target_month])
    return days * 86400.0

def load_glorys_state(simulation, filename, time_index=0, month_index=0, config=None):
    state = simulation.state.copy()
    month_lengths = config.glorys_month_lengths if config is not None else (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    start_month = config.glorys_start_month if config is not None else 1
    forcing_shift = _month_start_seconds(month_index, month_lengths, start_month)

    with state.variables.unlock():
        vs = state.variables
        settings = state.settings

        temp_raw = _read_glorys_field(filename, "temp", time_index=time_index)
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, 0], temp_raw * vs.maskT[2:-2, 2:-2, :])
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, 1], temp_raw * vs.maskT[2:-2, 2:-2, :])

        salt_raw = _read_glorys_field(filename, "salt", time_index=time_index)
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, 0], salt_raw * vs.maskT[2:-2, 2:-2, :])
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, 1], salt_raw * vs.maskT[2:-2, 2:-2, :])

        ssh_raw = _read_glorys_field(filename, "ssh", time_index=time_index)
        vs.psi = update(
            vs.psi,
            at[2:-2, 2:-2, 0],
            ssh_raw * settings.grav * vs.maskT[2:-2, 2:-2, -1],
        )
        vs.psi = update(
            vs.psi,
            at[2:-2, 2:-2, 1],
            ssh_raw * settings.grav * vs.maskT[2:-2, 2:-2, -1],
        )

        u_raw = _read_glorys_field(filename, "u", time_index=time_index)
        vs.u = update(vs.u, at[2:-2, 2:-2, :, 0], u_raw * vs.maskU[2:-2, 2:-2, :])
        vs.u = update(vs.u, at[2:-2, 2:-2, :, 1], u_raw * vs.maskU[2:-2, 2:-2, :])

        v_raw = _read_glorys_field(filename, "v", time_index=time_index)
        vs.v = update(vs.v, at[2:-2, 2:-2, :, 0], v_raw * vs.maskV[2:-2, 2:-2, :])
        vs.v = update(vs.v, at[2:-2, 2:-2, :, 1], v_raw * vs.maskV[2:-2, 2:-2, :])

        _set_scalar_like(state.variables, "tau", 0)
        _set_scalar_like(state.variables, "taum1", 1)
        _set_scalar_like(state.variables, "taup1", 2)
        _set_scalar_like(state.variables, "itt", 0)
        _set_scalar_like(state.variables, "time", 0.0)
        if hasattr(state.variables, "forcing_time_shift"):
            _set_scalar_like(state.variables, "forcing_time_shift", forcing_shift)

    return state
