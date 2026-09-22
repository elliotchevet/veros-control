from functools import partial
import os

import h5netcdf
import jax

import jax.numpy as jnp
import numpy as np
import optax
from veros import restart

from veros_control.core.checkpointing import load_checkpoint, save_checkpoint
from veros_control.core.forcing import make_batched_model
from veros_control.core.model import VerticalCorrectionMLP
from veros_control.core.preprocessing import FORCING_FEATURE_NAMES
from veros_control.training.config import TrainConfig
from veros_control.training.loss import compute_loss
from veros_control.training.trainset import load_snapshot
from veros.core.operators import update, at
from veros.state import veros_state_pytree_flatten, veros_variables_pytree_flatten
from veros.state import veros_state_pytree_unflatten, veros_variables_pytree_unflatten


RESTART_SKIP_VARS = {
    "qnet",
    "qsol",
    "taux",
    "tauy",
    "evap",
    "tp",
    "forcing_time_shift",
    "divpen_shortwave",
    "t_star",
    "s_star",
}

MEMORY_PROFILE_DIR = os.environ.get(
    "JAX_MEMORY_PROFILE_DIR",
    "/Odyssey/private/e25cheve/jax_traces",
)
IGNORED_INPUT_SUFFIXES = (".lock",)

GLORYS_VAR_MAP = {
    "temp": "thetao",
    "salt": "so",
    "ssh": "zos",
    "u": "uo",
    "v": "vo",
}


def create_optimizer(lr, grad_clip_norm=None):
    if grad_clip_norm is None:
        return optax.adamw(lr)
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(lr),
    )


def disable_diagnostics(simulation):
    for diagnostic in simulation.state.diagnostics.values():
        diagnostic.sampling_frequency = None
        diagnostic.output_frequency = None


def zero_tendency_sources(state):
    with state.variables.unlock():
        for source_name in ("temp_source", "salt_source", "u_source", "v_source"):
            if source_name not in state.var_meta:
                continue
            try:
                source_value = getattr(state.variables, source_name)
            except RuntimeError as exc:
                if "not active" in str(exc):
                    continue
                raise
            setattr(state.variables, source_name, jnp.zeros_like(source_value))


def load_restart_state(simulation, restart_file):
    with simulation.state.settings.unlock():
        simulation.state.settings.restart_input_filename = restart_file

    skipped_meta = {
        key: simulation.state.var_meta.pop(key)
        for key in RESTART_SKIP_VARS
        if key in simulation.state.var_meta
    }
    try:
        restart.read_restart(simulation.state)
    finally:
        simulation.state.var_meta.update(skipped_meta)

    return simulation.state


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


def make_model(config, nz):
    return VerticalCorrectionMLP(
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        nz=nz,
        nb_profiles=4,
        init_eps=config.epsilon,
        train_epsilon=config.train_epsilon,
        use_lon_lat=config.use_lon_lat,
        use_forcings=config.use_forcings,
        conservative_mixing=config.conservative_mixing,
        use_one_hot=config.use_one_hot,
        num_mlps=config.num_mlps,
        TS=config.TS,
        zero_output_init=config.zero_output_init,
    )


def init_params(model, nz, seed):
    n_input_channels = 4 + len(FORCING_FEATURE_NAMES) if model.use_forcings else 4
    x_tmp = jnp.zeros((nz, n_input_channels))
    coords_tmp = jnp.zeros((2, 4))
    dz_tmp = jnp.zeros((nz,))
    return model.init(jax.random.PRNGKey(seed), x_tmp, coords_tmp, dz_tmp)


def init_or_resume_training(model, optimizer, config, nz):
    checkpoint_path = config.resume_checkpoint_path or config.output_file
    checkpoint = None

    if config.resume_training or not config.overwrite_training:
        checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint and not config.overwrite_training:
        print(f"Resuming training from epoch {checkpoint['epoch'] + 1}")
        return (
            checkpoint["params"],
            checkpoint["opt_state"],
            checkpoint["epoch"] + 1,
            list(checkpoint.get("losses", [])),
        )

    params = init_params(model, nz, config.seed)
    opt_state = optimizer.init(params)
    print(f"Starting new training with seed {config.seed}")
    return params, opt_state, 0, []


def init_or_resume_A_h_training(optimizer, config, simulation):
    checkpoint_path = config.resume_checkpoint_path or config.output_file
    checkpoint = None

    if config.resume_training or not config.overwrite_training:
        checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint and not config.overwrite_training:
        print(f"Resuming A_h training from epoch {checkpoint['epoch'] + 1}")
        return (
            checkpoint["params"],
            checkpoint["opt_state"],
            checkpoint["epoch"] + 1,
            list(checkpoint.get("losses", [])),
        )

    if config.A_h_init is not None:
        A_h_init = config.A_h_init
    else:
        A_h_init = getattr(
            simulation.state.variables,
            "A_h",
            simulation.state.settings.A_h,
        )

    params = {"A_h": jnp.asarray(A_h_init)}
    opt_state = optimizer.init(params)
    print(f"Starting new A_h training from A_h={float(params['A_h']):.12g}")
    return params, opt_state, 0, []


def pure(state, step):
    next_state = state.copy()
    step(next_state)
    return next_state


def nn_set_forcing_step(
    state,
    params,
    simulation,
    model_batched,
    stats=None,
    normalize=False,
    scaling_method=None,
    use_forcings=False,
):
    next_state = state.copy()

    # Traced parameters must not escape through persistent simulation attributes.
    nn_fields = ("_nn_correction_initialized", "_nn_params", "_nn_model_batched", "_nn_stats")
    previous_nn = tuple(getattr(simulation, name, None) for name in nn_fields)
    try:
        simulation._nn_correction_initialized = True
        simulation._nn_params = params
        simulation._nn_model_batched = model_batched
        simulation._nn_stats = stats

        with next_state.settings.unlock():
            next_state.settings.baseline_model = False
            next_state.settings.normalize_inputs = normalize
            next_state.settings.use_forcings = use_forcings
            if scaling_method is not None:
                next_state.settings.scaling_method_features = scaling_method

        zero_tendency_sources(next_state)

        simulation.step(next_state)
    finally:
        for name, value in zip(nn_fields, previous_nn):
            setattr(simulation, name, value)
    return next_state


def baseline_physics_step(state, simulation):
    next_state = state

    with next_state.settings.unlock():
        next_state.settings.baseline_model = True

    zero_tendency_sources(next_state)

    simulation.step(next_state)
    return next_state


def A_h_set_forcing_step(state, params, step_function):
    next_state = state.copy()
    with next_state.variables.unlock():
        setattr(next_state.variables, "A_h", params["A_h"])

    with next_state.settings.unlock():
        next_state.settings.baseline_model = True

    return step_function(next_state)


def maybe_checkpoint(fn, enabled):
    if not enabled:
        return fn

    try:
        return jax.checkpoint(fn, policy=jax.checkpoint_policies.nothing_saveable)
    except AttributeError:
        return jax.checkpoint(fn)


def make_chunk_sizes(iterations, base=3):
    n = iterations
    chunks = []
    while n > 1:
        if n % base == 0:
            chunks.append(base)
            n //= base
        else:
            chunks.append(n)
            n = 1
    if not chunks:
        chunks = [iterations]
    return tuple(chunks)


def get_prognostic_keys(state):
    prognostic_keys = ["tau", "taum1", "taup1", "itt", "time"]
    for key, meta in state.var_meta.items():
        if meta.active and meta.dims is not None:
            if "timesteps" in meta.dims or meta.write_to_restart:
                if key not in prognostic_keys:
                    prognostic_keys.append(key)
    return prognostic_keys


def inject_dynamic_vars(base_state, dynamic_updates):
    var_leaves, var_aux = veros_variables_pytree_flatten(base_state.variables)
    var_keys = var_aux[0]
    new_leaves = [
        dynamic_updates[key] if key in dynamic_updates else leaf
        for key, leaf in zip(var_keys, var_leaves)
    ]
    new_vars = veros_variables_pytree_unflatten(var_aux, new_leaves)

    _, state_aux = veros_state_pytree_flatten(base_state)
    return veros_state_pytree_unflatten(state_aux, [new_vars])


def forward_unroll(
    params,
    state_init,
    targets,
    model_batched,
    simulation,
    keys,
    n_steps,
    sample_steps,
    use_jax_checkpoint=True,
    stats=None,
    normalize=False,
    scaling_method=None,
    use_forcings=False,
    control_mode="nn",
    step_function=None,
):
    state = state_init.copy()
    losses = []
    target_idx = 0

    if control_mode == "nn":
        step_with_forcing = partial(
            nn_set_forcing_step,
            simulation=simulation,
            model_batched=model_batched,
            stats=stats,
            normalize=normalize,
            scaling_method=scaling_method,
            use_forcings=use_forcings,
        )
    elif control_mode == "A_h":
        if step_function is None:
            raise ValueError("step_function is required for control_mode='A_h'")
        step_with_forcing = partial(
            A_h_set_forcing_step,
            step_function=step_function,
        )
    else:
        raise ValueError(f"Unsupported control_mode: {control_mode}")

    step_with_forcing = maybe_checkpoint(step_with_forcing, use_jax_checkpoint)

    for step_idx in range(n_steps):
        state = step_with_forcing(state, params)

        if (step_idx + 1) % sample_steps == 0:
            losses.append(
                compute_loss(
                    state,
                    targets[target_idx],
                    keys=keys,
                    tau=state.variables.tau,
                )
            )
            target_idx += 1

    return jnp.mean(jnp.stack(losses))


def rollout_state(
    params,
    state_init,
    model_batched,
    simulation,
    config,
    step_function=None,
    n_steps=None,
):
    state = state_init.copy()

    if config.control_mode == "nn":
        step_with_forcing = partial(
            nn_set_forcing_step,
            simulation=simulation,
            model_batched=model_batched,
            stats=config.stats_features,
            normalize=config.normalize_inputs,
            scaling_method=config.scaling_method_features,
        )
    elif config.control_mode == "A_h":
        if step_function is None:
            raise ValueError("step_function is required for control_mode='A_h'")
        step_with_forcing = partial(
            A_h_set_forcing_step,
            step_function=step_function,
        )
    else:
        raise ValueError(f"Unsupported control_mode: {config.control_mode}")

    step_with_forcing = maybe_checkpoint(step_with_forcing, config.use_jax_checkpoint)
    n_steps = config.n_steps if n_steps is None else n_steps
    for _ in range(n_steps):
        state = step_with_forcing(state, params)
    return state


def rollout_train_delay_state(
    params,
    state_init,
    model_batched,
    simulation,
    config,
    step_function=None,
    n_steps=0,
):
    if config.train_delay_use_nn:
        return rollout_state(
            params,
            state_init,
            model_batched,
            simulation,
            config,
            step_function=step_function,
            n_steps=n_steps,
        )

    state = state_init
    for step_idx in range(n_steps):
        state = baseline_physics_step(state, simulation)
        if (step_idx + 1) % 24 == 0:
            jax.block_until_ready(state)
    return state


def train_step(
    params,
    state_init,
    targets,
    model_batched,
    simulation,
    config,
    step_function=None,
):
    loss_fn = partial(
        forward_unroll,
        state_init=state_init,
        targets=targets,
        model_batched=model_batched,
        simulation=simulation,
        keys=config.keys_loss,
        n_steps=config.n_steps,
        sample_steps=config.sample_steps,
        use_jax_checkpoint=config.use_jax_checkpoint,
        stats=config.stats_features,
        normalize=config.normalize_inputs,
        scaling_method=config.scaling_method_features,
        use_forcings=config.use_forcings,
        control_mode=config.control_mode,
        step_function=step_function,
    )

    return jax.value_and_grad(loss_fn)(params)


def jvp_train_step(
    params,
    state_init,
    targets,
    model_batched,
    simulation,
    config,
    step_function=None,
):
    if config.control_mode != "A_h":
        raise ValueError("grad_mode='forward_jvp' is only supported for control_mode='A_h'")

    loss_fn = partial(
        forward_unroll,
        state_init=state_init,
        targets=targets,
        model_batched=model_batched,
        simulation=simulation,
        keys=config.keys_loss,
        n_steps=config.n_steps,
        sample_steps=config.sample_steps,
        use_jax_checkpoint=config.use_jax_checkpoint,
        stats=config.stats_features,
        normalize=config.normalize_inputs,
        scaling_method=config.scaling_method_features,
        use_forcings=config.use_forcings,
        control_mode=config.control_mode,
        step_function=step_function,
    )

    tangent = {"A_h": jnp.ones_like(params["A_h"])}
    loss, grad_A_h = jax.jvp(loss_fn, (params,), (tangent,))
    return loss, {"A_h": grad_A_h}


def _zero_like_params(params):
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def _floating_leaves(tree):
    leaves = []
    for leaf in jax.tree_util.tree_leaves(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or dtype == jax.dtypes.float0:
            continue
        if jnp.issubdtype(dtype, jnp.inexact):
            leaves.append(jnp.asarray(leaf))
    return leaves


def _tree_stats(tree):
    leaves = _floating_leaves(tree)
    if not leaves:
        return {"finite": True, "norm": 0.0, "max_abs": 0.0}

    finite = jnp.array(True)
    norm_sq = jnp.array(0.0)
    max_abs = jnp.array(0.0)
    for leaf in leaves:
        abs_leaf = jnp.abs(leaf)
        finite = finite & jnp.all(jnp.isfinite(leaf))
        norm_sq = norm_sq + jnp.sum(jnp.where(jnp.isfinite(abs_leaf), abs_leaf * abs_leaf, 0.0))
        max_abs = jnp.maximum(max_abs, jnp.max(jnp.where(jnp.isfinite(abs_leaf), abs_leaf, jnp.inf)))

    finite, norm, max_abs = jax.device_get((finite, jnp.sqrt(norm_sq), max_abs))
    return {
        "finite": bool(finite),
        "norm": float(norm),
        "max_abs": float(max_abs),
    }


def _tree_path_name(path):
    if hasattr(jax.tree_util, "keystr"):
        return jax.tree_util.keystr(path)
    return "/".join(str(entry) for entry in path)


def _bad_tree_leaf_summaries(tree, limit=8):
    summaries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or dtype == jax.dtypes.float0:
            continue
        if not jnp.issubdtype(dtype, jnp.inexact):
            continue

        arr = jnp.asarray(leaf)
        finite = jnp.isfinite(arr)
        all_finite = bool(jax.device_get(jnp.all(finite)))
        if all_finite:
            continue

        bad_count, has_nan, has_pos_inf, has_neg_inf = jax.device_get((
            jnp.sum(~finite),
            jnp.any(jnp.isnan(arr)),
            jnp.any(arr == jnp.inf),
            jnp.any(arr == -jnp.inf),
        ))
        flags = []
        if bool(has_nan):
            flags.append("nan")
        if bool(has_pos_inf):
            flags.append("+inf")
        if bool(has_neg_inf):
            flags.append("-inf")
        summaries.append(
            f"{_tree_path_name(path)} shape={getattr(arr, 'shape', ())} "
            f"bad={int(bad_count)}/{arr.size} ({','.join(flags)})"
        )
        if len(summaries) >= limit:
            break
    return summaries


def _state_variable_stats(state, names):
    vs = state.variables
    pieces = []
    all_finite = True
    for name in dict.fromkeys(names):
        if not hasattr(vs, name):
            continue
        value = getattr(vs, name)
        dtype = getattr(value, "dtype", None)
        if dtype is None or not jnp.issubdtype(dtype, jnp.inexact):
            continue
        finite, max_abs = jax.device_get((
            jnp.all(jnp.isfinite(value)),
            jnp.max(jnp.where(jnp.isfinite(value), jnp.abs(value), jnp.inf)),
        ))
        finite = bool(finite)
        all_finite = all_finite and finite
        status = "ok" if finite else "bad"
        pieces.append(f"{name}:{status} max={float(max_abs):.3e}")

    return all_finite, ", ".join(pieces)


def _training_diagnostic_keys(config):
    return (
        tuple(config.keys_loss)
        + ("temp", "salt", "u", "v", "w", "rho", "dtemp_nn", "dsalt_nn", "du_nn", "dv_nn")
    )


def _print_update_diagnostics(label, loss, grads, updates, params_after, next_state, config):
    loss_finite = bool(jax.device_get(jnp.isfinite(loss)))
    grad_stats = _tree_stats(grads)
    update_stats = _tree_stats(updates)
    param_stats = _tree_stats(params_after)
    state_finite, state_text = _state_variable_stats(next_state, _training_diagnostic_keys(config))

    print(
        f"  [{label}] "
        f"loss_finite={loss_finite} "
        f"grad_finite={grad_stats['finite']} grad_norm={grad_stats['norm']:.3e} grad_max={grad_stats['max_abs']:.3e} "
        f"update_finite={update_stats['finite']} update_norm={update_stats['norm']:.3e} update_max={update_stats['max_abs']:.3e} "
        f"params_finite={param_stats['finite']} params_norm={param_stats['norm']:.3e} params_max={param_stats['max_abs']:.3e} "
        f"state_finite={state_finite} {state_text}"
    )

    if not grad_stats["finite"]:
        print("    bad grad leaves: " + "; ".join(_bad_tree_leaf_summaries(grads)))
    if not update_stats["finite"]:
        print("    bad update leaves: " + "; ".join(_bad_tree_leaf_summaries(updates)))
    if not param_stats["finite"]:
        print("    bad param leaves: " + "; ".join(_bad_tree_leaf_summaries(params_after)))

    bad = []
    if not loss_finite:
        bad.append("loss")
    if not grad_stats["finite"]:
        bad.append("grads")
    if not update_stats["finite"]:
        bad.append("updates")
    if not param_stats["finite"]:
        bad.append("params_after_update")
    if not state_finite:
        bad.append("next_state")

    if bad and config.fail_on_nonfinite:
        raise FloatingPointError(
            f"Non-finite values detected after {label}: {', '.join(bad)}. "
            "Try lowering lr and/or epsilon, or set grad_clip_norm."
        )


def _add_optional_trees(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return jax.tree_util.tree_map(_add_cotangent, left, right)


def _add_cotangent(left, right):
    if getattr(left, "dtype", None) == jax.dtypes.float0:
        return right
    if getattr(right, "dtype", None) == jax.dtypes.float0:
        return left
    return left + right


def _forward_to_step(
    state_init,
    params,
    step_fn,
    n_steps,
):
    state = state_init.copy()
    for _ in range(n_steps):
        state = step_fn(state, params)
    return state


def save_memory_profile(name):
    os.makedirs(MEMORY_PROFILE_DIR, exist_ok=True)
    jax.profiler.save_device_memory_profile(
        os.path.join(MEMORY_PROFILE_DIR, name)
    )


def manual_vjp_train_step(
    params,
    state_init,
    targets,
    model_batched,
    simulation,
    config,
    step_function=None,
):
    if config.control_mode == "nn":
        step_fn = partial(
            nn_set_forcing_step,
            simulation=simulation,
            model_batched=model_batched,
            stats=config.stats_features,
            normalize=config.normalize_inputs,
            scaling_method=config.scaling_method_features,
            use_forcings=config.use_forcings,
        )
    elif config.control_mode == "A_h":
        if step_function is None:
            raise ValueError("step_function is required for control_mode='A_h'")
        step_fn = partial(
            A_h_set_forcing_step,
            step_function=step_function,
        )
    else:
        raise ValueError(f"Unsupported control_mode: {config.control_mode}")

    step_fn = maybe_checkpoint(step_fn, config.use_jax_checkpoint)

    sample_steps = tuple(range(config.sample_steps, config.n_steps + 1, config.sample_steps))
    loss_vjps = {}
    losses = []

    state = state_init.copy()
    target_idx = 0
    for step_idx in range(1, config.n_steps + 1):
        state = step_fn(state, params)
        if step_idx in sample_steps:
            loss_fn = partial(
                compute_loss,
                target=targets[target_idx],
                keys=config.keys_loss,
                tau=state.variables.tau,
            )
            loss_value, loss_vjp = jax.vjp(loss_fn, state)
            loss_vjps[step_idx] = loss_vjp
            losses.append(loss_value)
            target_idx += 1

    total_loss = jnp.mean(jnp.stack(losses))
    grad_params = _zero_like_params(params)
    state_cotangent = None

    for step_idx in range(config.n_steps, 0, -1):
        if step_idx in loss_vjps:
            loss_state_cotangent, = loss_vjps[step_idx](1.0 / len(sample_steps))
            state_cotangent = _add_optional_trees(state_cotangent, loss_state_cotangent)

        if state_cotangent is None:
            continue

        state_before = _forward_to_step(
            state_init,
            params,
            step_fn,
            step_idx - 1,
        )
        _, step_vjp = jax.vjp(step_fn, state_before, params)
        state_cotangent, param_cotangent = step_vjp(state_cotangent)
        jax.block_until_ready((state_cotangent, param_cotangent))
        save_memory_profile(f"after_apply_step_vjp_{step_idx}.prof")
        grad_params = jax.tree_util.tree_map(_add_cotangent, grad_params, param_cotangent)

    return total_loss, grad_params


def compute_train_step(
    params,
    state_init,
    targets,
    model_batched,
    simulation,
    config,
    step_function=None,
):
    if config.grad_mode == "forward_jvp":
        return jvp_train_step(
            params,
            state_init,
            targets,
            model_batched,
            simulation,
            config,
            step_function=step_function,
        )
    if config.grad_mode == "builtin":
        return train_step(
            params,
            state_init,
            targets,
            model_batched,
            simulation,
            config,
            step_function=step_function,
        )
    if config.grad_mode == "manual_vjp":
        return manual_vjp_train_step(
            params,
            state_init,
            targets,
            model_batched,
            simulation,
            config,
            step_function=step_function,
        )
    if config.grad_mode == "chunked_vjp":
        if config.control_mode == "A_h":
            return chunked_vjp_A_h_train_step(
                params,
                state_init,
                targets,
                model_batched,
                simulation,
                config,
                step_function=step_function,
            )
        if config.control_mode == "nn":
            return chunked_vjp_nn_train_step(
                params,
                state_init,
                targets,
                model_batched,
                simulation,
                config,
            )
        raise ValueError("grad_mode='chunked_vjp' supports control_mode='A_h' or 'nn'")

    raise ValueError(f"Unsupported grad_mode: {config.grad_mode}")


def _target_files_for_window(restart_files, start_idx, n_targets, target_stride):
    return [
        restart_files[start_idx + (target_idx + 1) * target_stride]
        for target_idx in range(n_targets)
    ]


def _load_targets(target_files, keys):
    return [load_snapshot(filename, keys) for filename in target_files]


def _stack_targets(targets, keys):
    return {
        key: jnp.stack([jnp.asarray(target[key]) for target in targets])
        for key in keys
    }


def _target_from_stacked(stacked_targets, target_idx):
    return {
        key: values[target_idx]
        for key, values in stacked_targets.items()
    }


def chunked_vjp_A_h_train_step(
    params,
    state_init,
    targets,
    model_batched,
    simulation,
    config,
    step_function=None,
):
    if config.control_mode != "A_h":
        raise ValueError("grad_mode='chunked_vjp' is only supported for control_mode='A_h'")
    if step_function is None:
        raise ValueError("step_function is required for control_mode='A_h'")

    del model_batched

    full_state = state_init.copy()
    prognostic_keys = get_prognostic_keys(full_state)
    chunk_sizes = make_chunk_sizes(config.n_steps, base=config.chunk_base)
    stacked_targets = _stack_targets(targets, config.keys_loss)
    n_targets = max(config.n_steps // config.sample_steps, 1)

    def step_with_A_h(curr_state, A_h_value):
        next_state = curr_state.copy()
        with next_state.variables.unlock():
            setattr(next_state.variables, "A_h", A_h_value)
        with next_state.settings.unlock():
            next_state.settings.baseline_model = True
        return step_function(next_state)

    step_with_A_h = maybe_checkpoint(step_with_A_h, config.use_jax_checkpoint)

    def forward_trajectory(A_h_value):
        init_carry = {
            key: getattr(full_state.variables, key)
            for key in prognostic_keys
        }
        is_prognostic = "A_h" in prognostic_keys
        if is_prognostic:
            init_carry["A_h"] = A_h_value

        @jax.checkpoint
        def innermost_step(inner_tuple, step_idx):
            dynamic_vars, step_A_h = inner_tuple

            updates = dict(dynamic_vars)
            if not is_prognostic:
                updates["A_h"] = step_A_h

            curr_state = inject_dynamic_vars(full_state, updates)
            next_state = step_with_A_h(curr_state, step_A_h)
            next_carry = {
                key: getattr(next_state.variables, key)
                for key in prognostic_keys
            }

            is_sample_step = (step_idx % config.sample_steps) == 0
            target_idx = jnp.maximum(step_idx // config.sample_steps - 1, 0)
            target_idx = jnp.minimum(target_idx, n_targets - 1)

            def sampled_loss(state):
                target = _target_from_stacked(stacked_targets, target_idx)
                return compute_loss(
                    state,
                    target,
                    keys=config.keys_loss,
                    tau=state.variables.tau,
                )

            step_loss = jax.lax.cond(
                is_sample_step,
                sampled_loss,
                lambda state: jnp.asarray(0.0),
                next_state,
            )
            return (next_carry, step_A_h), step_loss

        def make_chunk_func(func_to_scan, length):
            def chunk_def(carry, xs_chunk):
                out, chunk_losses = jax.lax.scan(
                    func_to_scan,
                    carry,
                    xs=xs_chunk,
                    length=length,
                    unroll=1,
                )
                return out, chunk_losses

            return maybe_checkpoint(chunk_def, config.use_jax_checkpoint)

        current_func = innermost_step
        for size in reversed(chunk_sizes[1:]):
            current_func = make_chunk_func(current_func, size)

        step_indices = jnp.arange(1, config.n_steps + 1).reshape(chunk_sizes)
        _, all_chunk_losses = jax.lax.scan(
            current_func,
            (init_carry, A_h_value),
            xs=step_indices,
            length=chunk_sizes[0],
            unroll=1,
        )

        return jnp.sum(all_chunk_losses) / n_targets

    loss, grad_A_h = jax.value_and_grad(forward_trajectory)(params["A_h"])
    return loss, {"A_h": grad_A_h}


def chunked_vjp_nn_train_step(
    params,
    state_init,
    targets,
    model_batched,
    simulation,
    config,
):
    if config.control_mode != "nn":
        raise ValueError("chunked_vjp_nn_train_step requires control_mode='nn'")
    if model_batched is None:
        raise ValueError("model_batched is required for control_mode='nn'")

    full_state = state_init.copy()
    dynamic_keys = get_prognostic_keys(full_state)
    chunk_sizes = make_chunk_sizes(config.n_steps, base=config.chunk_base)
    stacked_targets = _stack_targets(targets, config.keys_loss)
    n_targets = max(config.n_steps // config.sample_steps, 1)

    step_with_nn = partial(
        nn_set_forcing_step,
        simulation=simulation,
        model_batched=model_batched,
        stats=config.stats_features,
        normalize=config.normalize_inputs,
        scaling_method=config.scaling_method_features,
        use_forcings=config.use_forcings,
    )
    step_with_nn = maybe_checkpoint(step_with_nn, config.use_jax_checkpoint)

    def forward_trajectory(nn_params):
        init_carry = {
            key: getattr(full_state.variables, key)
            for key in dynamic_keys
        }

        @jax.checkpoint
        def innermost_step(dynamic_vars, step_idx):
            curr_state = inject_dynamic_vars(full_state, dict(dynamic_vars))
            next_state = step_with_nn(curr_state, nn_params)
            next_carry = {
                key: getattr(next_state.variables, key)
                for key in dynamic_keys
            }

            is_sample_step = (step_idx % config.sample_steps) == 0
            target_idx = jnp.maximum(step_idx // config.sample_steps - 1, 0)
            target_idx = jnp.minimum(target_idx, n_targets - 1)

            def sampled_loss(state):
                target = _target_from_stacked(stacked_targets, target_idx)
                return compute_loss(
                    state,
                    target,
                    keys=config.keys_loss,
                    tau=state.variables.tau,
                )

            step_loss = jax.lax.cond(
                is_sample_step,
                sampled_loss,
                lambda state: jnp.asarray(0.0),
                next_state,
            )
            return next_carry, step_loss

        def make_chunk_func(func_to_scan, length):
            def chunk_def(carry, xs_chunk):
                out, chunk_losses = jax.lax.scan(
                    func_to_scan,
                    carry,
                    xs=xs_chunk,
                    length=length,
                    unroll=1,
                )
                return out, chunk_losses

            return maybe_checkpoint(chunk_def, config.use_jax_checkpoint)

        current_func = innermost_step
        for size in reversed(chunk_sizes[1:]):
            current_func = make_chunk_func(current_func, size)

        step_indices = jnp.arange(1, config.n_steps + 1).reshape(chunk_sizes)
        _, all_chunk_losses = jax.lax.scan(
            current_func,
            init_carry,
            xs=step_indices,
            length=chunk_sizes[0],
            unroll=1,
        )

        return jnp.sum(all_chunk_losses) / n_targets

    return jax.value_and_grad(forward_trajectory)(params)


def save_training_checkpoint(path, params, opt_state, epoch, losses, config):
    save_checkpoint(
        path,
        {
            "params": params,
            "opt_state": opt_state,
            "epoch": epoch,
            "losses": losses,
            "lr": config.lr,
            "grad_clip_norm": config.grad_clip_norm,
            "diagnostics_every": config.diagnostics_every,
            "fail_on_nonfinite": config.fail_on_nonfinite,
            "hidden_dim_nn": config.hidden_dim,
            "depth_nn": config.depth,
            "epsilon": config.epsilon,
            "train_epsilon": config.train_epsilon,
            "use_lon_lat": config.use_lon_lat,
            "use_forcings": config.use_forcings,
            "conservative_mixing": config.conservative_mixing,
            "use_one_hot": config.use_one_hot,
            "num_mlps": config.num_mlps,
            "TS": config.TS,
            "zero_output_init": config.zero_output_init,
            "stats_features": config.stats_features,
            "normalize_inputs": config.normalize_inputs,
            "scaling_method_features": config.scaling_method_features,
            "control_mode": config.control_mode,
            "A_h_init": config.A_h_init,
            "grad_mode": config.grad_mode,
            "chunk_base": config.chunk_base,
            "train_delay_days": config.train_delay_days,
            "train_delay_use_nn": config.train_delay_use_nn,
            "initialization_mode": config.initialization_mode,
            "glorys_files": config.glorys_files,
            "glorys_time_index": config.glorys_time_index,
            "glorys_start_month": config.glorys_start_month,
            "glorys_month_lengths": config.glorys_month_lengths,
            "glorys_rollouts_per_month": config.glorys_rollouts_per_month,
        },
    )


def _training_files_for_mode(config):
    if config.initialization_mode == "restart":
        return _filter_training_files(config.restart_files)
    if config.initialization_mode == "glorys":
        return _filter_training_files(config.glorys_files or config.restart_files)
    raise ValueError(f"Unsupported initialization_mode: {config.initialization_mode}")


def _filter_training_files(files):
    files = list(files or [])
    return [
        filename
        for filename in files
        if not str(filename).endswith(IGNORED_INPUT_SUFFIXES)
    ]


def _n_targets_per_rollout(config):
    n_targets = config.n_steps // config.sample_steps
    if n_targets < 1:
        raise ValueError("n_steps must be at least sample_steps")
    return n_targets


def _steps_from_days(simulation, days, label):
    if days is None or days <= 0:
        return 0
    dt = float(simulation.state.settings.dt_tracer)
    raw_steps = days * 86400.0 / dt
    rounded = round(raw_steps)
    if not np.isclose(raw_steps, rounded):
        raise ValueError(
            f"{label}={days} days is not an integer number of tracer steps "
            f"for dt_tracer={dt}: got {raw_steps} steps"
        )
    return int(rounded)


def _month_steps(simulation, config, month_index):
    month0 = config.glorys_start_month - 1
    month_days = config.glorys_month_lengths[
        (month0 + month_index) % len(config.glorys_month_lengths)
    ]
    raw_steps = month_days * 86400.0 / float(simulation.state.settings.dt_tracer)
    rounded = round(raw_steps)
    if not np.isclose(raw_steps, rounded):
        raise ValueError(
            f"Month {month_index} is not an integer number of tracer steps: got {raw_steps}"
        )
    return int(rounded)


def _glorys_rollouts_for_month(simulation, config, month_index, train_delay_steps=0):
    if config.glorys_rollouts_per_month is not None:
        return config.glorys_rollouts_per_month

    month_steps = _month_steps(simulation, config, month_index)
    remaining_steps = month_steps - train_delay_steps
    if remaining_steps <= 0:
        raise ValueError(
            f"train_delay_days={config.train_delay_days} leaves no training steps "
            f"for month_index={month_index} ({month_steps} total steps)"
        )

    ratio = remaining_steps / config.n_steps
    rounded = round(ratio)
    if not np.isclose(ratio, rounded):
        raise ValueError(
            "Could not infer an integer number of GLORYS rollouts for month "
            f"{month_index} after train_delay_days={config.train_delay_days}: "
            f"remaining_steps={remaining_steps}, rollout_steps={config.n_steps}. "
            "Set glorys_rollouts_per_month explicitly."
        )
    return int(rounded)


def _apply_training_runtime_settings(simulation):
    with simulation.state.settings.unlock():
        simulation.state.settings.enable_tke = False
        simulation.state.settings.kappaH_0 = 0.06
        simulation.state.settings.kappaM_0 = 0.06


def _initialize_training_state(simulation, config, filename, month_index=0):
    if config.initialization_mode == "restart":
        return load_restart_state(simulation, filename)
    if config.initialization_mode == "glorys":
        return load_glorys_state(
            simulation,
            filename,
            time_index=config.glorys_time_index,
            month_index=month_index,
            config=config,
        )
    raise ValueError(f"Unsupported initialization_mode: {config.initialization_mode}")


def train_glorys_monthly(
    simulation,
    config,
    params,
    opt_state,
    start_epoch,
    losses,
    model_batched,
    optimizer,
    step_function=None,
):
    glorys_files = _training_files_for_mode(config)
    if len(glorys_files) < 2:
        raise ValueError("GLORYS training requires at least two monthly files")

    if config.sample_steps <= 0:
        raise ValueError("TrainConfig.sample_steps must be positive")

    n_targets = _n_targets_per_rollout(config)
    n_pairs = len(glorys_files) - 1
    print(
        "Using GLORYS monthly initialization mode: "
        f"{n_pairs} monthly pairs, {n_targets} target evaluations per rollout"
    )

    for epoch in range(start_epoch, config.epochs):
        epoch_losses = []

        for month_idx in range(n_pairs):
            source_file = glorys_files[month_idx]
            target_file = glorys_files[month_idx + 1]
            state = _initialize_training_state(
                simulation,
                config,
                source_file,
                month_index=month_idx,
            )
            target = load_glorys_snapshot(
                simulation,
                target_file,
                config.keys_loss,
                time_index=config.glorys_time_index,
            )
            targets = [target] * n_targets
            train_delay_steps = _steps_from_days(
                simulation,
                config.train_delay_days,
                "train_delay_days",
            )
            n_rollouts = _glorys_rollouts_for_month(
                simulation,
                config,
                month_idx,
                train_delay_steps=train_delay_steps,
            )

            print(
                f"Epoch {epoch + 1}/{config.epochs}, "
                f"month_pair={month_idx + 1}/{n_pairs}, "
                f"train_delay_days={config.train_delay_days:g} "
                f"({train_delay_steps} steps), "
                f"train_rollouts={n_rollouts}, target={target_file}"
            )

            if train_delay_steps:
                print(
                    f"  delaying training: advancing {train_delay_steps} steps "
                    "without loss or parameter updates "
                    f"({'NN active' if config.train_delay_use_nn else 'physics only'})"
                )
                _apply_training_runtime_settings(simulation)
                state = rollout_train_delay_state(
                    params,
                    state,
                    model_batched,
                    simulation,
                    config,
                    step_function=step_function,
                    n_steps=train_delay_steps,
                )
                jax.block_until_ready(state)
                delay_finite, delay_text = _state_variable_stats(
                    state,
                    _training_diagnostic_keys(config),
                )
                print(f"  train delay done: state_finite={delay_finite} {delay_text}")
                if not delay_finite and config.fail_on_nonfinite:
                    raise FloatingPointError(
                        "Non-finite values detected during train_delay rollout. "
                        "Try reducing epsilon, using a smaller lr, or disabling the delay corrections."
                    )

            for rollout_idx in range(n_rollouts):
                _apply_training_runtime_settings(simulation)
                loss, grads = compute_train_step(
                    params,
                    state,
                    targets,
                    model_batched,
                    simulation,
                    config,
                    step_function=step_function,
                )

                next_state = rollout_state(
                    params,
                    state,
                    model_batched,
                    simulation,
                    config,
                    step_function=step_function,
                )
                jax.block_until_ready((loss, grads, next_state))

                updates, opt_state = optimizer.update(grads, opt_state, params)
                params_after_update = optax.apply_updates(params, updates)
                jax.block_until_ready((updates, params_after_update))

                if config.diagnostics_every and (rollout_idx + 1) % config.diagnostics_every == 0:
                    _print_update_diagnostics(
                        f"epoch={epoch + 1} month_pair={month_idx + 1} rollout={rollout_idx + 1}",
                        loss,
                        grads,
                        updates,
                        params_after_update,
                        next_state,
                        config,
                    )

                params = params_after_update
                losses.append(loss)
                epoch_losses.append(loss)
                state = next_state

                print(
                    f"  rollout {rollout_idx + 1}/{n_rollouts}, "
                    f"loss={float(loss):.12f}"
                )

        avg_loss = jnp.mean(jnp.stack(epoch_losses))
        print(f"Epoch {epoch + 1}/{config.epochs}, mean_loss={float(avg_loss):.12f}")

        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            save_training_checkpoint(
                config.output_file,
                params,
                opt_state,
                epoch,
                losses,
                config,
            )
            print(f"Saved checkpoint to {config.output_file}")

    return params, opt_state, losses


def train(simulation, config: TrainConfig):
    disable_diagnostics(simulation)
    print(
        f"Training with grad_mode={config.grad_mode}, "
        f"use_jax_checkpoint={config.use_jax_checkpoint}"
    )
    print(
        f"Training data mode={config.initialization_mode}, "
        f"n_restart_files={len(config.restart_files or [])}, "
        f"n_glorys_files={len(config.glorys_files or [])}"
    )
    if config.control_mode == "nn":
        print(f"NN input options: use_lon_lat={config.use_lon_lat}, use_forcings={config.use_forcings}")

    training_files = _training_files_for_mode(config)
    n_raw_files = len(config.glorys_files or config.restart_files or [])
    if n_raw_files != len(training_files):
        print(f"Ignoring {n_raw_files - len(training_files)} lock/input sidecar files")
    if not training_files:
        raise ValueError(
            "TrainConfig.restart_files must contain restart files "
            "or TrainConfig.glorys_files must contain GLORYS files. "
            f"Got initialization_mode={config.initialization_mode!r}, "
            f"len(restart_files)={len(config.restart_files or [])}, "
            f"len(glorys_files)={len(config.glorys_files or [])}. "
            "If initialization_mode='glorys', check that glob.glob(...) matches files "
            "on the compute node and that you pass the sorted list into TrainConfig."
        )

    if config.sample_steps <= 0:
        raise ValueError("TrainConfig.sample_steps must be positive")

    n_targets = _n_targets_per_rollout(config)

    target_stride = None
    max_start = None
    if config.initialization_mode == "restart":
        if config.sample_steps % config.restart_interval_steps != 0:
            raise ValueError(
                "sample_steps must be a multiple of restart_interval_steps. "
                "Current targets are restart files, so losses can only be evaluated "
                "at available restart times."
            )

        target_stride = config.sample_steps // config.restart_interval_steps
        max_start = len(training_files) - n_targets * target_stride - 1
        if max_start < 0:
            raise ValueError("Not enough restart files for the requested rollout")

    nz = simulation.state.settings.nz
    optimizer = create_optimizer(config.lr, grad_clip_norm=config.grad_clip_norm)
    if config.control_mode == "nn":
        model = make_model(config, nz)
        model_batched = make_batched_model(model)
        params, opt_state, start_epoch, losses = init_or_resume_training(
            model,
            optimizer,
            config,
            nz,
        )
    elif config.control_mode == "A_h":
        model_batched = None
        params, opt_state, start_epoch, losses = init_or_resume_A_h_training(
            optimizer,
            config,
            simulation,
        )
    else:
        raise ValueError(f"Unsupported control_mode: {config.control_mode}")

    step_function = None
    if config.control_mode == "A_h":
        step_function = partial(pure, step=simulation.step)
        step_function = maybe_checkpoint(step_function, config.use_jax_checkpoint)

    if config.initialization_mode == "glorys":
        return train_glorys_monthly(
            simulation,
            config,
            params,
            opt_state,
            start_epoch,
            losses,
            model_batched,
            optimizer,
            step_function=step_function,
        )

    rng = np.random.default_rng(config.seed)

    for epoch in range(start_epoch, config.epochs):
        batch_indices = rng.choice(
            max_start + 1,
            size=min(config.batch_size, max_start + 1),
            replace=False,
        )
        batch_losses = []
        batch_grads = []

        for start_idx in batch_indices:
            state_init = load_restart_state(simulation, training_files[start_idx])
            target_files = _target_files_for_window(
                training_files,
                start_idx,
                n_targets,
                target_stride,
            )
            targets = _load_targets(target_files, config.keys_loss)

            _apply_training_runtime_settings(simulation)

            loss, grads = compute_train_step(
                params,
                state_init,
                targets,
                model_batched,
                simulation,
                config,
                step_function=step_function,
            )
            batch_losses.append(loss)
            batch_grads.append(grads)

        avg_loss = jnp.mean(jnp.stack(batch_losses))
        avg_grads = jax.tree_util.tree_map(
            lambda *xs: jnp.mean(jnp.stack(xs), axis=0),
            *batch_grads,
        )
        updates, opt_state = optimizer.update(avg_grads, opt_state, params)
        params_after_update = optax.apply_updates(params, updates)
        jax.block_until_ready((updates, params_after_update))

        if config.diagnostics_every and (epoch + 1) % config.diagnostics_every == 0:
            _print_update_diagnostics(
                f"epoch={epoch + 1}",
                avg_loss,
                avg_grads,
                updates,
                params_after_update,
                state_init,
                config,
            )

        params = params_after_update
        losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{config.epochs}, loss={float(avg_loss):.12f}")

        if (epoch + 1) % config.save_every == 0 or epoch == config.epochs - 1:
            save_training_checkpoint(
                config.output_file,
                params,
                opt_state,
                epoch,
                losses,
                config,
            )
            print(f"Saved checkpoint to {config.output_file}")

    return params, opt_state, losses

