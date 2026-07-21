from functools import partial

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np
import optax
from veros import restart
from veros.core.operators import update, at

from veros_control.core.checkpointing import load_checkpoint, save_checkpoint
from veros_control.core.forcing import make_batched_model
from veros_control.core.model import VerticalCorrectionMLP
from veros_control.training.config import TrainConfig
from veros_control.training.loss import compute_loss
from veros_control.training.trainset import load_snapshot


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


def create_optimizer(lr):
    return optax.adamw(lr)


def disable_diagnostics(simulation):
    for diagnostic in simulation.state.diagnostics.values():
        diagnostic.sampling_frequency = None
        diagnostic.output_frequency = None


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


def make_model(config, nz):
    return VerticalCorrectionMLP(
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        nz=nz,
        nb_profiles=4,
        init_eps=config.epsilon,
        train_epsilon=config.train_epsilon,
        use_lon_lat=config.use_lon_lat,
        conservative_mixing=config.conservative_mixing,
        use_one_hot=config.use_one_hot,
        num_mlps=config.num_mlps,
        TS=config.TS,
    )


def init_params(model, nz, seed):
    x_tmp = jnp.zeros((nz, 4))
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


def init_or_resume_r_bot_training(optimizer, config, simulation):
    checkpoint_path = config.resume_checkpoint_path or config.output_file
    checkpoint = None

    if config.resume_training or not config.overwrite_training:
        checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint and not config.overwrite_training:
        print(f"Resuming r_bot training from epoch {checkpoint['epoch'] + 1}")
        return (
            checkpoint["params"],
            checkpoint["opt_state"],
            checkpoint["epoch"] + 1,
            list(checkpoint.get("losses", [])),
        )

    if config.r_bot_init is not None:
        r_bot_init = config.r_bot_init
    else:
        r_bot_init = getattr(simulation.state.variables, "r_bot", 1e-5)

    params = {"r_bot": jnp.asarray(r_bot_init)}
    opt_state = optimizer.init(params)
    print(f"Starting new r_bot training from r_bot={float(params['r_bot']):.12g}")
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
):
    next_state = state.copy()

    simulation._nn_correction_initialized = True
    simulation._nn_params = params
    simulation._nn_model_batched = model_batched
    simulation._nn_stats = stats

    with next_state.settings.unlock():
        next_state.settings.baseline_model = False
        next_state.settings.normalize_inputs = normalize
        if scaling_method is not None:
            next_state.settings.scaling_method_features = scaling_method

    simulation.step(next_state)
    return next_state


def r_bot_set_forcing_step(state, params, step_function):
    next_state = state.copy()
    with next_state.variables.unlock():
        next_state.variables.r_bot = update(
            next_state.variables.r_bot,
            at[...],
            params["r_bot"],
        )

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
        )
    elif control_mode == "r_bot":
        if step_function is None:
            raise ValueError("step_function is required for control_mode='r_bot'")
        step_with_forcing = partial(
            r_bot_set_forcing_step,
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
        control_mode=config.control_mode,
        step_function=step_function,
    )

    return jax.value_and_grad(loss_fn)(params)


def _zero_like_params(params):
    return jax.tree_util.tree_map(jnp.zeros_like, params)


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
    for step_idx in range(n_steps):
        with jax.profiler.TraceAnnotation("forward_step", step_idx=step_idx):
            state = step_fn(state, params)
    return state


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
        )
    elif config.control_mode == "r_bot":
        if step_function is None:
            raise ValueError("step_function is required for control_mode='r_bot'")
        step_fn = partial(
            r_bot_set_forcing_step,
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
            with jax.profiler.TraceAnnotation("loss_vjp", step_idx=step_idx):
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
        state_before = _forward_to_step(state_init, params, step_fn, step_idx - 1)
        jax.block_until_ready(state_before)
        jax.profiler.save_device_memory_profile(
        f"/Odyssey/private/e25cheve/jax_traces/before_step_vjp_{step_idx}.prof"
        )
        _, step_vjp = jax.vjp(step_fn, state_before, params)
        jax.block_until_ready(step_vjp)
        jax.profiler.save_device_memory_profile(
        f"/Odyssey/private/e25cheve/jax_traces/after_build_step_vjp_{step_idx}.prof"
        )
        state_cotangent, param_cotangent = step_vjp(state_cotangent)
        jax.block_until_ready(state_cotangent)
        jax.profiler.save_device_memory_profile(
        f"/Odyssey/private/e25cheve/jax_traces/after_apply_step_vjp_{step_idx}.prof"
        )
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
    if config.grad_mode == "builtin":
        return train_step(
            params,
            state_init,
            targets,
            model_batched,
            simulation,
            config,
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

    raise ValueError(f"Unsupported grad_mode: {config.grad_mode}")


def _target_files_for_window(restart_files, start_idx, n_targets, target_stride):
    return [
        restart_files[start_idx + (target_idx + 1) * target_stride]
        for target_idx in range(n_targets)
    ]


def _load_targets(target_files, keys):
    return [load_snapshot(filename, keys) for filename in target_files]


def save_training_checkpoint(path, params, opt_state, epoch, losses, config):
    save_checkpoint(
        path,
        {
            "params": params,
            "opt_state": opt_state,
            "epoch": epoch,
            "losses": losses,
            "hidden_dim_nn": config.hidden_dim,
            "depth_nn": config.depth,
            "epsilon": config.epsilon,
            "train_epsilon": config.train_epsilon,
            "use_lon_lat": config.use_lon_lat,
            "conservative_mixing": config.conservative_mixing,
            "use_one_hot": config.use_one_hot,
            "num_mlps": config.num_mlps,
            "TS": config.TS,
            "stats_features": config.stats_features,
            "normalize_inputs": config.normalize_inputs,
            "scaling_method_features": config.scaling_method_features,
            "control_mode": config.control_mode,
            "r_bot_init": config.r_bot_init,
            "grad_mode": config.grad_mode,
        },
    )


def train(simulation, config: TrainConfig):
    disable_diagnostics(simulation)
    print(
        f"Training with grad_mode={config.grad_mode}, "
        f"use_jax_checkpoint={config.use_jax_checkpoint}"
    )

    if not config.restart_files:
        raise ValueError("TrainConfig.restart_files must contain restart files")

    if config.sample_steps <= 0:
        raise ValueError("TrainConfig.sample_steps must be positive")
    if config.n_steps % config.sample_steps != 0:
        raise ValueError("n_steps must be a multiple of sample_steps")
    if config.sample_steps % config.restart_interval_steps != 0:
        raise ValueError(
            "sample_steps must be a multiple of restart_interval_steps. "
            "Current targets are restart files, so losses can only be evaluated "
            "at available restart times."
        )

    n_targets = config.n_steps // config.sample_steps
    if n_targets < 1:
        raise ValueError("n_steps must be at least sample_steps")

    target_stride = config.sample_steps // config.restart_interval_steps
    max_start = len(config.restart_files) - n_targets * target_stride - 1
    if max_start < 0:
        raise ValueError("Not enough restart files for the requested rollout")

    nz = simulation.state.settings.nz
    optimizer = create_optimizer(config.lr)
    if config.control_mode == "nn":
        model = make_model(config, nz)
        model_batched = make_batched_model(model)
        params, opt_state, start_epoch, losses = init_or_resume_training(
            model,
            optimizer,
            config,
            nz,
        )
    elif config.control_mode == "r_bot":
        model_batched = None
        params, opt_state, start_epoch, losses = init_or_resume_r_bot_training(
            optimizer,
            config,
            simulation,
        )
    else:
        raise ValueError(f"Unsupported control_mode: {config.control_mode}")

    step_function = None
    if config.control_mode == "r_bot":
        step_function = partial(pure, step=simulation.step)
        step_function = maybe_checkpoint(step_function, config.use_jax_checkpoint)


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
            state_init = load_restart_state(simulation, config.restart_files[start_idx])
            target_files = _target_files_for_window(
                config.restart_files,
                start_idx,
                n_targets,
                target_stride,
            )
            targets = _load_targets(target_files, config.keys_loss)

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
        params = optax.apply_updates(params, updates)
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

