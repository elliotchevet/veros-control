from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from veros import restart

from veros_control.core.checkpointing import load_checkpoint, save_checkpoint
from veros_control.core.forcing import make_batched_model
from veros_control.core.model import VerticalCorrectionMLP
from veros_control.training.config import TrainConfig
from veros_control.training.loss import compute_loss
from veros_control.training.trainset import load_snapshot


def create_optimizer(lr):
    return optax.adamw(lr)


def load_restart_state(simulation, restart_file):
    with simulation.state.settings.unlock():
        simulation.state.settings.restart_input_filename = restart_file
    restart.read_restart(simulation.state)
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
):
    state = state_init.copy()
    losses = []
    target_idx = 0

    step_with_forcing = partial(
        nn_set_forcing_step,
        simulation=simulation,
        model_batched=model_batched,
        stats=stats,
        normalize=normalize,
        scaling_method=scaling_method,
    )
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
    )

    return jax.value_and_grad(loss_fn)(params)


def _target_files_for_window(restart_files, start_idx, n_targets):
    return restart_files[start_idx + 1 : start_idx + 1 + n_targets]


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
        },
    )


def train(simulation, config: TrainConfig):
    if not config.restart_files:
        raise ValueError("TrainConfig.restart_files must contain restart files")

    if config.sample_steps <= 0:
        raise ValueError("TrainConfig.sample_steps must be positive")
    if config.n_steps % config.sample_steps != 0:
        raise ValueError("n_steps must be a multiple of sample_steps")

    n_targets = config.n_steps // config.sample_steps
    if n_targets < 1:
        raise ValueError("n_steps must be at least sample_steps")

    max_start = len(config.restart_files) - n_targets - 1
    if max_start < 0:
        raise ValueError("Not enough restart files for the requested rollout")

    nz = simulation.state.settings.nz
    model = make_model(config, nz)
    model_batched = make_batched_model(model)
    optimizer = create_optimizer(config.lr)
    params, opt_state, start_epoch, losses = init_or_resume_training(
        model,
        optimizer,
        config,
        nz,
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
            state_init = load_restart_state(simulation, config.restart_files[start_idx])
            target_files = _target_files_for_window(
                config.restart_files,
                start_idx,
                n_targets,
            )
            targets = _load_targets(target_files, config.keys_loss)

            with simulation.state.settings.unlock():
                simulation.state.settings.enable_tke = False
                simulation.state.settings.kappaH_0 = 0.06
                simulation.state.settings.kappaM_0 = 0.06

            loss, grads = train_step(
                params,
                state_init,
                targets,
                model_batched,
                simulation,
                config,
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
