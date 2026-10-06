"""The setup applies corrections after physical forcing, before integration."""
from veros import veros_routine
from veros_control.core.correction import predict_nn_forcing, add_nn_forcing_to_state


@veros_routine
def control(state):
    """Post-step plugin hook; tendency injection belongs in setup.set_forcing."""
    pass


def apply_correction(setup, state):
    """Call at the end of set_forcing; rollout.py temporarily supplies parameters."""
    params = getattr(setup, '_nn_params', None)
    if params is None or state.settings.baseline_model:
        return
    tendencies = predict_nn_forcing(
        state, params, setup._nn_model_batched, setup._nn_stats,
        normalize=state.settings.normalize_inputs,
        scaling_method=state.settings.scaling_method_features,
        use_forcings=state.settings.use_forcings,
    )
    add_nn_forcing_to_state(state, *tendencies)


def install_checkpoint(setup, path):
    """Enable a saved correction in an ordinary Veros run after setup()."""
    from veros_control.core.checkpoint import load_checkpoint, model_from_checkpoint
    from veros_control.core.correction import physical_model
    checkpoint = load_checkpoint(path)
    model, config, correction, stats, normalization = model_from_checkpoint(checkpoint,setup.state.settings.nz)
    if correction.momentum and not setup.state.settings.enable_momentum_sources:
        raise ValueError('Enable momentum sources in the setup before using these weights')
    setup._nn_params = checkpoint['params']
    setup._nn_model_batched = physical_model(model,correction,setup.state.variables.zt)
    setup._nn_stats = stats
    setup._nn_correction_initialized = True
    with setup.state.settings.unlock():
        setup.state.settings.baseline_model = False
        setup.state.settings.use_forcings = config.use_forcings
        setup.state.settings.normalize_inputs = normalization != 'none'
        setup.state.settings.scaling_method_features = normalization if normalization != 'none' else 'meanstd'
