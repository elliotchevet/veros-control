"""Initialize a configured Veros setup and load a pair of reference snapshots."""
from dataclasses import dataclass
import importlib
import importlib.util
import sys
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from veros_control.data.reference import load_glorys_state, load_glorys_snapshot


@dataclass
class ReferencePair:
    initial: dict
    target: dict
    initial_time: float
    duration_seconds: float


def create_simulation(config, simulation=None):
    from veros_control.data.dated import validate_date_options
    validate_date_options(config)
    if simulation is None:
        module_name, factory = config.setup.rsplit(':',1)
        if module_name.endswith('.py'):
            module_id = '_veros_control_setup_' + hashlib.sha256(str(Path(module_name).resolve()).encode()).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(module_id,module_name)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_id] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
        simulation = getattr(module,factory)()
    if not simulation.state.settings.enable_tempsalt_sources:
        raise ValueError('Setup must enable temperature and salinity sources')
    if config.correction.momentum and not simulation.state.settings.enable_momentum_sources:
        raise ValueError('Momentum corrections require enable_momentum_sources')
    initialization = SimpleNamespace(glorys_start_month=config.start_month,
        glorys_month_lengths=(31,28,31,30,31,30,31,31,30,31,30,31))
    simulation.state = load_glorys_state(simulation,config.initial,config.time_index,config=initialization)
    if getattr(config, 'forcing_time_shift', None) is not None:
        import jax.numpy as jnp
        with simulation.state.variables.unlock():
            if 'forcing_time_shift' in simulation.state.var_meta:
                simulation.state.variables.forcing_time_shift = jnp.asarray(config.forcing_time_shift)
            elif config.forcing_time_shift != 0:
                raise ValueError('Setup needs a forcing_time_shift variable for monthly forcing alignment')
    with simulation.state.settings.unlock():
        simulation.state.settings.use_date_embedding = config.model.use_date_embedding
        simulation.state.settings.disable_surface_restoring = config.disable_surface_restoring
        simulation.state.settings.correct_evaporation_sign = config.correct_evaporation_sign
        simulation.state.settings.dated_forcing = bool(config.initial_date)
        simulation.state.settings.use_forcings = config.model.use_forcings
        simulation.state.settings.normalize_inputs = config.normalization != 'none'
        simulation.state.settings.scaling_method_features = config.normalization if config.normalization != 'none' else 'meanstd'
        simulation.state.settings.baseline_model = True
    for diagnostic in simulation.state.diagnostics.values():
        diagnostic.output_frequency = 0
        diagnostic.sampling_frequency = 0
    reference_seconds = config.reference_hours*3600
    if config.initial_date:
        from datetime import datetime
        from veros_control.data.dated import model_timestamp, MODEL_YEAR
        import jax.numpy as jnp
        date = datetime.fromisoformat(config.initial_date)
        from veros_control.data.reference import snapshot_date
        if snapshot_date(config.initial) != date:
            raise ValueError('initial_date differs from GLORYS timestamp')
        reference_seconds = model_timestamp(snapshot_date(config.target))-model_timestamp(date)
        if reference_seconds <= 0:
            raise ValueError('Dated target must follow the initial snapshot')
        if not hasattr(simulation, 'prepare_dated_forcing'):
            raise ValueError('Setup must implement prepare_dated_forcing for dated data')
        with simulation.state.variables.unlock():
            simulation.state.variables.forcing_time_shift = jnp.asarray(model_timestamp(date)-(date.year-1993)*MODEL_YEAR)
            simulation.state.variables.date_year_offset = jnp.asarray(float(date.year-1993))
        simulation.prepare_dated_forcing(simulation.state, config.forcing_directory, date, config.n_steps)
    elif hasattr(simulation,'prepare_ERA5_forcing_window'):
        simulation.prepare_ERA5_forcing_window(simulation.state,config.n_steps)
    dt = float(simulation.state.settings.dt_tracer)
    if config.loss.interpolation != 'next' and config.n_steps*dt > reference_seconds:
        raise ValueError('Rollout extends beyond the supplied reference pair; supply a longer reference interval')
    initial = load_glorys_snapshot(simulation,config.initial,('temp','salt'),config.time_index)
    target = load_glorys_snapshot(simulation,config.target,('temp','salt'),config.time_index)
    reference = ReferencePair(initial,target,float(simulation.state.variables.time),reference_seconds)
    return simulation, reference
