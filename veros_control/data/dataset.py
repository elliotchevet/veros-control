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


def create_simulation(config):
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
    with simulation.state.settings.unlock():
        simulation.state.settings.use_forcings = config.model.use_forcings
        simulation.state.settings.normalize_inputs = config.normalization != 'none'
        simulation.state.settings.scaling_method_features = config.normalization if config.normalization != 'none' else 'meanstd'
        simulation.state.settings.baseline_model = True
    for diagnostic in simulation.state.diagnostics.values():
        diagnostic.output_frequency = 0
        diagnostic.sampling_frequency = 0
    if hasattr(simulation,'prepare_ERA5_forcing_window'):
        simulation.prepare_ERA5_forcing_window(simulation.state,config.n_steps)
    dt = float(simulation.state.settings.dt_tracer)
    if config.loss.interpolation != 'next' and config.n_steps*dt > config.reference_hours*3600:
        raise ValueError('Rollout extends beyond the supplied reference pair; supply a longer reference interval')
    initial = load_glorys_snapshot(simulation,config.initial,('temp','salt'),config.time_index)
    target = load_glorys_snapshot(simulation,config.target,('temp','salt'),config.time_index)
    reference = ReferencePair(initial,target,float(simulation.state.variables.time),config.reference_hours*3600)
    return simulation, reference
