from veros.settings import Setting

SETTINGS = {
    'baseline_model': Setting(True, bool, 'Disable NN corrections'),
    'normalize_inputs': Setting(False, bool, 'Normalize NN features using saved statistics'),
    'scaling_method_features': Setting('meanstd', str, 'Feature scaling: meanstd or minmax'),
    'use_forcings': Setting(True, bool, 'Include physical forcing features'),
    # Used by the supplied global-ocean example.
    'start_date': Setting('', str, 'Simulation start date'),
    'dt_forcing': Setting(3, int, 'Atmospheric forcing record stride in hours'),
    'smooth': Setting(2, int, 'Topography smoothing scale'),
}
