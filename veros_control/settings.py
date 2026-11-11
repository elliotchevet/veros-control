from veros.settings import Setting

SETTINGS = {
    'use_date_embedding': Setting(False, bool, 'Append seasonal phase and elapsed decades'),
    'disable_surface_restoring': Setting(False, bool, 'Disable SST and SSS restoring in supported setups'),
    'correct_evaporation_sign': Setting(False, bool, 'Use signed CDS evaporation in supported setups'),
    'dated_forcing': Setting(False, bool, 'Use Gregorian ERA5 on a 360-day calendar'),
    'baseline_model': Setting(True, bool, 'Disable NN corrections'),
    'normalize_inputs': Setting(False, bool, 'Normalize NN features using saved statistics'),
    'scaling_method_features': Setting('meanstd', str, 'Feature scaling: meanstd or minmax'),
    'use_forcings': Setting(True, bool, 'Include physical forcing features'),
    # Used by the supplied global-ocean example.
    'start_date': Setting('', str, 'Simulation start date'),
    'dt_forcing': Setting(3, int, 'Atmospheric forcing record stride in hours'),
    'smooth': Setting(2, int, 'Topography smoothing scale'),
}
