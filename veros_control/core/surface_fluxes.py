"""Surface tracer flux conventions shared by forcing and regression tests."""
def surface_tracer_fluxes(qnet, temperature, temperature_target, salinity,
                          salinity_target, precipitation, evaporation,
                          heat_restoring, salt_restoring, rho0, cp0, *,
                          disable_restoring=False, correct_evaporation_sign=False):
    heat_coefficient = 0.0 if disable_restoring else heat_restoring
    salt_coefficient = 0.0 if disable_restoring else salt_restoring
    freshwater = precipitation + evaporation if correct_evaporation_sign else precipitation - evaporation
    return ((qnet + heat_coefficient * (temperature-temperature_target)) / (rho0*cp0),
            salt_coefficient * (salinity-salinity_target) - freshwater * salinity)
