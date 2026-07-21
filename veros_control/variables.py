from veros.variables import Variable, T_GRID, U_GRID, V_GRID, TIMESTEPS

VARIABLES = {'dtemp_nn': Variable('Temperature tendency correction', T_GRID+TIMESTEPS,"deg C/s", "NN prediction of temperature tendency correction"),
             'dsalt_nn': Variable('Salinity tendency correction', T_GRID+TIMESTEPS,"g/(kg s)", "NN prediction of salinity tendency correction"),
             'du_nn': Variable('Zonal velocity tendency correction', U_GRID+TIMESTEPS,"m/s^2", "NN prediction of zonal velocity tendency correction"),
             'dv_nn': Variable('Meridional velocity tendency correction', V_GRID+TIMESTEPS,"m/s^2", "NN prediction of meridional velocity tendency correction")
             }

