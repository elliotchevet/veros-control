import os
import h5netcdf
import xarray as xr
import scipy.ndimage
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
import veros_control

from veros.core.utilities import enforce_boundaries
from veros import VerosSetup, tools, time, veros_routine, veros_kernel, KernelOutput
from veros.variables import Variable, allocate
from veros.core.operators import numpy as npx, update, update_multiply, at
from jax import lax

from veros_control.core.checkpointing import load_checkpoint, checkpoint_model_kwargs
from veros_control.core.forcing import (
    add_nn_forcing_to_state,
    make_batched_model,
    predict_nn_forcing,
)
from veros_control.core.model import VerticalCorrectionMLP

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = tools.get_assets("global_1deg_control", os.path.join(BASE_PATH, "assets.json"))


class GlobalOneDegreeSetup(VerosSetup):
    """Global 1 degree model with 60 vertical levels.

    `Adapted from pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2/1x1%20global%20model>`.

    """
    min_depth = 10.0
    max_depth = 5400.0
    __veros_plugins__ = (veros_control,)

    @veros_routine
    def set_parameter(self, state):
        """
        set main parameters
        """
        settings = state.settings

        settings.identifier = "global_1deg_control"
        settings.description = "Start from Glorys"
        settings.start_date = "1986-01-01"
        #settings.restart_input_filename = "./global_1deg_139052.restart.h5"

        
        settings.nx = 360
        settings.ny = 160
        settings.nz = 60
        settings.dt_mom = 450
        settings.dt_tracer = 450
        settings.dt_forcing = 3
        settings.runlen = 5*3600*24 #(30 * 360 ) * 24 * 3600

        settings.x_origin = 91.0
        settings.y_origin = -79.0

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.enable_hor_friction = True
        settings.A_h = 5e4
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1
        settings.enable_quadratic_bottom_friction = False
        settings.r_quad_bot = 1e-3
        settings.enable_ray_friction = False
        settings.r_ray = 1e-5  # weak interior damping
        settings.enable_tempsalt_sources = True
        settings.enable_implicit_vert_friction = True

        settings.eq_of_state_type = 5

        # isoneutral
        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 50.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.005
        settings.enable_skew_diffusion = True

        # tke
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 1
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True
        settings.enable_tke_superbee_advection = True

        # eke
        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        # idemix
        settings.enable_idemix = False
        settings.enable_eke_diss_surfbot = True
        settings.eke_diss_surfbot_frac = 0.2
        settings.enable_idemix_superbee_advection = True
        settings.enable_idemix_hor_diffusion = True
        settings.enable_streamfunction= False

        # additional settings
        settings.smooth = 3
        self._nn_correction_initialized = False
        self._nn_params = None
        self._nn_model_batched = None
        self._nn_stats = None

        # custom variables
        state.dimensions["nmonths"] = 12
        state.dimensions["n_window"] = 2
        state.var_meta.update(
            qnet=Variable("qnet", ("xt", "yt", "n_window"), "", "", time_dependent=False),
            qsol=Variable("qsol", ("xt", "yt", "n_window"), "", "", time_dependent=False),
            divpen_shortwave=Variable("divpen_shortwave", ("zt",), "", "", time_dependent=False),
            taux=Variable("taux", ("xt", "yt", "n_window"), "", "", time_dependent=False),
            tauy=Variable("tauy", ("xt", "yt", "n_window"), "", "", time_dependent=False),
            evap=Variable("evap", ("xt", "yt", "n_window"), "", "", time_dependent=False),
            tp =Variable("tp", ("xt", "yt", "n_window"), "", "", time_dependent=False),
            t_star =Variable("t_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
            s_star =Variable("s_star", ("xt", "yt", "nmonths"), "", "", time_dependent=False),
        )

        # define ERA5 start time index
        self.n0 = self.ERA5_start_index(DATA_FILES["forcing_era5"], settings.start_date)
        print('n0 = ',self.n0)

    def _init_nn_correction(self, state):
        settings = state.settings

        if self._nn_correction_initialized:
            return

        self._nn_correction_initialized = True

        if settings.baseline_model:
            print("NN correction disabled: baseline_model=True")
            return

        checkpoint = load_checkpoint(settings.nn_checkpoint_path)
        if checkpoint is None:
            print("NN correction disabled: no checkpoint loaded")
            return

        if "params" not in checkpoint:
            print("NN correction disabled: checkpoint has no 'params' entry")
            return

        model_defaults = dict(
            hidden_dim=settings.hidden_dim_nn,
            depth=settings.depth_nn,
            nz=settings.nz,
            nb_profiles=4,
            init_eps=settings.epsilon,
            train_epsilon=settings.train_epsilon,
            use_lon_lat=settings.use_lon_lat,
            conservative_mixing=settings.conservative_mixing,
            use_one_hot=settings.use_one_hot,
            num_mlps=settings.num_mlps,
            TS=settings.TS,
        )
        model_kwargs = checkpoint_model_kwargs(checkpoint, model_defaults)
        model_kwargs["nz"] = settings.nz

        model = VerticalCorrectionMLP(**model_kwargs)
        self._nn_model_batched = make_batched_model(model)
        self._nn_params = checkpoint["params"]
        self._nn_stats = checkpoint.get("stats_features", checkpoint.get("features_stats"))

        print(
            "NN correction enabled from checkpoint "
            f"{settings.nn_checkpoint_path} with {model_kwargs}"
        )

    def _apply_nn_forcing(self, state):
        settings = state.settings

        self._init_nn_correction(state)
        if self._nn_params is None:
            return

        dtemp, dsalt, du, dv = predict_nn_forcing(
            state,
            self._nn_params,
            self._nn_model_batched,
            stats=self._nn_stats,
            normalize=settings.normalize_inputs,
            scaling_method=settings.scaling_method_features,
        )

        add_nn_forcing_to_state(state, dtemp, dsalt, du, dv)

    def _read_forcing(self, var):
        from veros.core.operators import numpy as npx
        default_var = ["xt","yt","zt","dz","temperature","sst","salinity","bathymetry","swf","dqdt","tidal_energy","wind_energy"]
        if var in default_var:
            with h5netcdf.File(DATA_FILES["forcing_default"], "r") as infile:
                var = infile.variables[var]
                return npx.asarray(var).T
        else:
            raise ValueError("Variable {} not in default_var list".format(var))

    def _read_init(self,var):
        from veros.core.operators import numpy as npx
        with h5netcdf.File(DATA_FILES["ic_glorys12"], "r") as infile:
            var = infile.variables[var]
            var = npx.asarray(var).T
            return npx.where(npx.isnan(var), 0.0, var)

    def _read_restoring(self,var):
        from veros.core.operators import numpy as npx
        with h5netcdf.File(DATA_FILES["restoring_glorys12"], "r") as infile:
            var = infile.variables[var]
            return npx.asarray(var).T

    def _read_mask(self,var):
        from veros.core.operators import numpy as npx
        with h5netcdf.File(DATA_FILES["mask"], "r") as infile:
            var = infile.variables[var]
            return npx.asarray(var)
        

    def ERA5_start_index(self,forcing_file, start_date):
        from datetime import datetime
        ref = datetime(1900, 1, 1)
        d0 = datetime.strptime(start_date, "%Y-%m-%d")
        with h5netcdf.File(forcing_file, "r") as f:
            time0 = int(f.variables["time"][0])
        h0 = int((d0 - ref).total_seconds() // 3600)
        return h0 - time0

    def _read_ERA5(self,var):
        if not hasattr(self, "_era5_ds"):
            self._era5_ds = xr.open_dataset(
                DATA_FILES["forcing_era5"],
                engine="h5netcdf",
                chunks={"time": 1},  
            )
        return self._era5_ds[var]

    def get_ERA5(self, var, n1, n2):
        from datetime import datetime, timedelta
        var_da = self._read_ERA5(var)
        sliced = var_da.isel(time=[self.n0+n1,self.n0+n2])
        if var == 'ewss':
            t0 = datetime(1900,1,1)
            print('time = ',var_da.time.values[self.n0+n1])
        return npx.asarray(sliced).T

    @veros_routine(dist_safe=False, local_variables=["dxt", "dyt", "dzt"])
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        vs.dzt = tools.get_vinokur_grid_steps(settings.nz, self.max_depth, self.min_depth, refine_towards="lower")
        vs.dxt = update(vs.dxt, at[...], 1.0)
        vs.dyt = update(vs.dyt, at[...], 1.0)

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt[npx.newaxis, :] / 180.0 * settings.pi)
        )

    def fill_zeros_downward(self,arr):
        valid = arr != 0
        arr_rev = arr[:, :, ::-1]
        valid_rev = valid[:, :, ::-1]
        arr_rev = npx.transpose(arr_rev, (2, 0,1))
        valid_rev = npx.transpose(valid_rev, (2, 0,1))
        def scan_fn(carry, x):
            val, is_valid = x
            new_carry = npx.where(is_valid, val, carry)
            return new_carry, new_carry

        init = npx.zeros_like(arr_rev[0])
        _, filled_rev = lax.scan(scan_fn, init, (arr_rev, valid_rev))
        filled_rev = npx.transpose(filled_rev, (1, 2,0))
        filled = filled_rev[:, :, ::-1]
        not_all_zero = npx.any(valid, axis=2, keepdims=True)
        return npx.where(not_all_zero, filled, arr)

    def flood_fill_nearest(self,field):
        import numpy as np
        field_np = np.asarray(field)
        invalid = field_np == 0
        if np.all(invalid):
            return field
        _, indices = distance_transform_edt(
            invalid,
            return_indices=True
        )
        nearest_values = field_np[tuple(indices)]
        field_jax = npx.asarray(field)
        nearest_jax = npx.asarray(nearest_values)
        filled = npx.where(invalid, nearest_jax, field_jax)
        return filled

    @veros_routine(dist_safe=False, local_variables=["kbot","xt","yt","zt"])
    def set_topography(self, state):
        vs = state.variables
        settings = state.settings
        kbot_data = self._read_mask("kbot")
        vs.kbot = update(vs.kbot, at[2:-2, 2:-2], kbot_data)
        
        # If custom bathymetry needed :

        #t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)
        #bathymetry_data = self._read_forcing("bathymetry")
        #
        ## smooth topography to match grid resolution
        ##gaussian_sigma = (settings.smooth * len(bathymetry_data[0]) / settings.nx, settings.smooth * len(bathymetry_data[1]) / settings.ny)
        ##topo_z_smoothed = scipy.ndimage.gaussian_filter(bathymetry_data, sigma=gaussian_sigma)
        ##topo_z_smoothed = npx.where(bathymetry_data >= -1, 0, topo_z_smoothed)
        #salt_raw = self._read_init("so")[:, :,:,0]
        ##salt_filled = self.fill_zeros_downward(salt_raw)
        ##xt_init, yt_init, zt_init = (self._read_forcing(k) for k in ("xt","yt","zt"))
        ##salt_data = tools.interpolate((xt_init, yt_init, zt_init[::-1]), salt_filled, t_grid)
        ##for k in range(settings.nz):
        ##    salt_data = salt_data.at[:, :, k].set(self.flood_fill_nearest(salt_data[:, :, k]))
        ##
        #land_mask = (vs.zt[npx.newaxis, npx.newaxis, :] <= bathymetry_data[..., npx.newaxis]) | (salt_raw == 0)

        #vs.kbot = update(vs.kbot, at[2:-2, 2:-2], 1 + npx.sum(land_mask.astype("int"), axis=2))
        #mask_bathy = bathymetry_data == 0
        #vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_bathy)
        #vs.kbot = vs.kbot * (vs.kbot < settings.nz)

        ## close some channels
        #i, j = npx.indices((settings.nx, settings.ny))

        #mask_channel = (i >= 207) & (i < 214) & (j < 5)  # i = 208,214; j = 1,5
        #vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_channel)

        ## Aleutian islands
        #mask_channel = (i == 104) & (j == 134)  # i = 105; j = 135
        #vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_channel)

        ## Engl channel
        #mask_channel = (i >= 269) & (i < 271) & (j == 130)  # i = 270,271; j = 131
        #vs.kbot = update_multiply(vs.kbot, at[2:-2, 2:-2], ~mask_channel)
        #vs.kbot = enforce_boundaries(vs.kbot, settings.enable_cyclic_x, local=True)

    @veros_routine(
        dist_safe=False,
        local_variables=[     
            "qnet",
            "qsol",
            "evap",
            "t_star",
            "s_star",
            "tp",
            "divpen_shortwave",
            "taux",
            "tauy",
            "temp",
            "salt",
            "ssh",
            "u",
            "v",
            "forc_iw_bottom",
            "forc_iw_surface",
            "kbot",
            "maskT",
            "maskU",
            "maskV",
            "maskW",
            "xt",
            "xu",
            "yt",
            "yu",
            "zt",
            "zw",
            "dzt",
        ],
    )
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        rpart_shortwave = 0.58
        efold1_shortwave = 0.35
        efold2_shortwave = 23.0


        # create masks.nc

        #ds = xr.Dataset(
        #    {
        #        "maskT": (("xt", "yt", "zt"), vs.maskT[2:-2,2:-2]),
        #        "maskU": (("xu", "yt", "zt"), vs.maskU[2:-2,2:-2]),
        #        "maskV": (("xt", "yu", "zt"), vs.maskV[2:-2,2:-2]),
        #        "maskW": (("xt", "yt", "zw"), vs.maskW[2:-2,2:-2]),
        #        "kbot": (("xt", "yt"), vs.kbot[2:-2,2:-2])
        #    },
        #    coords={
        #        "xt": vs.xt[2:-2],
        #        "xu": vs.xu[2:-2],
        #        "yt": vs.yt[2:-2],
        #        "yu": vs.yu[2:-2],
        #        "zt": vs.zt,
        #        "zw": vs.zw
        #    },
        #)
        #ds.to_netcdf("masks.nc")

        # If custom mask or levels :

        # grids

        #t_grid = (vs.xt[2:-2], vs.yt[2:-2], vs.zt)
        #u_grid = (vs.xu[2:-2], vs.yt[2:-2], vs.zt)
        #v_grid = (vs.xt[2:-2], vs.yu[2:-2], vs.zt)

        #xt_init, yt_init, zt_init = (self._read_forcing(k) for k in ("xt","yt","zt"))

        # initial conditions
        
        temp_raw = self._read_init("thetao")[:,:,:,0]
        #temp_filled = self.fill_zeros_downward(temp_raw)
        #temp_data = tools.interpolate((xt_init, yt_init, zt_init[::-1]), temp_filled, t_grid)
        #for k in range(settings.nz):
        #    temp_data = temp_data.at[:, :, k].set(self.flood_fill_nearest(temp_data[:, :, k]))
        #    temp_data = temp_data.at[:,:,k].set(gaussian_filter(temp_data[:,:,k],sigma=2))
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, 0], temp_raw * vs.maskT[2:-2, 2:-2, :])
        vs.temp = update(vs.temp, at[2:-2, 2:-2, :, 1], temp_raw * vs.maskT[2:-2, 2:-2, :])

        salt_raw = self._read_init("so")[:,:,:,0]
        #salt_filled = self.fill_zeros_downward(salt_raw)
        #salt_data = tools.interpolate((xt_init, yt_init, zt_init[::-1]), salt_filled, t_grid)
        #for k in range(settings.nz):
        #    salt_data = salt_data.at[:, :, k].set(self.flood_fill_nearest(salt_data[:, :, k]))
        #    salt_data = salt_data.at[:,:,k].set(gaussian_filter(salt_data[:,:,k],sigma=2))
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, 0], salt_raw * vs.maskT[2:-2, 2:-2, :])
        vs.salt = update(vs.salt, at[2:-2, 2:-2, :, 1], salt_raw * vs.maskT[2:-2, 2:-2, :])

        ssh_raw = self._read_init("zos")[...,0]
        vs.ssh = update(vs.ssh, at[2:-2, 2:-2], ssh_raw * vs.maskT[2:-2, 2:-2, -1])

        u_raw = self._read_init("uo")[...,0]
        #u_data = tools.interpolate((xu_init, yt_init, zt_init), u_raw, u_grid)
        vs.u = update(vs.u, at[2:-2, 2:-2, :, 0], u_raw * vs.maskU[2:-2, 2:-2, :])
        vs.u = update(vs.u, at[2:-2, 2:-2, :, 1], u_raw * vs.maskU[2:-2, 2:-2, :])

        v_raw = self._read_init("vo")[...,0]
        #v_data = tools.interpolate((xt_init, yu_init, zt_init), v_raw, v_grid)
        vs.v = update(vs.v, at[2:-2, 2:-2, :, 0], v_raw * vs.maskV[2:-2, 2:-2, :])
        vs.v = update(vs.v, at[2:-2, 2:-2, :, 1], v_raw * vs.maskV[2:-2, 2:-2, :])

        t_star = self._read_restoring("thetao")
        vs.t_star = update(vs.t_star, at[2:-2, 2:-2,:], t_star * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])
        s_star = self._read_restoring("so")
        vs.s_star = update(vs.s_star, at[2:-2, 2:-2,:], s_star * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])
        self.n_flow = -1

        if settings.enable_idemix:
            tidal_energy_data = self._read_forcing("tidal_energy")
            mask = (
                npx.maximum(0, vs.kbot[2:-2, 2:-2] - 1)[:, :, npx.newaxis]
                == npx.arange(settings.nz)[npx.newaxis, npx.newaxis, :]
            )
            tidal_energy_data *= vs.maskW[2:-2, 2:-2, :][mask].reshape(settings.nx, settings.ny) / settings.rho_0
            vs.forc_iw_bottom = update(vs.forc_iw_bottom, at[2:-2, 2:-2], tidal_energy_data)

            wind_energy_data = self._read_forcing("wind_energy")
            wind_energy_data *= vs.maskW[2:-2, 2:-2, -1] / settings.rho_0 * 0.2
            vs.forc_iw_surface = update(vs.forc_iw_surface, at[2:-2, 2:-2], wind_energy_data)

        """
        Initialize penetration profile for solar radiation and store divergence in divpen
        note that pen is set to 0.0 at the surface instead of 1.0 to compensate for the
        shortwave part of the total surface flux
        """
        swarg1 = vs.zw / efold1_shortwave
        swarg2 = vs.zw / efold2_shortwave
        pen = rpart_shortwave * npx.exp(swarg1) + (1.0 - rpart_shortwave) * npx.exp(swarg2)

        vs.divpen_shortwave = allocate(state.dimensions, ("zt",))
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[1:], (pen[1:] - pen[:-1]) / vs.dzt[1:])
        vs.divpen_shortwave = update(vs.divpen_shortwave, at[0], pen[0] / vs.dzt[0])

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        settings = state.settings
        year_in_seconds = time.convert_time(1.0, "years", "seconds")
        n_forcing = 365 * 24 / settings.dt_forcing
        (n1, f1), (n2, f2) = tools.get_periodic_interval(vs.time, year_in_seconds, year_in_seconds / n_forcing, n_forcing)
        if self.n_flow != n1:
            self.n_flow = n1
            n1 = n1*settings.dt_forcing
            n2 = n2*settings.dt_forcing
            print("forcing update n°{}".format(n1))
            taux_data = self.get_ERA5("ewss",n1,n2)/3600
            vs.taux = update(vs.taux, at[2:-2, 2:-2, :], taux_data) 
            tauy_data = self.get_ERA5("nsss",n1,n2)/3600
            vs.tauy = update(vs.tauy, at[2:-2, 2:-2, :], tauy_data) 
            vs.update(set_forcing_kernel(state))
            str_ = self.get_ERA5("str",n1,n2)
            Qir = str_/3600
            slhf = self.get_ERA5("slhf",n1,n2)
            Qlat = slhf/3600
            sshf = self.get_ERA5("sshf",n1,n2)
            Qh = sshf/3600
            Qnet = Qir + Qlat + Qh
            vs.qnet = update(vs.qnet, at[2:-2, 2:-2, :], Qnet * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])
            ssr = self.get_ERA5("ssr",n1,n2)
            Qsol = ssr/3600
            vs.qsol = update(vs.qsol, at[2:-2, 2:-2, :], Qsol* vs.maskT[2:-2, 2:-2, -1, npx.newaxis])
            
            # Evaporation 
            evap_data = self.get_ERA5("e",n1,n2)/3600
            vs.evap = update(vs.evap, at[2:-2, 2:-2, :], evap_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

            # Total precipitation
            tp_data = self.get_ERA5("tp",n1,n2)/3600
            vs.tp = update(vs.tp, at[2:-2, 2:-2, :], tp_data * vs.maskT[2:-2, 2:-2, -1, npx.newaxis])

        vs.update(set_forcing_kernel(state))
        self._apply_nn_forcing(state)

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings

        average_vars = [
            "temp",
            "salt",
            "ssh",
            "u",
            "v",
            "w",
            "qsol",
            "qnet",
            "taux",
            "tauy",
            "Nsqr",
            "Hd",
            "K_diss_v",
            "P_diss_v",
            "kbot",
            "P_diss_nonlin",
            "P_diss_iso",
            "kappaH",
            "maskT",
            "dtemp_nn",
            "dsalt_nn",
            "du_nn",
            "dv_nn",
        ]
        if settings.enable_skew_diffusion:
            average_vars += ["B1_gm", "B2_gm"]
        if settings.enable_TEM_friction:
            average_vars += ["kappa_gm", "K_diss_gm"]
        if settings.enable_tke:
            average_vars += ["tke", "Prandtlnumber", "mxl", "tke_diss", "forc_tke_surface", "tke_surf_corr"]
        if settings.enable_idemix:
            average_vars += ["E_iw", "forc_iw_surface", "forc_iw_bottom", "iw_diss", "c0", "v0"]
        if settings.enable_eke:
            average_vars += ["eke", "K_gm", "L_rossby", "L_rhines"]

        state.diagnostics["averages"].output_variables = average_vars
        state.diagnostics["cfl_monitor"].output_frequency = 10*86400
        state.diagnostics["snapshot"].output_frequency = 3600
        state.diagnostics["overturning"].output_frequency = 360 * 86400
        state.diagnostics["overturning"].sampling_frequency = 5 * 86400 
        state.diagnostics["energy"].output_frequency = 1800
        state.diagnostics["energy"].sampling_frequency = 450
        state.diagnostics["averages"].output_frequency = 360 * 86400 / 12
        state.diagnostics["averages"].sampling_frequency = 86400/2

    @veros_routine
    def after_timestep(self, state):
        #import numpy as np
        #vs = state.variables
        #print(vs.xt[175])
        #print(vs.yt[9])
        #print(
        #    "max |u| =", float(npx.nanmax(npx.abs(vs.u))),
        #    "max |v| =", float(npx.nanmax(npx.abs(vs.v))),
        #    "max |w| =", float(npx.nanmax(npx.abs(vs.w))),
        #    "max T =", float(npx.nanmax(vs.temp)),
        #    "min T =", float(npx.nanmin(vs.temp)),
        #    "max S =", float(npx.nanmax(vs.salt)),
        #    "rho max =", vs.rho.max(),
        #    "rho min =", vs.rho.min()
        #)

        #bad = ~npx.isfinite(vs.rho)
        #
        #n_bad = int(npx.sum(bad))
        #print("Number bad rho cells:", n_bad)
        #
        #if n_bad > 0:
        #    # move to numpy for inspection
        #    bad_np = np.array(bad)
        #
        #    bad_indices = np.argwhere(bad_np)
        #    print("Bad cell at:")
        #    for idx in bad_indices:
        #        print(tuple(idx))
        pass
                


@veros_kernel
def set_forcing_kernel(state):
    vs = state.variables
    settings = state.settings

    cp_0 = 3991.86795711963  # J/kg /K
    dQdT = -40 # W/m2 /K (2 months for a 50m mld)
    Ks = -6.37e-6 # piston velocity in m/s (3 months for a 50m mld)

    year_in_seconds = time.convert_time(1.0, "years", "seconds")
    n_forcing = 365 * 24 / settings.dt_forcing
    (n1, f1), (n2, f2) = tools.get_periodic_interval(vs.time, year_in_seconds, year_in_seconds / n_forcing, n_forcing)

    (n1_rest, f1_rest), (n2_rest, f2_rest) = tools.get_periodic_interval(vs.time, year_in_seconds, year_in_seconds / 12.0, 12)
    vs.surface_taux = update(vs.surface_taux, at[:, :], f1 * vs.taux[:, :, 0] + f2 * vs.taux[:, :, 1])
    vs.surface_tauy = update(vs.surface_tauy, at[:, :], f1 * vs.tauy[:, :, 0] + f2 * vs.tauy[:, :, 1])

    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
            )** (3.0 / 2.0),
        )

    qqnet = f1 * vs.qnet[..., 0] + f2 * vs.qnet[..., 1]
    t_star_cur = f1_rest * vs.t_star[..., n1_rest] + f2_rest * vs.t_star[..., n2_rest]
    vs.forc_temp_surface = (qqnet + dQdT * (vs.temp[..., -1,vs.tau] - t_star_cur))  * vs.maskT[..., -1] / cp_0 / settings.rho_0
    evap_cur =  f1 * vs.evap[..., 0]+ f2 * vs.evap[..., 1]
    tp_cur =  f1 * vs.tp[..., 0] + f2 * vs.tp[..., 1]
    s_star_cur = f1_rest * vs.s_star[..., n1_rest] + f2_rest * vs.s_star[..., n2_rest]
    vs.forc_salt_surface = (Ks*(vs.salt[...,-1,vs.tau]-s_star_cur) -(tp_cur - evap_cur) * vs.salt[..., -1,vs.tau]) * vs.maskT[..., -1]

    # apply simple ice mask
    mask1 = vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] > -1.8
    mask2 = vs.forc_temp_surface > 0
    ice = npx.logical_or(mask1, mask2)
    vs.forc_temp_surface *= ice
    vs.forc_salt_surface *= ice

    # solar radiation
    if settings.enable_tempsalt_sources:
        vs.temp_source = (
            (f1 * vs.qsol[..., 0, None] + f2 * vs.qsol[..., 1, None])
            * vs.divpen_shortwave[None, None, :]
            * ice[..., None]
            * vs.maskT[..., :]
            / cp_0
            / settings.rho_0
        )

    return KernelOutput(
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy, 
        temp_source=vs.temp_source,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
    )
