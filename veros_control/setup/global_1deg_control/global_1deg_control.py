import os

import h5netcdf
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
    reset_nn_momentum_sources,
    make_batched_model,
    predict_nn_forcing,
)
from veros_control.core.model import VerticalCorrectionMLP
from veros_control.setup.global_1deg_control.forcing_window import (
    plan_forcing_window, local_forcing_indices,
)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_FILES = tools.get_assets("global_1deg_control", os.path.join(BASE_PATH, "assets.json"))


class GlobalOneDegreeSetup(VerosSetup):
    """Global 1 degree model with 40 vertical levels.

    `Adapted from pyOM2 <https://wiki.zmaw.de/ifm/TO/pyOM2/1x1%20global%20model>`.

    """
    min_depth = 10.0
    max_depth = 5400.0
    __veros_plugins__ = (veros_control,)

    def __init__(self, *args, skip_setup_initial_conditions=None, **kwargs):
        super().__init__(*args, **kwargs)
        if skip_setup_initial_conditions is None:
            skip_setup_initial_conditions = os.environ.get(
                "VEROS_CONTROL_SKIP_SETUP_INITIAL_CONDITIONS",
                "",
            ).lower() in ("1", "true", "yes", "on")
        self._skip_setup_initial_conditions = skip_setup_initial_conditions

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
        settings.nz = 40
        settings.dt_mom = 3600
        settings.dt_tracer = 3600
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
        settings.enable_momentum_sources = True
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
        settings.baseline_model = True
        self._nn_correction_initialized = False
        self._nn_params = None
        self._nn_model_batched = None
        self._nn_stats = None

        # custom variables
        state.dimensions["nmonths"] = 12
        if (365 * 24) % settings.dt_forcing != 0:
            raise ValueError("dt_forcing must divide the 365-day hourly forcing year")
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
            forcing_time_shift=Variable(
                "forcing_time_shift", tuple(), "s", "Forcing clock offset from January 1", time_dependent=False
            ),
        )

    def _init_nn_correction(self, state):
        settings = state.settings

        if self._nn_correction_initialized:
            return

        if settings.baseline_model:
            print("NN correction disabled: baseline_model=True")
            return

        self._nn_correction_initialized = True

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
            use_forcings=settings.use_forcings,
            conservative_mixing=settings.conservative_mixing,
            use_one_hot=settings.use_one_hot,
            num_mlps=settings.num_mlps,
            TS=settings.TS,
        )
        model_kwargs = checkpoint_model_kwargs(checkpoint, model_defaults)
        model_kwargs["nz"] = settings.nz

        model = VerticalCorrectionMLP(**model_kwargs)
        self._nn_model_batched = make_batched_model(
            model,
            chunk_size=settings.nn_chunk_size or None,
        )
        self._nn_params = checkpoint["params"]
        self._nn_stats = checkpoint.get("stats_features", checkpoint.get("features_stats"))
        if self._nn_stats is not None:
            checkpoint_normalize = checkpoint.get("normalize_inputs")
            checkpoint_scaling = checkpoint.get("scaling_method_features")
            if checkpoint_normalize is not None or checkpoint_scaling is not None:
                with settings.unlock():
                    if checkpoint_normalize is not None:
                        settings.normalize_inputs = bool(checkpoint_normalize)
                    if checkpoint_scaling is not None:
                        settings.scaling_method_features = checkpoint_scaling
        checkpoint_use_forcings = checkpoint.get("use_forcings")
        if checkpoint_use_forcings is not None:
            with settings.unlock():
                settings.use_forcings = bool(checkpoint_use_forcings)

        print(
            "NN correction enabled from checkpoint "
            f"{settings.nn_checkpoint_path} with {model_kwargs}; "
            f"normalize_inputs={settings.normalize_inputs}, "
            f"scaling_method_features={settings.scaling_method_features}, "
            f"use_forcings={settings.use_forcings}"
        )

    def _apply_nn_forcing(self, state):
        settings = state.settings

        if self._nn_params is None and not self._nn_correction_initialized:
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
            use_forcings=settings.use_forcings,
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
        

    def forcing_time_shift_seconds(self, start_date, model_year_seconds):
        from datetime import datetime

        date = datetime.strptime(start_date, "%Y-%m-%d")
        month_days = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        if date.month == 2 and date.day == 29:
            raise ValueError("forcing start_date cannot be February 29 with 365-day forcing data")
        day_of_year = sum(month_days[: date.month - 1]) + date.day - 1

        return day_of_year / 365.0 * model_year_seconds

    def _read_ERA5_forcing(self, var, n_forcing, dt_forcing):
        forcing_window = getattr(self, "_era5_forcing_window", None)
        if forcing_window is None and getattr(self, 'setup_forcing_window', False):
            # Setup evaluates forcing at t=0 only. Avoid reading an entire year
            # before the requested monthly rollout window is prepared.
            forcing_window = dict(start=0, length=3, period_count=n_forcing)
            self._era5_forcing_window = forcing_window
        if forcing_window is not None:
            cache_key = (var, forcing_window["start"], forcing_window["length"], dt_forcing)
            if not hasattr(self, "_era5_forcing_cache"):
                self._era5_forcing_cache = {}
            if cache_key not in self._era5_forcing_cache:
                self._era5_forcing_cache[cache_key] = self._load_ERA5_forcing_window(
                    var,
                    forcing_window,
                    dt_forcing,
                )
            return npx.asarray(self._era5_forcing_cache[cache_key])

        with h5netcdf.File(getattr(self, "era5_forcing_file", DATA_FILES["forcing_era5"]), "r") as infile:
            return npx.asarray(infile.variables[var]).T[..., ::dt_forcing][..., :n_forcing]

    def _load_ERA5_forcing_window(self, var, forcing_window, dt_forcing):
        import numpy as np

        indices = (
            forcing_window["start"] + np.arange(forcing_window["length"])
        ) % forcing_window["period_count"]
        with h5netcdf.File(getattr(self, "era5_forcing_file", DATA_FILES["forcing_era5"]), "r") as infile:
            data = np.stack(
                [np.asarray(infile.variables[var][int(idx) * dt_forcing]) for idx in indices],
                axis=0,
            )
        return data.T

    def prepare_ERA5_forcing_window(self, state, n_steps, margin=0, leading_margin=0):
        import numpy as np

        vs = state.variables
        settings = state.settings
        period_count = 365 * 24 // settings.dt_forcing
        model_year_seconds = time.convert_time(1.0, "years", "seconds")
        forcing_interval = model_year_seconds / period_count
        forcing_time = float(np.asarray(vs.time + vs.forcing_time_shift)) % model_year_seconds

        self._era5_forcing_window = plan_forcing_window(
            forcing_time, n_steps, settings.dt_tracer, forcing_interval, period_count, margin, leading_margin,
        )
        length = self._era5_forcing_window["length"]
        self._era5_forcing_cache = {}

        for var in ("ewss", "nsss", "str", "slhf", "sshf", "ssr", "e", "tp"):
            cache_key = (var, self._era5_forcing_window["start"], length, settings.dt_forcing)
            self._era5_forcing_cache[cache_key] = self._load_ERA5_forcing_window(
                var,
                self._era5_forcing_window,
                settings.dt_forcing,
            )

        print(
            "Using ERA5 benchmark forcing window: "
            f"start={self._era5_forcing_window['start']} "
            f"length={length}/{period_count}"
        )

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
            "forcing_time_shift",
            "t_star",
            "s_star",
            "tp",
            "divpen_shortwave",
            "taux",
            "tauy",
            "temp",
            "salt",
            "psi",
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
        if getattr(self, "_skip_setup_initial_conditions", False):
            print("Skipping setup initial-condition fields; external GLORYS state will be loaded after setup")
        else:
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
            ssh_surface = ssh_raw * vs.maskT[2:-2, 2:-2, -1]
            vs.psi = update(vs.psi, at[2:-2, 2:-2, 0], ssh_surface * settings.grav)
            vs.psi = update(vs.psi, at[2:-2, 2:-2, 1], ssh_surface * settings.grav)

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

        vs.forcing_time_shift = update(
            vs.forcing_time_shift,
            at[...],
            self.forcing_time_shift_seconds(
                settings.start_date,
                time.convert_time(1.0, "years", "seconds"),
            ),
        )
        self._init_nn_correction(state)

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
        reset_nn_momentum_sources(state)
        vs = state.variables
        settings = state.settings
        n_forcing = 365 * 24 // settings.dt_forcing

        taux_forcing = self._read_ERA5_forcing("ewss", n_forcing, settings.dt_forcing) / 3600
        tauy_forcing = self._read_ERA5_forcing("nsss", n_forcing, settings.dt_forcing) / 3600
        qnet_forcing = (
            self._read_ERA5_forcing("str", n_forcing, settings.dt_forcing)
            + self._read_ERA5_forcing("slhf", n_forcing, settings.dt_forcing)
            + self._read_ERA5_forcing("sshf", n_forcing, settings.dt_forcing)
        ) / 3600
        qsol_forcing = self._read_ERA5_forcing("ssr", n_forcing, settings.dt_forcing) / 3600
        evap_forcing = self._read_ERA5_forcing("e", n_forcing, settings.dt_forcing) / 3600
        tp_forcing = self._read_ERA5_forcing("tp", n_forcing, settings.dt_forcing) / 3600

        vs.update(
            set_forcing_kernel(
                state,
                taux_forcing,
                tauy_forcing,
                qnet_forcing,
                qsol_forcing,
                evap_forcing,
                tp_forcing,
                getattr(self, "_era5_forcing_window", {"start": 0})["start"],
                n_forcing,
            )
        )
        if not state.settings.baseline_model:
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
def set_zero_forcing_kernel(state):
    vs = state.variables

    vs.taux = update(vs.taux, at[...], npx.zeros_like(vs.taux))
    vs.tauy = update(vs.tauy, at[...], npx.zeros_like(vs.tauy))
    vs.qnet = update(vs.qnet, at[...], npx.zeros_like(vs.qnet))
    vs.qsol = update(vs.qsol, at[...], npx.zeros_like(vs.qsol))
    vs.evap = update(vs.evap, at[...], npx.zeros_like(vs.evap))
    vs.tp = update(vs.tp, at[...], npx.zeros_like(vs.tp))
    vs.surface_taux = update(vs.surface_taux, at[...], npx.zeros_like(vs.surface_taux))
    vs.surface_tauy = update(vs.surface_tauy, at[...], npx.zeros_like(vs.surface_tauy))
    vs.temp_source = update(vs.temp_source, at[...], npx.zeros_like(vs.temp_source))
    vs.forc_tke_surface = update(vs.forc_tke_surface, at[...], npx.zeros_like(vs.forc_tke_surface))
    vs.forc_temp_surface = update(vs.forc_temp_surface, at[...], npx.zeros_like(vs.forc_temp_surface))
    vs.forc_salt_surface = update(vs.forc_salt_surface, at[...], npx.zeros_like(vs.forc_salt_surface))

    return KernelOutput(
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy,
        taux=vs.taux,
        tauy=vs.tauy,
        qnet=vs.qnet,
        qsol=vs.qsol,
        evap=vs.evap,
        tp=vs.tp,
        temp_source=vs.temp_source,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
    )

@veros_kernel
def set_forcing_kernel(
    state,
    taux_forcing,
    tauy_forcing,
    qnet_forcing,
    qsol_forcing,
    evap_forcing,
    tp_forcing,
    forcing_index_start,
    forcing_period_count,
):
    vs = state.variables
    settings = state.settings

    cp_0 = 3991.86795711963  # J/kg /K
    dQdT = -40 # W/m2 /K (2 months for a 50m mld)
    Ks = -6.37e-6 # piston velocity in m/s (3 months for a 50m mld)

    model_year_seconds = time.convert_time(1.0, "years", "seconds")
    forcing_time = vs.time + vs.forcing_time_shift

    (n1, f1), (n2, f2) = tools.get_periodic_interval(
        forcing_time,
        model_year_seconds,
        model_year_seconds / forcing_period_count,
        forcing_period_count,
    )

    (n1_rest, f1_rest), (n2_rest, f2_rest) = tools.get_periodic_interval(
        forcing_time,
        model_year_seconds,
        model_year_seconds / 12.0,
        12,
    )

    window_length = taux_forcing.shape[-1]
    local_n1, local_n2, forcing_covered = local_forcing_indices(
        n1, n2, forcing_index_start, forcing_period_count, window_length, npx,
    )
    # JAX gathers may silently clamp out-of-range indices. Mark missing forcing
    # explicitly nonfinite instead of recycling unrelated records. The rollout
    # finite-state checks then reject an exhausted or incorrectly sized window.
    f1 = npx.where(forcing_covered, f1, npx.nan)
    f2 = npx.where(forcing_covered, f2, npx.nan)

    mask_surface = vs.maskT[2:-2, 2:-2, -1, npx.newaxis]
    taux_data = npx.stack([taux_forcing[:, :, local_n1], taux_forcing[:, :, local_n2]], axis=-1) * mask_surface
    tauy_data = npx.stack([tauy_forcing[:, :, local_n1], tauy_forcing[:, :, local_n2]], axis=-1) * mask_surface
    qnet_data = npx.stack([qnet_forcing[:, :, local_n1], qnet_forcing[:, :, local_n2]], axis=-1) * mask_surface
    qsol_data = npx.stack([qsol_forcing[:, :, local_n1], qsol_forcing[:, :, local_n2]], axis=-1) * mask_surface
    evap_data = npx.stack([evap_forcing[:, :, local_n1], evap_forcing[:, :, local_n2]], axis=-1) * mask_surface
    tp_data = npx.stack([tp_forcing[:, :, local_n1], tp_forcing[:, :, local_n2]], axis=-1) * mask_surface

    taux_window = update(npx.zeros_like(vs.taux), at[2:-2, 2:-2, :], taux_data)
    tauy_window = update(npx.zeros_like(vs.tauy), at[2:-2, 2:-2, :], tauy_data)
    qnet_window = update(npx.zeros_like(vs.qnet), at[2:-2, 2:-2, :], qnet_data)
    qsol_window = update(npx.zeros_like(vs.qsol), at[2:-2, 2:-2, :], qsol_data)
    evap_window = update(npx.zeros_like(vs.evap), at[2:-2, 2:-2, :], evap_data)
    tp_window = update(npx.zeros_like(vs.tp), at[2:-2, 2:-2, :], tp_data)

    vs.taux = update(vs.taux, at[...], taux_window)
    vs.tauy = update(vs.tauy, at[...], tauy_window)
    vs.qnet = update(vs.qnet, at[...], qnet_window)
    vs.qsol = update(vs.qsol, at[...], qsol_window)
    vs.evap = update(vs.evap, at[...], evap_window)
    vs.tp = update(vs.tp, at[...], tp_window)

    surface_taux = f1 * taux_window[:, :, 0] + f2 * taux_window[:, :, 1]
    surface_tauy = f1 * tauy_window[:, :, 0] + f2 * tauy_window[:, :, 1]
    vs.surface_taux = update(vs.surface_taux, at[:, :], surface_taux)
    vs.surface_tauy = update(vs.surface_tauy, at[:, :], surface_tauy)

    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
            )** (3.0 / 2.0),
        )

    qqnet = f1 * qnet_window[:, :, 0] + f2 * qnet_window[:, :, 1]
    t_star_cur = f1_rest * vs.t_star[..., n1_rest] + f2_rest * vs.t_star[..., n2_rest]
    vs.forc_temp_surface = (qqnet + dQdT * (vs.temp[..., -1,vs.tau] - t_star_cur))  * vs.maskT[..., -1] / cp_0 / settings.rho_0
    evap_cur =  f1 * evap_window[:, :, 0] + f2 * evap_window[:, :, 1]
    tp_cur =  f1 * tp_window[:, :, 0] + f2 * tp_window[:, :, 1]
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
            (f1 * qsol_window[:, :, 0, None] + f2 * qsol_window[:, :, 1, None])
            * vs.divpen_shortwave[None, None, :]
            * ice[..., None]
            * vs.maskT[..., :]
            / cp_0
            / settings.rho_0
        )

    return KernelOutput(
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy, 
        taux=vs.taux,
        tauy=vs.tauy,
        qnet=vs.qnet,
        qsol=vs.qsol,
        evap=vs.evap,
        tp=vs.tp,
        temp_source=vs.temp_source,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
    )
