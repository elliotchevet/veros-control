# veros-control

A Veros plugin for training and applying column-wise neural corrections to
temperature, salinity and momentum. This branch reorganizes the training behavior
of `main` (revision `9f4fb66`) into shared model, data, training and diagnostic modules.
No raw GLORYS download, filtering or coarse-graining tools are included.

## Install on another machine

Use Python 3.11+ and a working Veros/JAX installation for your CPU or GPU:

```bash
pip install -e '.[diagnostics]'
export VEROS_BACKEND=jax
export VEROS_DEVICE=gpu                 # use cpu for CPU execution
export VEROS_LINEAR_SOLVER=scipy_jax
export VEROS_DISKLESS_MODE=true
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Provide preprocessed forcing, initial and restoring data on the model grid.
The supplied global setup also needs its grid/bathymetry/mask assets. Copy
`examples/global_ocean/assets.example.json`, fill in absolute local paths, and set
`VEROS_CONTROL_ASSETS` to that manifest's absolute path. The example repeats its
supplied ERA5 forcing year and monthly restoring cycle. It does not silently switch
to other forcing years. Its original restoring coefficients are −40 W/m²/K for
heat and −6.37e−6 m/s for salinity.

Only the diagnostic extra needs xarray, GSW, Cartopy and pypdf. Plots use locally
cached coastlines when available and otherwise render without coastlines; they do
not require a network connection. `VEROS_EVALUATION_COASTLINES` can name a Cartopy
cache directory.

## Train an experiment

Edit file paths in one of these configurations:

| Configuration | Corrections and loss | Initial states |
| --- | --- | --- |
| `examples/train_surface.json` | Surface only | One state |
| `examples/train_config.json` | Full depth | One state |
| `examples/train_monthly.json` | Surface only | Twelve monthly states, batches of three |

All three examples use curriculum **27 → 81 → 243 steps**, with **150 / 100 / 50
updates**, and samples every 27 steps. With the example's hourly timestep these
are hours. An update means one Adam update, not one complete pass through the data.

```bash
veros-control-train examples/train_surface.json
# Or submit from the repository root with your cluster's partition/account:
sbatch examples/train.slurm examples/train_surface.json
```

Paths in a configuration are relative to that JSON file. `setup` is a factory
`module:function` or `path/to/setup.py:function` returning an initialized Veros setup.
The package never imports a sibling `main` checkout or research script.

Each stage writes `config.json`, `loss_calibration.json`, `latest.pkl` and `best.pkl`
under `output/window_27`, `window_81`, or `window_243`. A transition carries the
**latest parameters, Adam state, feature statistics and LR** into the next stage.
It resets the stage update counter and scheduler streak, recalibrates the no-NN
loss for the new horizon, and starts a new best-checkpoint comparison.

### Model and loss options

- `model.surface_only`: retain the full-depth inputs and predict only surface
  outputs; deeper direct tendencies are exactly zero. Use `loss.levels: 1`.
  Set it to false and `loss.levels: 0` for full-depth outputs/loss.
- `model.residual_blocks`: zero selects a plain MLP; positive values select main's
  residual architecture. Width, layers per block, forcing inputs and periodic
  longitude/latitude encoding are configurable.
- `correction`: tracer/momentum scales, optional momentum, smooth hourly bounds,
  depth taper and profile chunk size. Outputs are tendencies, not increments.
- `loss.mode`: `basin` uses the original six rectangles; `pointwise` uses all wet
  cells. `pattern_lambda: 1` gives area-weighted MSE within each region/level;
  zero gives squared regional-mean bias. No experimental 12-basin masks are used.
- `loss.spatial_filter`: main's differentiable wet-cell Helmholtz loss filter.
  The examples blend 10% raw and 90% filtered MSE, with wavelengths 0/400/800 km
  at lead times 27/81/243 hours. This filters loss errors, not raw GLORYS data.
- `loss.normalization: no_nn`: before each stage, run the uncorrected model from
  **every selected training state**, average its raw T/S terms, and freeze those
  denominators. This is the latest rebalancing: equal relative T/S contributions.
  `initial` instead calibrates from initial model parameters; `none` disables
  outer normalization. `balance: false` also disables it.
- `loss.depth_loss`: optional saved depth calibration object from main, containing
  `depths_m`, `thickness_m`, `thickness_weighting: true`, positive T/S
  `denominators`, and optional `band_boundaries_m`. This preserves thickness/band
  weighting when reproducing a specific full-depth experiment. With no object,
  active region-level pairs receive equal weights. To reproduce an existing run,
  copy its calibration and model settings, not just its architecture.

The six training regions are the Atlantic and Pacific split at the equator,
Indian, and Southern Ocean south of 60°S. The Indian-sector area north of 30°N is
outside this loss, exactly as on main. Scientific diagnostic sectors retain the
separate definitions used in the surface-control plots; they do not redefine the
training loss.

### Monthly batches and clocks

`initial_states` explicitly lists initial/target files and month identifiers.
The target can wrap December to January for a repeating training year. By default,
NetCDF timestamps determine the forcing offset and interpolation interval,
including mid-month monthly-mean timestamps, scaled to Veros's model-year length.
Set `forcing_time_shift` (seconds) and `reference_hours` on a case to override this.
Single-state configurations use `start_month`, `reference_hours`, and optional
`forcing_time_shift` explicitly.

Each update averages independent case gradients before one Adam update. Batches
follow deterministic shuffled passes; the last batch can be smaller.
`checkpoint_months` selects a **fixed training panel** for every best/latest score
(default: all selected states). Batch losses are logged separately. Execution is
sequential and keeps one compiled case in memory; this prioritizes bounded memory
and portability over main's optional multi-GPU worker implementation.

### Learning rate and continuation

`scheduler.mode` supports:

- `keep_update`: keep physically valid updates even if loss increases. From
  `start_update` (20 in examples), a failed same-batch loss test halves the **next**
  LR once; ten consecutive strict improvements recover 50%, capped at
  `learning_rate`. The example floor is `min_rate: 1e-5`, ceiling `1e-3`.
- `backtracking`: retry the same Adam direction at smaller rates, with loss-decrease
  tests after warmup and recovery after five first-try acceptances, as on main.
- `constant`: no loss-decrease test; physical candidate failures can still reduce
  the rate. Numerical invalidity never commits parameters or Adam state.

For same-stage continuation, copy its saved `config.json`, set `curriculum: []`,
set `transition: null` and `weights: null`, set `resume` to `latest.pkl`,
choose a **new output directory**, and set `updates`
to the additional count. Parameters, Adam, LR, scheduler streak, normalization,
batch order and history are restored. Objective, seed, clipping and LR ceiling
must match. Use `transition` instead of `resume` to start a longer horizon with
carried optimizer state and fresh calibration. Use `weights` for a new optimizer.
Recent main single-MLP plain/residual and surface/full-depth checkpoints support
weights-only initialization and inference. Main's legacy optimizer format and
older clean objective signatures are not silently migrated.

## Training diagnostics

Learning curves distinguish changing batches from the fixed checkpoint panel:

```bash
python -m veros_control.diagnostics.train_diags \
  outputs/training/surface/window_243/latest.pkl outputs/training/surface/figures
```

The continuous **30-year paired rollout is `train_diags`**, because it starts from
a training state. It performs no optimizer updates. Edit `examples/paired_config.json`:

```bash
veros-control-train-diags examples/paired_config.json
veros-control-report --run-dir outputs/train_diags/paired_30years \
  --glorys-dir /path/to/preprocessed/GLORYS --years 1994 2003
```

The runner records both baseline and NN trajectories, hourly stability checks,
full-depth monthly NetCDF means, velocity second moments, and correction means /
second moments. It uses hourly timesteps and 30-day months on a 360-day calendar,
without annual state resets. Set `months: 1` for a short check. `stop_on_invalid`
defaults to true; checks are evaluated hourly and handled after each 24-step block.
A warn-only run with nonfinite fields is marked invalid and cannot be reported as
a successful comparison. The summary records the first invalid step.

Reports contain SST/SSS/mean-removed SSH maps, winter/summer density-threshold MLD,
zonal T/S sections, resolved overturning, heat content, volume-mean temperature,
resolved kinetic energy, monthly time series, saved-objective rollout losses,
correction profiles/RMS and equivalent heat/salt fluxes. PNGs, CSVs, NetCDF metrics,
annual PDFs and a combined `train_diags.pdf` are written locally. Report years
require all twelve completed months and matching preprocessed reference files
named `GLORYS12_1deg_YYYYMM_processed.nc`. A 1993-start 30-year run covers 1993–2022;
use 31 years to evaluate 2023.

Reports use fixed common wet support across selected years and trajectories.
Model monthly means have equal 30-day weights; GLORYS annual/seasonal means use
Gregorian month lengths. MLD is computed from monthly mean profiles, not averaged
hourly MLD. KE from monthly velocities and mean hourly KE are kept distinct.
Monthly loss is the loss of monthly means, not the average hourly training loss.
Model EOS5 CT/SA convention is the diagnostic default; `--model-thermodynamics
pt-sp` selects potential-temperature/practical-salinity inputs. No raw-data
conversion or regridding is performed by the plugin.

## Held-out evaluation and another setup

`veros-control-evaluate examples/evaluation_config.json` retains the short paired
held-out workflow, with `python -m veros_control.diagnostics.test_diags DIRECTORY`
for its plots. Keep those initial states separate from the training set.

To integrate your own setup, register `veros_control` in `__veros_plugins__`, enable
tracer sources (and momentum sources when used), rebuild physical sources each
step, then call `apply_correction(self, state)` from `veros_control.control` at the
end of `set_forcing`. For ordinary Veros runs, call `install_checkpoint(setup, path)`
after setup. The correction is applied before integration, never as a post-step
state adjustment. Checkpoints are trusted-local pickle files.

## Layout and verification

- `core/`: model, bounded correction, feature processing, shared rollout, checkpoints.
- `data/`: already-remapped references, initialization, calendar and forcing indices.
- `training/`: strict configuration, batches, loss/filter, scheduler, training.
- `evaluation/`: frozen-weight held-out inference.
- `diagnostics/`: training curves, paired archives, scientific metrics and reports.
- `examples/`: portable configuration, setup and Slurm examples; no site paths.
- `tests/`: numerical regressions and small real Veros integration workflows.

See [VALIDATION.md](VALIDATION.md) for checks and their scope. Training tests verify
execution and numerical parity; they do not establish 30-year physical stability.
