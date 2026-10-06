# veros-control

Column-wise neural corrections for Veros temperature, salinity, and momentum.
Training and evaluation share the same correction function and integration code.

## Install and configure

Use a Python 3.11+ environment with a working Veros/JAX installation, then:

```bash
pip install -e .
export VEROS_BACKEND=jax
export VEROS_DEVICE=gpu                 # cpu also works
export VEROS_LINEAR_SOLVER=scipy_jax
export VEROS_DISKLESS_MODE=true
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The example uses GLORYS snapshots already remapped to the model's grid and levels.
It does not download or regrid data. Copy `examples/global_ocean/assets.example.json`,
fill in local file paths, and set `VEROS_CONTROL_ASSETS` to its absolute path.
Choose atmospheric forcing and reference periods deliberately; filenames are not used
to infer dates or substitute forcing years.

Edit the example configurations, then run from this repository:

```bash
veros-control-train examples/train_config.json
veros-control-evaluate examples/evaluation_config.json
python -m veros_control.diagnostics.train_diags outputs/training/example/latest.pkl outputs/training/example/figures
python -m veros_control.diagnostics.test_diags outputs/evaluation/example
```

Configuration paths are relative to their JSON file. `setup` names a factory as
`module:function` or `path/to/setup.py:function`; the factory returns an initialized
Veros setup. The supplied example is in `examples/global_ocean/setup.py`.

## Supported controls

- One vertical-column MLP: hidden width/depth, coordinates, and forcing inputs.
- Fixed tracer/momentum scales, optional momentum corrections, smooth hourly bounds,
  and optional physical-depth taper. The NN returns tendencies, not state increments.
- `basin` loss: squared basin-mean bias plus `pattern_lambda` times within-basin
  error variance. At lambda 1 this is area-weighted pointwise MSE within each basin/level.
- `pointwise` loss: area-weighted MSE over all wet cells, independently at each level.
- `levels: 0` selects every depth; a positive value selects that many shallowest levels.
  Active region-level pairs are averaged equally, not weighted by water volume.
- Loss at every `sample_steps`, averaged through `n_steps`; the latter must be a multiple
  of the former. Endpoint-only training sets both to the same value.
- `sampled` interpolation uses elapsed time / `reference_hours`; `endpoint` uses one fixed
  rollout-end target; `next` uses the supplied next snapshot directly. Monthly reference
  means are not instantaneous observations. No interpolation past the reference interval.
- Optional fixed T/S loss balancing, feature normalization, gradient clipping, and
  candidate feasibility checks. A finite, feasible update can still increase the loss.
- Nested gradient checkpointing is always used. There is no benchmark gradient-mode switch.

The six basin rectangles preserve the research definitions: Atlantic [-70,20), Pacific
[120,180) plus [-180,-70), split at the equator; southern portions end at 60 S.
Indian covers [20,120) and [-60,30); Southern covers latitudes below 60 S.
**The sector [20,120) north of 30 N is outside the basin loss.** Use pointwise loss
when all wet regions should be included. Rate bounds and CFL checks do not guarantee
physical temperature ranges or long-horizon accuracy.

## Checkpoints and continuation

Every accepted update is evaluated and saved atomically as `latest.pkl`, with matching
optimizer state, preprocessing, objective, normalization, and exact `checkpoint_loss`.
`best.pkl` keeps the lowest evaluated loss. Identical best/latest states share storage.
History records both pre-update and post-update losses.

- Set `resume` to a public-format checkpoint to restore parameters, optimizer, history,
  and fixed loss denominators. Supply the same objective and a **new output directory**.
  `updates` is the number of additional updates. An unchanged configured learning rate
  preserves any saved backoff; an explicitly different rate overrides it.
- Set `weights` instead to initialize parameters and saved preprocessing while starting
  a new optimizer and loss normalization. Recent single-MLP research checkpoints work
  here and in evaluation. Their removed experimental architectures are rejected.
- Full optimizer resume from the old research format is deliberately rejected rather
  than silently migrating its objective. Existing research runs remain on their branch.
- Pickle checkpoints must come from a trusted source.

## Integrate another Veros setup

Register `veros_control` in the setup's `__veros_plugins__` and enable temperature/salinity
sources (and momentum sources when used). At the end of `set_forcing`, call:

```python
from veros_control.control import apply_correction
apply_correction(self, state)
```

Rebuild physical source fields each step so prior NN sources cannot accumulate.
The shared rollout supplies parameters during training/evaluation. For an ordinary
Veros run, call `install_checkpoint(setup, path)` from `veros_control.control` after
`setup.setup()`. This restores scales, bounds, taper, and feature normalization.
The plugin's post-step hook intentionally does not inject tendencies after integration.

## Structure and checks

`core/` contains model, correction, preprocessing, rollout, and checkpoint code.
`data/` owns shared initialization, forcing indices, and reference loading.
`training/`, `evaluation/`, and `diagnostics/` expose the three user workflows.
Example setups and scheduler scripts live outside the installed package.

```bash
VEROS_BACKEND=jax VEROS_DEVICE=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -m unittest discover -s tests -v
```

These are regression checks, not the historical multi-length JVP/VJP experiments.
Generated files belong under a configurable `outputs/` directory and are excluded from Git.
