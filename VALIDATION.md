# Validation of the clean training port

Validated on 2026-11-02 against `main` revision **9f4fb66**, using Python 3.12,
JAX with x64 enabled, and the installed Veros on CPU. No production experiments
or raw GLORYS preprocessing jobs were launched or changed.

## Results

| Check | Result |
| --- | --- |
| Unit/regression suite | 34 passed |
| Direct main comparison | Model values/parameter gradients for plain/residual and surface/full-depth networks; basin/depth loss values/gradients; 60-update scheduler sequence agree |
| Horizontal loss filter | Dense linear-solve reference, conservation, land barriers, vertical independence, gradient and filtered loss checks pass |
| Real small-grid training | 8×8×4 Veros model: one accepted update at each of 27, 81, 243 steps; optimizer-preserving continuation at 243; one full-depth update |
| Monthly resume | Two-case shuffled training, no-NN calibration and fixed-panel selection: uninterrupted versus resumed parameters, Adam state, LR, batch history, denominators and score agree exactly |
| Real paired integration | Baseline and NN each complete 720 hourly steps; native monthly NetCDF means, correction second moments and correction plots pass |
| Scientific report | Synthetic full-year monthly NetCDF → TEOS-10 metrics, eleven PNG/PDF plot sets, CSV/NetCDF time series and combined PDF |
| Saved-objective diagnostics | Monthly loss scoring and batch/fixed-panel/LR learning curves pass |
| Packaging | Wheel built; isolated installation imports from outside the repository and exposes all four CLI entry points |

The small model uses a simplified ACC-derived setup. Its bottom-friction process
is disabled because the local Veros version expects a bottom-friction variable
that its shipped ACC example does not initialize. This test-fixture setting does
not change the global ocean example's physics.

These checks cover execution and selected numerical parity, not production-grid
performance or scientific convergence. A 30-year global rollout was **not** run.
Monthly batches execute sequentially, rather than using main's optional multi-GPU
worker pool. Numerical results are compared with main's model/loss/scheduler;
full production-trajectory bitwise equality across machines is not claimed.

## Reproduce

Install `.[diagnostics]` and set the backend before importing the plugin:

```bash
export VEROS_BACKEND=jax VEROS_DEVICE=cpu JAX_PLATFORMS=cpu
export VEROS_LINEAR_SOLVER=scipy_jax VEROS_DISKLESS_MODE=true JAX_ENABLE_X64=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=.

# Optional main checkout used only by parity tests (otherwise these tests skip):
export VEROS_CONTROL_MAIN=/path/to/main/veros_control
python -m unittest discover -s tests -v

# Always use new output directories.
python tests/smoke_training.py outputs/smoke_training
python tests/smoke_batches.py outputs/smoke_training outputs/smoke_batches
python tests/smoke_paired.py outputs/smoke_training outputs/smoke_paired
python tests/diagnostic_workflow.py --keep-dir outputs/smoke_diagnostics
python -m veros_control.diagnostics.report \
  --run-dir outputs/smoke_diagnostics --glorys-dir outputs/smoke_diagnostics/refs \
  --years 1994 --model-thermodynamics pt-sp
python -m pip wheel --no-deps --no-build-isolation --wheel-dir outputs/wheels .
```

Diagnostic test data and plots are synthetic fixtures, not scientific experiment
results. All generated data, wheels and checkpoints stay under ignored `outputs/`.

## Compile reuse and smooth curricula (2026-11-05)

- CPU unit suite: 38 tests, 35 passed and 3 optional main-parity tests skipped
  (no comparison checkout configured).
- Real small-grid Veros: legacy versus cached loss/gradient agreement across
  different initial states, targets, reference intervals and external forcing
  buffers (rtol1e-9, atol1e-10); one cached gradient compilation across cases.
- Two persistent authenticated TCP workers reproduce serial gradients/losses and
  stability reports. Round-trip reuse, key permissions, timeout and remote error
  propagation are covered separately.
- Real two-stage training verifies best-checkpoint transitions, inherited first-
  stage loss denominators/calibration, per-stage LR and no warmup. Candidate
  rejection is configured; the integration fixture's candidates were accepted.
- Nested-scan forward/gradient checks include54,81,135,189,243 steps.

Reproduce the additional integration checks using the existing smoke data:

```bash
python tests/smoke_speedups.py outputs/port_smoke_20261028b
python tests/smoke_smooth_curriculum.py outputs/port_smoke_20261028b
```

The earlier statement that batches are serial describes the previous revision.
Persistent workers are now optional. Production H200 speed/memory and multi-node
Slurm execution of this clean implementation have not been benchmarked; the
research implementation's measured speedup is not assumed to transfer unchanged.
Existing production jobs and their snapshots were not modified.
