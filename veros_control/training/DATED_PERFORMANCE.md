# Dated surface training and performance options

`curriculum_training.py --training-years 1993 2002` samples all 120 monthly
initial conditions with matching-year ERA5 forcing and next-month GLORYS targets.
Forcing records are mapped onto the 360-day model calendar, including leap years
and year boundaries. Only the required forcing window is read.

Optional model/physics flags (existing defaults remain unchanged):

- `--use-date-embedding`: seasonal sine/cosine and elapsed decades, appended
  after physical-input normalization.
- `--disable-surface-restoring`: zero SST/SSS restoring fluxes.
- `--correct-evaporation-sign`: use `-(precipitation + signed_evaporation) * S`.

## Persistent workers

For the existing Slurm monthly worker mode, enable:

```sh
export MONTHLY_TRANSPORT=tcp
export MONTHLY_COMPILE_CACHE=1
export MONTHLY_WORKER_MAX_REQUESTS=10000
```

TCP uses an authenticated persistent connection per worker. A random key in a
mode-0600 file is shared at startup; request/result payloads no longer poll shared
storage. Python pickle transport is for trusted workers within the allocation.
The original file transport remains available by leaving these variables unset.

Compile reuse currently supports dated, surface-only T/S training with six basins,
one surface level and pattern lambda=1. State, forcing, target dates and loss
normalizers are dynamic arguments; geometry and rollout structure stay fixed.
Forward, gradient and fused stability/scoring functions each compile separately.
Workers retain the compiled functions across dates until recycled. Select a
worker request limit appropriate to the allocation's memory budget.

A 27h A100 benchmark across Jan1993, Feb1996 and Dec2002 produced identical losses
and gradients to the original implementation: one gradient trace, approximately
219–225 seconds per original call versus 8.7–9.4 seconds after the cached compile.
A separate two-node 1MiB transport test measured about 70 seconds for shared-file
polling versus 0.026 seconds for TCP round trips. These are component benchmarks,
not a promised end-to-end speedup. Longer training subsequently ran with these
options; this does not constitute an independent gradient-correctness audit.

## Conservative stage restarts

Use `--fresh-training --restart-stage-lr --curriculum-source CHECKPOINT
--nn-checkpoint CHECKPOINT --reuse-source-normalization` to carry weights, Adam
and fixed normalization into a new window while resetting its epoch/LR schedule.
`--stage-initial-lr` selects a starting rate below `--train-lr`, which remains the
recovery ceiling. `--lr-decrease-start-epoch 1` disables warmup.

With `--loss-halving-continue --loss-backtracking`, the optional
`--reject-worsening-updates` discards candidates failing the same-batch loss test
and rolls back both parameters and Adam. The next rate still halves once, bounded
by `--train-min-lr`; recovery uses `--lr-recovery-interval` strict improvements.
At the floor, failed candidates are skipped rather than accepted. Numerical
stability failures retain the existing failure behavior.

For the recent restart: floor1e-5, ceiling1e-3, initial1e-4 at81h and1e-5 at243h,
no warmup, recovery after2 improvements. Pass `train.best.pkl` between stages.
Best scores use a fixed reference panel of training months, not held-out data.
Resume within a window with `--resume-stage` to preserve scheduler state and
numbering; rejection mode must match the saved checkpoint.
