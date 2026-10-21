# Implementation updates

Since `eab0ccd`:

- Optional residual MLP blocks and surface-only outputs; full-depth corrections remain available.
- Basin bias/pattern losses, sampled GLORYS targets, full-column or three-band thickness weighting, baseline depth normalization, and spatial filtering.
- Curriculum and monthly initial-state batches with averaged gradients, optional parallel workers, and timestamp-aligned cyclic ERA5 forcing.
- Best/latest checkpoints, optimizer-preserving resume, candidate validation, LR backtracking/recovery, and worker recycling.
- Safe forcing-window indexing and correction depth taper; focused regression tests.

Only runtime code, reusable training runners, tests and documentation are added.
Experiment launchers, calibration outputs, paired-rollout analyses, error-region
experiments, checkpoints and generated results are excluded. Training still uses
the six rectangular basins; learned regions are not integrated.
