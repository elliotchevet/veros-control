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


Since `de6eb9c`:

- Optional keep-update scheduler: one halving of the next LR on loss-test failure,
  50% recovery after ten consecutive improvements, floor/ceiling, and numerical
  rejection checks. Original backtracking remains available.
- Persist scheduler mode, next LR and improving streak; enforce resume compatibility
  and expose the decisions in epoch logs/history.
- Optional monthly subsets, including January-only training, with no-NN loss
  denominators calibrated over that same subset.
- Optional restoring-provenance JSON in checkpoints (metadata, not a physics switch).
- Regression tests and usage documentation. Surface-only learning remains opt-in
  through `--surface-only`, with surface loss selected by `--basin-levels 1`.

The main implementation retains only the original six basins. Experimental
learned-region/12-basin loss integration and generated experiment artifacts are
not included. Existing six-basin loss, normalization and filtering are unchanged.
