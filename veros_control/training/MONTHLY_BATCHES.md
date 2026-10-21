# Monthly GLORYS initial-state batches

Opt in with `scripts/split_epsilon/curriculum_training.py`:

```
--monthly-initial-states /Odyssey/private/e25cheve/veros/veros_assets/global_1deg_realistic
--initial-state-batch-size 3
--initial-state-seed 0
--checkpoint-months 1 5 9
```

Use the existing sampled basin-loss, normalization, bounds and validation options.
Configure the horizons and update counts through successive runner invocations.
Monthly batching is opt-in; existing jobs are unchanged.

- Files match `*4*0_1993*.nc`. January–November are initial states; each target is
  the following month. December is target-only by default. No missing next-year
  truth is silently invented. `--cyclic-glorys-targets` explicitly permits
  December → January 1993 if a periodic climatological experiment is desired.
- Each shuffled pass visits every initial month once. Default batches contain
  three states (the last batch contains two). Months are chosen deterministically
  from seed and update, so stage resume restores the exact sampling sequence.
- Compute each member's rollout and gradient sequentially with unchanged nested
  checkpointing. Average losses and gradients, then perform **one** Adam update.
  Candidate stability checks and backtracking use the same months as the gradient.
- Best/latest scores use the fixed January/May/September panel, not changing batch
  losses. These months are also training months: this is a comparable monitoring
  score, **not** independent validation. This costs three extra forward rollouts
  per saved update. `--checkpoint-months` can select another fixed panel.
- The actual NetCDF timestamps anchor initialization, forcing phase and the linear
  target interpolation. Monthly means in these files are dated near midmonth.
  Gregorian timestamps are mapped onto Veros' 360-day model year by 360/365;
  the target interval is the difference of adjacent timestamps with the same
  mapping. This replaces the legacy fixed 720-hour target interval in this mode.
- ERA5 defaults to `ERA5_1deg_1993_Flux_processed.nc` in the supplied asset folder;
  override with `--era5-forcing-file`. Its 365-day record sequence repeats on the
  model-year clock. Forcing subsets include both interpolation records, including
  wrapping across year end. Files outside veros_control are read-only.
- Rollouts are restricted to the interval before the next GLORYS anchor. Longer
  than monthly training needs a sequence of targets rather than extrapolation.
- Supply fixed depth denominators and input statistics explicitly. Checkpoints
  record the monthly setup and histories record the actual batch months.


Optional controls: `--monthly-parallel-workers`, `--monthly-worker-nodes`,
`--monthly-fused-validation`, and `--checkpoint-score-every`. The coordinator
averages member gradients into one Adam update. Fused validation reuses candidate
losses; a fixed reference panel selects best checkpoints. Worker recycling and
targeted shutdown preserve coordinator-owned optimizer state.

`--monthly-baseline-normalization` calibrates fixed outer T/S denominators from
all twelve no-NN monthly initial states at each horizon; stage resume restores
these denominators. It requires the twelve-month cyclic-target configuration.
