# Optional surface basin-mean GLORYS objective

Add to the benchmark CLI:

    --loss-mode glorys_target --basin-mean-loss --basin-levels 5

Use `--basin-levels 10` for ten shallowest levels. The default without `--basin-mean-loss` remains the original spatial objective. Existing GLORYS interpolation and sample-step settings still apply. Only temperature and salinity are supported for this objective.

For each variable, compute a separate ocean-area-weighted mean at each selected level in each rectangular basin. Compare model and GLORYS means, square their difference, then average equally over nonempty basin-level pairs. Average the per-variable losses for the scalar objective. With `--balance-loss-terms`, the existing initial-rollout calibration divides each variable's basin loss by its fixed initial value plus 1e-12. Calibration is specific to this objective; spatial-loss denominators must not be reused.

Coordinates are normalized to [-180,180). Boundaries are nonoverlapping: latitude zero belongs to north; latitude -60 belongs to the named southern subdivisions/Indian, not Southern Ocean.

| Region | Longitude | Latitude |
| --- | --- | --- |
| North Atlantic | [-70,20) | [0,90] |
| South Atlantic | [-70,20) | [-60,0) |
| North Pacific | [120,180) or [-180,-70) | [0,90] |
| South Pacific | [120,180) or [-180,-70) | [-60,0) |
| Indian | [20,120) | [-60,30) |
| Southern | all | [-90,-60) |

These preserve the agreed rectangles exactly. There is no separate Arctic basin; waters in longitude [20,120) north of 30 N are not assigned. Land, the two halo rows/columns on every side, and empty basin-level pairs are excluded. Basin boundaries are approximate, not coastline-following geographic basin masks.

The shallowest levels are found by sorting absolute `zt`, not by taking array indices 0:5. Selected indices, depths, active pairs, target means, and definition version are saved in checkpoint metadata. Mean biases can cancel spatial errors; this loss does not enforce realistic local fields or deep water. Correction bounds and candidate validation remain separate controls.

Default output names include `_basin_top5` or `_basin_top10`. Resuming a different objective or level count is rejected. To initialize a new objective from existing weights, use `--nn-checkpoint PATH` with a fresh training output rather than reusing its optimizer checkpoint and loss denominators. The split-epsilon continuation runner intentionally resumes the same objective; direct migration through that runner is rejected.

The running training job and its Slurm script have not been changed or restarted.

## Endpoint spatial-pattern experiment

`--basin-pattern-lambda` defaults to 0 (the existing basin-mean objective).
For each tracer, basin and selected level the new optional contribution is
`mean_area(error)^2 + lambda * mean_area((error - mean_area(error))^2)`.
At lambda=1 this is exactly area-weighted pointwise MSE, evaluated directly.
Active basin-level pairs receive equal weight. Existing tracer balancing divides
these terms by their initial rollout values plus the existing numerical floor.
The checkpoint stores lambda and rejects incompatible objective resumes.

`scripts/train_endpoint_job.slurm` starts fresh for 80 optimizer updates with
`--sample-steps 27 --train-n-steps 27 --basin-levels 5 --basin-pattern-lambda 1`.
Only the rollout endpoint enters the loss, using the interpolated GLORYS target
at alpha=27/720. Loss normalization is calibrated anew for this objective.
Outputs are isolated under `scripts/basin_endpoint_training/job_JOBID`.

`--correction-depth-taper 5 5` in the split-epsilon runner multiplies all four
bounded NN corrections by a physical-depth raised cosine. It equals 1 through
level 5, transitions as `(1+cos(pi*(depth-depth5)/(depth10-depth5)))/2`, and is
zero at level 10 and deeper. Levels are counted from the shallowest. This
reduces the correction caps as well as the tendencies. Only levels 1–5 enter
the loss; deeper ocean states can still change through the model physics.
The taper is optional (default off) and stored in checkpoints for resume checks.
Nested checkpointing, split epsilon, bounds and candidate validation are unchanged.

## All-depth sampled trajectory experiment

`--basin-levels 0` selects all model levels, preserving equal weights across nonempty basin-level pairs (not volume weighting). Any positive level count is also supported. Geographic basin masks are unchanged. Omit `--correction-depth-taper` to allow bounded corrections through the full column.

`--glorys-sampled-interpolation` opts into a target interpolated at each sampled model time: alpha = (time - initial_time)/(720 * dt_tracer). It requires `--basin-mean-loss --glorys-linear-interpolation`; without this flag the previous fixed-endpoint target remains unchanged. The interpolation mode is saved and checked when resuming.

`train_sampled243_alllevels_job.slurm` starts fresh parameters (random hidden layers, zero output layer), optimizer and loss normalization. It averages nine losses at steps 27, 54, ..., 243 with lambda 1 and all 40 levels, for 80 updates. Split epsilons, smooth bounds, candidate validation and nested checkpointing remain enabled. No previous weights are loaded.
