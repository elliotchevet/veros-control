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
