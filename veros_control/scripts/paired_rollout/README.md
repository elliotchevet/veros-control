# Paired basin-trained replay

Local rollback checkpoints are kept separately from the published implementation.

`sbatch scripts/paired_rollout/run.slurm` from veros_control runs a forward-only baseline and trained replay from the same GLORYS state and ERA5 forcing window. The source is basin checkpoint job_55098, update 100. Source parameters, normalization statistics, amplitudes, bounds, loss denominators, pattern-loss weight, and optional depth taper are restored. `run_endpoint.slurm` selects the endpoint experiment checkpoint. The checkpoint SHA256 is checked before and after replay. No training state is written.

The baseline uses the same stepping path with identically zero NN outputs. The trained model applies the exact split-epsilon tanh wrapper. Diagnostics read the stored, masked and restaggered tendencies at the entering timestep index after each model step. Per-step reductions report signed mean, RMS, and fraction at least 95% of the bound. Time RMS is the square root of mean squared RMS, not a mean of RMS values.

The target is fixed at `(1 - 27/720) * initial_GLORYS + (27/720) * next_GLORYS`. Intermediate error curves show distance to this final target, not contemporaneous observation errors. Raw basin errors, pointwise basin RMSE, and normalized objective are distinct. The latter uses saved denominators and equal active basin-level weighting on the five trained levels. Both runs use the same denominators.

Spatial reductions use native tracer/U/V cell areas and masks, exclude two halo cells, and retain all depths. Basin rectangles match training. Snapshots are saved at steps 1, 9 and 27; complete scalar/basin-level histories are saved for every step. Integrated actual NN sources are compared with paired state changes; their difference is not a closed process budget. Vertical velocity maps use native interfaces, with corresponding tracer-level labels.

Outputs live in `scripts/paired_rollout/job_JOBID`: REPORT.md, PNG figures, CSV basin/depth statistics, NPZ arrays, summary.json, and metadata.json. Map color limits clip at the 99th percentile; NPZ data retain extrema. `plot.py OUTPUT` regenerates the report without unpickling checkpoints.

Validation: `python -B scripts/paired_rollout/test_metrics.py` checks area weights, masking, cancellation of mean bias despite nonzero RMSE, empty basin pairs, and shared normalization. Runtime checks enforce finite active fields, zero baseline corrections, and trained source bounds.

Additional plots:

- `plot_surface_basin_improvement.py` compares surface basin bias and RMSE against the no-NN replay.
- `plot_training_baseline.py` adds the fixed no-NN objective to training curves.
- `plot_heat_flux_equivalent.py OUTPUT` converts time-mean temperature tendencies to surface-layer and full-column heat-flux equivalents. The default color range is ±200 W/m²; `--color-limit 400` changes the display range and `--max-scale` uses each map's maximum absolute value. These options change plotting limits only; saved values and area-weighted statistics retain the full corrections.

## One-month extrapolation with frozen endpoint weights

`sbatch scripts/paired_rollout/run_month.slurm` runs two 720-hour forward trajectories from the same initialization: no NN and job 55286's update-80, 27-hour endpoint weights (lambda 1). It restores the checkpoint's normalization, split epsilons, smooth bounds and taper. No optimizer is constructed for training or updated. The forcing window covers the full horizon.

Hourly error metrics use GLORYS interpolation at hour/720. At hour 720 the reference is the next available monthly GLORYS field, without interpolation. This is the existing 30-model-day convention, not a claim that the monthly means are instantaneous observations or exactly 30 calendar days apart.

Outputs are isolated in `month_job_JOBID`. Histories and cumulative correction statistics are flushed daily; full field snapshots are saved daily and at hours 1, 27 and 720. A trajectory stops on nonfinite active state/correction values, nonpositive salinity, or directional CFL > 1; it saves the failed fields and reason while allowing the other trajectory to run. This CFL threshold is a diagnostic guard, not a proof of numerical stability. `plot_month.py OUTPUT` renders the histories and latest common-time comparison, even for an incomplete pair. Source checkpoint SHA256 is checked before and after inference.
