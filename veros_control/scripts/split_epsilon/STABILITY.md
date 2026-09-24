# Bounded split-epsilon continuation

Rollback commit before this change: `ffcc16dbe57976ed475f2facbd2c5800832327c8`.

These options belong to `scripts/split_epsilon/continue_training.py`, not the generic benchmark CLI. Without them, existing correction behavior is unchanged.

Enable with `--bound-corrections --validate-candidates`.

For channel i, raw NN output z_i, and its epsilon e_i, apply

    tendency_i = B_i * tanh(e_i * z_i / B_i)
    B_i = max_correction_per_hour_i / 3600

Defaults, in T/S/u/v order: `--max-correction-per-hour 0.05 0.05 0.005 0.005`.
Temperature is deg C/hour, salinity g/kg/hour, and velocity (m/s)/hour.
At timestep dt seconds, the direct NN contribution is bounded by the listed value times dt/3600.
These are configurable conservative starting values, not physically calibrated limits. They bound NN forcing, not the total change caused by all physical processes. For small corrections the response is e_i*z_i. Changing epsilon changes small-signal gain, while changing B changes the saturation ceiling. The split epsilons remain 1e-6 for tracers and 1e-7 for momentum. Velocity restaggering averages bounded corrections; masks cannot increase them. Pointwise bounds are incompatible with the current conservative-column-mixing option, so that combination is rejected.

Validation checks the initial state and every rollout step for finite active temperature, salinity, velocities, density, potential density, stratification, TKE, EKE, enthalpy, and mixing coefficients. Salinity must be positive in interior ocean cells. Directional CFL using Veros CFLMonitor formulas must be <= `--max-rollout-cfl` (default 1). These are feasibility checks, not a complete proof of numerical stability. In particular a finite forward rollout does not guarantee all derivatives are finite.

The starting checkpoint is validated before differentiation. If invalid, stop: lowering the next optimizer step cannot repair the current trajectory. For a finite gradient, compute the AdamW direction and proposed moment state once. Replay candidate parameters with the proposed learning rate. If invalid, halve the learning rate and retry the same direction, up to `--candidate-max-retries` halvings (default 8, hence 9 attempts including the initial one). A successful reduction persists on following updates via a multiplier on the learning-rate schedule. Only an accepted candidate commits parameters and AdamW state. Exhaustion raises without overwriting the last checkpoint. Validation does not require a lower loss.

The epsilons, bounds, validation settings, retry count, actual learning rate, and persistent backoff multiplier are saved. Resume bounded checkpoints through this runner with the same bounds and epsilons. The saved optimizer state and original loss-normalization denominators are retained.

Nested gradient checkpointing is unchanged. Validation uses nested forward scans with the same chunk lengths, without gradients. It adds at least one forward rollout per optimizer update, plus retries if needed.

`validate_bounded_update.slurm` runs only update 44 from a copy of checkpoint 43. `bounded_to_100.slurm` is prepared for a separate continuation through update 100; it is not automatically submitted. Both write to new job directories and preserve their source checkpoint.
