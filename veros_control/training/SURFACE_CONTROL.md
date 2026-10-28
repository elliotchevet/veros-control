# Optional surface corrections

`--surface-only` retains full-column inputs but predicts only surface corrections;
all deeper direct tendencies are exactly zero. Omit it for full-depth outputs.
Conservative mixing is incompatible with this mode. Checkpoints preserve the
architecture choice. Surface curriculum loss requires `--basin-levels 1`.

The surface curriculum defaults to minimum LR 1e-5 and loss-decrease rejection
from update 20. At the LR floor, finite increases may be accepted; invalid
physical rollouts remain rejected. Physical dynamics can propagate surface
changes downward; long-term stability is not guaranteed.


## Keep-update learning-rate scheduler

The curriculum runner optionally accepts `--loss-halving-continue` together with
`--loss-backtracking --validate-candidates --train-keep-best`. This selects an
alternative to repeated loss-based candidate backtracking:

- Make one candidate update and compare its loss to the pre-update loss on the
  same batch. Keep the update even when the loss-decrease test fails.
- A failed test halves the **next epoch's** LR once, bounded by `--train-min-lr`.
- Ten consecutive strict improvements increase the next LR by 50%, capped at
  `--train-lr`. Failure or a flat loss resets the streak; recovery resets it too.
- Tests start at `--lr-decrease-start-epoch` (surface default: 20). The default
  surface floor is 1e-5; use `--train-lr 1e-3` for the experiment's ceiling.
- There are no candidate retries in this mode. Nonfinite gradients, parameters,
  losses, or physically invalid candidate rollouts still abort before committing
  parameters or Adam state. Keeping an update applies only to the loss test.

Logs/history include applied and next LR, pass/fail, scheduler action and streak.
Same-stage resume restores the streak and rejects scheduler-setting changes.
A new curriculum horizon carries weights, Adam and LR but resets epoch/streak.
The original scheduler remains the default; existing checkpoints remain valid.

Only the original six rectangular basin masks are available. With the curriculum
runner, use `--surface-only --basin-mean-loss --basin-levels 1` for surface-only
outputs **and** surface-only loss. Full-column inputs, including optional forcing
and location inputs, are retained. Omit `--surface-only` for full-depth outputs.

If `SURFACE_RESTORING_METADATA` names a JSON file, its contents are recorded as
`restoring_configuration` in checkpoints. This records experiment provenance;
it does not change the physical restoring coefficients.
