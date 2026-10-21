# Optional surface corrections

`--surface-only` retains full-column inputs but predicts only surface corrections;
all deeper direct tendencies are exactly zero. Omit it for full-depth outputs.
Conservative mixing is incompatible with this mode. Checkpoints preserve the
architecture choice. Surface curriculum loss requires `--basin-levels 1`.

The surface curriculum defaults to minimum LR 1e-5 and loss-decrease rejection
from update 20. At the LR floor, finite increases may be accepted; invalid
physical rollouts remain rejected. Physical dynamics can propagate surface
changes downward; long-term stability is not guaranteed.
