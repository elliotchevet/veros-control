from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    output_file: str

    # model
    hidden_dim: int = 64
    depth: int = 2
    epsilon: float = 1e-6
    train_epsilon: bool = False
    use_lon_lat: bool = False
    use_forcings: bool = False
    conservative_mixing: bool = False
    use_one_hot: bool = False
    num_mlps: int = 1
    TS: bool = False
    zero_output_init: bool = True
    seed: int = 0
    control_mode: str = "A_h"  # "A_h" or "nn"
    A_h_init: float | None = None

    # training
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-3
    grad_clip_norm: float | None = None
    overwrite_training: bool = False
    resume_training: bool = False
    resume_checkpoint_path: str | None = None
    save_every: int = 5
    diagnostics_every: int = 1
    fail_on_nonfinite: bool = True

    # rollout
    n_steps: int = 72
    sample_steps: int = 72
    restart_interval_steps: int = 72
    train_delay_days: float = 15.0
    train_delay_use_nn: bool = False
    use_jax_checkpoint: bool = True
    grad_mode: str = "forward_jvp"  # "forward_jvp", "builtin", "manual_vjp", or "chunked_vjp"
    chunk_base: int = 3

    # data
    initialization_mode: str = "restart"  # "restart" or "glorys"
    restart_files: list[str] = field(default_factory=list)
    glorys_files: list[str] = field(default_factory=list)
    glorys_time_index: int = 0
    glorys_start_month: int = 1
    glorys_month_lengths: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    glorys_rollouts_per_month: int | None = None
    keys_loss: tuple[str, ...] = ("temp", "salt")

    # normalization
    normalize_inputs: bool = False
    scaling_method_features: str | None = None
    stats_features: object | None = None

