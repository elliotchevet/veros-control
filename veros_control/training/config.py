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
    conservative_mixing: bool = False
    use_one_hot: bool = False
    num_mlps: int = 1
    TS: bool = False
    seed: int = 0

    # training
    epochs: int = 10
    batch_size: int = 4
    lr: float = 1e-3
    overwrite_training: bool = False
    resume_training: bool = False
    resume_checkpoint_path: str | None = None
    save_every: int = 5

    # rollout
    n_steps: int = 120
    sample_steps: int = 120
    use_jax_checkpoint: bool = True

    # data
    restart_files: list[str] = field(default_factory=list)
    keys_loss: tuple[str, ...] = ("temp", "salt")

    # normalization
    normalize_inputs: bool = False
    scaling_method_features: str | None = None
    stats_features: object | None = None
