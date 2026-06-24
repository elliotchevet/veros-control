from veros import runtime_settings

runtime_settings.backend = "jax"
runtime_settings.device = "gpu"
runtime_settings.force_overwrite = False
runtime_settings.linear_solver = "scipy_jax"

from global_1deg_control import GlobalOneDegreeControl
from veros_control.training.train import train
from veros_control.training.config import TrainConfig


def main():

    sim = GlobalOneDegreeControl()
    sim.setup()

    config = TrainConfig(
        epochs=2,
        n_steps=4,
        sample_steps=2,
        batch_size=1,
        lr=1e-4,
        ...
    )

    train(sim, config)


if __name__ == "__main__":
    main()
