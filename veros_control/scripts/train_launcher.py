from veros import runtime_settings
import jax.profiler
runtime_settings.backend = "jax"
runtime_settings.device = "gpu"
runtime_settings.force_overwrite = False
runtime_settings.linear_solver = "scipy_jax"
import glob
from veros_control.setup.global_1deg_control.global_1deg_control import GlobalOneDegreeSetup
from veros_control.training.train import train
from veros_control.training.config import TrainConfig


def main():

    sim = GlobalOneDegreeSetup()
    sim.setup()

    config = TrainConfig(
        output_file='/Odyssey/private/e25cheve/control/veros_control/checkpoints/checkpoint.pkl',
        epochs=1,
        initialization_mode="glorys",
        epsilon=1e-9,
        n_steps=3,
        sample_steps=3,
        batch_size=1,
        grad_mode="chunked_vjp",
        TS=True,
        keys_loss=("temp",),
        lr=1e-9,
        control_mode="nn",
        train_delay_days=0.0,
        glorys_files=sorted(glob.glob(
        '/Odyssey/private/e25cheve/veros/veros_assets/global_1deg_realistic/glorys12_initial_conditions_40*'
    ))
    )

    train(sim, config)

if __name__ == "__main__":
    main()
