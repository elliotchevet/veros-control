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
        hidden_dim=1,
        depth=60,
        epochs=1,
        n_steps=2,
        sample_steps=2,
        batch_size=1,
        restart_interval_steps=2,
        grad_mode="manual_vjp",
        TS=False,
        keys_loss=("temp","salt"),
        lr=1e-3,
        control_mode="r_bot",
        restart_files=glob.glob(
        '/Odyssey/private/e25cheve/simu_veros/runs/ground_truth/restarts/*.h5'
    )
    )

    train(sim, config)

if __name__ == "__main__":
    main()
