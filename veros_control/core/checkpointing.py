import os
import pickle


def load_checkpoint(path):
    if not path:
        return None

    if not os.path.exists(path):
        print(f"NN checkpoint not found: {path}")
        return None

    print(f"Loading NN checkpoint from {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(path, checkpoint):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def checkpoint_model_kwargs(checkpoint, defaults):
    kwargs = dict(defaults)
    if checkpoint is None:
        return kwargs

    kwargs["surface_only"] = checkpoint.get("surface_only", False)
    kwargs["residual_blocks"] = checkpoint.get("residual_blocks", 0)
    key_map = {
        "hidden_dim_nn": "hidden_dim",
        "depth_nn": "depth",
        "epsilon": "init_eps",
        "train_epsilon": "train_epsilon",
        "use_lon_lat": "use_lon_lat",
        "use_forcings": "use_forcings",
        "conservative_mixing": "conservative_mixing",
        "use_one_hot": "use_one_hot",
        "num_mlps": "num_mlps",
        "TS": "TS",
        "zero_output_init": "zero_output_init",
    }

    for checkpoint_key, model_key in key_map.items():
        if checkpoint_key in checkpoint:
            kwargs[model_key] = checkpoint[checkpoint_key]

    return kwargs

