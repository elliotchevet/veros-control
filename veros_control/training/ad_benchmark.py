import argparse
import gc
import glob
import os
import pickle
import re
import subprocess
import sys
import threading
import time
from functools import partial

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

import jax

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from veros import runtime_settings


def _truthy_env(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _diskless_mode_requested():
    return (
        "--diskless-mode" in sys.argv
        or _truthy_env("VEROS_CONTROL_DISKLESS_MODE")
        or _truthy_env("VEROS_DISKLESS_MODE")
    )


def set_runtime_setting(name, value):
    try:
        setattr(runtime_settings, name, value)
    except RuntimeError as exc:
        if getattr(runtime_settings, name) != value:
            raise exc


set_runtime_setting("backend", "jax")
set_runtime_setting("device", "gpu")
set_runtime_setting("force_overwrite", False)
set_runtime_setting("linear_solver", "scipy_jax")
if _diskless_mode_requested():
    set_runtime_setting("diskless_mode", True)

from veros_control.core.checkpointing import checkpoint_model_kwargs, load_checkpoint
from veros_control.core.forcing import make_batched_model, predict_nn_forcing
from veros_control.core.model import VerticalCorrectionMLP
from veros_control.core.preprocessing import FORCING_FEATURE_NAMES
from veros_control.setup.global_1deg_control.global_1deg_control import GlobalOneDegreeSetup
from veros_control.training.train import (
    disable_diagnostics,
    get_prognostic_keys,
    inject_dynamic_vars,
    load_glorys_snapshot,
    load_glorys_state,
    load_restart_state,
    make_chunk_sizes,
    zero_tendency_sources,
)
from veros_control.training.build_feature_stats import load_feature_stats_file
from veros_control.training.loss import compute_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare JVP and chunked VJP gradients of the Veros A_h control."
    )
    parser.add_argument(
        "--n-step-powers",
        nargs=2,
        type=int,
        default=(1, 8),
        metavar=("START", "STOP"),
        help="Run n_steps = 3^START through 3^STOP inclusive. Default: 1 8.",
    )
    parser.add_argument(
        "--n-steps",
        nargs="+",
        type=int,
        default=None,
        help="Explicit n_steps values. Overrides --n-step-powers.",
    )
    parser.add_argument("--sample-steps", type=int, default=2)
    parser.add_argument("--chunk-base", type=int, default=3)
    parser.add_argument(
        "--diskless-mode",
        action="store_true",
        help="Run Veros in diskless mode to suppress model-side disk output.",
    )
    parser.add_argument(
        "--method-pair",
        choices=("jvp-vjp", "jvp-jvp", "vjp-only"),
        default="jvp-vjp",
        help="Compare JVP against chunked VJP, run the same JVP twice, or run VJP only.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("A_h", "nn_epsilon", "nn_param", "nn_param_vector"),
        default="A_h",
        help="Scalar sensitivity to benchmark. Default: A_h.",
    )
    parser.add_argument("--A-h", type=float, default=5e4)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=None,
        help="Epsilon values to sweep in nn_epsilon mode. Defaults to --epsilon.",
    )
    parser.add_argument("--nn-checkpoint", default=None)
    parser.add_argument(
        "--feature-stats-file",
        default=None,
        help=(
            "Optional pickle from build_feature_stats.py. Overrides checkpoint stats "
            "and enables input normalization."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--use-lon-lat", action="store_true")
    parser.add_argument("--use-forcings", action="store_true")
    parser.add_argument("--conservative-mixing", action="store_true")
    parser.add_argument("--use-one-hot", action="store_true")
    parser.add_argument("--num-mlps", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--TS", action="store_true")
    parser.add_argument("--nn-chunk-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nn-param-leaf-index",
        type=int,
        default=0,
        help="Parameter pytree leaf to use for --control-mode nn_param. Default: 0.",
    )
    parser.add_argument(
        "--nn-param-flat-index",
        type=int,
        default=0,
        help="Flat scalar index inside the selected parameter leaf for --control-mode nn_param. Default: 0.",
    )
    parser.add_argument(
        "--nn-param-count",
        type=int,
        default=10,
        help="Number of random NN scalar parameters for --control-mode nn_param_vector. Default: 10.",
    )
    parser.add_argument(
        "--nn-param-path-filter",
        default=None,
        help=(
            "Restrict random NN scalar parameters to leaves whose path contains this text, "
            "for example Dense_2 for the output layer of a depth=2 MLP."
        ),
    )
    parser.add_argument(
        "--nn-param-seed",
        type=int,
        default=None,
        help="Seed used to select random NN scalar parameters. Default: --seed.",
    )
    parser.add_argument(
        "--zero-output-init",
        action="store_true",
        help="Initialize NN output Dense layers with zero kernel and bias for fresh parameters.",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--restart-file",
        default=None,
        help="Restart file, or glob pattern, to use as the initial benchmark state.",
    )
    parser.add_argument(
        "--restart-index",
        type=int,
        default=0,
        help="Index to select when --restart-file is a glob pattern. Default: 0.",
    )
    parser.add_argument(
        "--glorys-file",
        default=None,
        help="GLORYS file, or glob pattern, to use as the initial benchmark state.",
    )
    parser.add_argument(
        "--glorys-index",
        type=int,
        default=0,
        help="Index to select when --glorys-file is a glob pattern. Default: 0.",
    )
    parser.add_argument("--glorys-time-index", type=int, default=0)
    parser.add_argument("--glorys-month-index", type=int, default=0)
    parser.add_argument("--glorys-start-month", type=int, default=1)
    parser.add_argument(
        "--glorys-target-file",
        default=None,
        help="Optional GLORYS target file/glob for --loss-mode glorys_target.",
    )
    parser.add_argument(
        "--glorys-target-index",
        type=int,
        default=0,
        help="Index to select when --glorys-target-file is a glob pattern. Default: 0.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("u2", "glorys_target"),
        default="u2",
        help="Loss used by the AD benchmark. Default: old u**2 benchmark loss.",
    )
    parser.add_argument(
        "--loss-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="How to reduce sampled losses over the rollout. Default: sum.",
    )
    parser.add_argument(
        "--keys-loss",
        nargs="+",
        default=("temp", "salt"),
        help="Variables for --loss-mode glorys_target. Default: temp salt.",
    )
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load existing result pickle files and regenerate figures without running JAX AD.",
    )
    parser.add_argument(
        "--clear-between-epsilons",
        action="store_true",
        help="Clear JAX caches and run Python GC before each epsilon in a multi-epsilon run.",
    )
    parser.add_argument(
        "--no-freeze-vars",
        dest="freeze_vars",
        action="store_false",
        help="Carry the full Veros state in chunked VJP instead of freezing variables outside the carry.",
    )
    parser.add_argument(
        "--save-sample-trajectories",
        action="store_true",
        help="Save JVP and VJP u fields at every loss sample step for manual loss checks.",
    )
    parser.add_argument(
        "--trajectory-folder",
        default=None,
        help="Folder for sampled u trajectory files. Default: <output-folder>/sample_trajectories.",
    )
    parser.set_defaults(freeze_vars=True)
    return parser.parse_args()


def get_gpu_name():
    try:
        devices = jax.devices()
        if not devices:
            return "unknown (no JAX devices found)"
        dev = devices[0]
        name = getattr(dev, "device_kind", None) or str(dev)
        platform = getattr(dev, "platform", "unknown")
        return f"{name} ({platform}, {len(devices)} device(s) visible)"
    except Exception as exc:
        return f"unknown (error querying JAX devices: {exc})"


def get_gpu_tag():
    try:
        devices = jax.devices()
        if not devices:
            return "unknown_gpu"
        dev = devices[0]
        raw_name = getattr(dev, "device_kind", None) or str(dev)
        stats = dev.memory_stats() or {}
        vram_gb = None
        if stats.get("bytes_limit"):
            vram_gb = round(stats["bytes_limit"] / (1024**3))
        safe_name = re.sub(r"[^A-Za-z0-9\-]+", "_", raw_name).strip("_")
        if vram_gb is not None:
            return f"{safe_name}_{vram_gb}GB"
        return safe_name
    except Exception as exc:
        return f"unknown_gpu_error_{type(exc).__name__}"


def make_steps_list(args):
    if args.n_steps is not None:
        return args.n_steps
    start_power, stop_power = args.n_step_powers
    if start_power > stop_power:
        raise ValueError("--n-step-powers START must be <= STOP")
    return [3**power for power in range(start_power, stop_power + 1)]


def pure(state, step):
    next_state = state.copy()
    step(next_state)
    return next_state


def set_var(var_name, state, var_value):
    next_state = state.copy()
    with next_state.variables.unlock():
        setattr(next_state.variables, var_name, var_value)
    return next_state


def agg_sum(state, key_sum="u"):
    return ((getattr(state.variables, key_sum)[:, :, :] - 0) ** 2).mean()


def reduce_sampled_losses(losses, iterations, sample_steps, loss_reduction):
    total = jnp.sum(losses)
    if loss_reduction == "sum":
        return total
    if loss_reduction == "mean":
        sample_count = max(int(iterations) // int(sample_steps), 1)
        return total / jnp.asarray(sample_count, dtype=total.dtype)
    raise ValueError(f"Unsupported loss reduction: {loss_reduction}")


IGNORED_INPUT_SUFFIXES = (".lock",)
DEFAULT_GLORYS_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def resolve_input_file(file_pattern, file_index, label):
    if file_pattern is None:
        return None

    matches = [
        path
        for path in sorted(glob.glob(file_pattern))
        if not str(path).endswith(IGNORED_INPUT_SUFFIXES)
    ]
    if not matches:
        if os.path.exists(file_pattern) and not str(file_pattern).endswith(IGNORED_INPUT_SUFFIXES):
            return file_pattern
        raise FileNotFoundError(f"No {label} file matches {file_pattern!r}")

    try:
        return matches[file_index]
    except IndexError as exc:
        raise IndexError(
            f"{label} index {file_index} is out of range for "
            f"{len(matches)} files matching {file_pattern!r}"
        ) from exc


def resolve_restart_file(restart_file, restart_index):
    return resolve_input_file(restart_file, restart_index, "restart")


class GlorysInitConfig:
    glorys_month_lengths = DEFAULT_GLORYS_MONTH_LENGTHS

    def __init__(self, glorys_start_month):
        self.glorys_start_month = glorys_start_month


def init_simulation(
    warmup,
    restart_file=None,
    glorys_file=None,
    glorys_time_index=0,
    glorys_month_index=0,
    glorys_start_month=1,
):
    if restart_file is not None and glorys_file is not None:
        raise ValueError("Use either --restart-file or --glorys-file, not both")

    simulation = GlobalOneDegreeSetup(skip_setup_initial_conditions=glorys_file is not None)
    simulation.setup()
    disable_diagnostics(simulation)

    if restart_file is not None:
        print(f"Loading initial state from restart: {restart_file}")
        load_restart_state(simulation, restart_file)
    elif glorys_file is not None:
        print(f"Loading initial state from GLORYS: {glorys_file}")
        simulation.state = load_glorys_state(
            simulation,
            glorys_file,
            time_index=glorys_time_index,
            month_index=glorys_month_index,
            config=GlorysInitConfig(glorys_start_month),
        )

    disable_diagnostics(simulation)

    for _ in range(warmup):
        simulation.step(simulation.state)

    disable_diagnostics(simulation)

    return simulation


def get_all_active_keys(state):
    return [
        key
        for key, meta in state.var_meta.items()
        if meta.active
    ]


def init_nn_benchmark(simulation, args):
    checkpoint = load_checkpoint(args.nn_checkpoint)
    settings = simulation.state.settings
    defaults = {
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "nz": settings.nz,
        "nb_profiles": 4,
        "init_eps": 1.0,
        "train_epsilon": False,
        "use_lon_lat": args.use_lon_lat,
        "use_forcings": args.use_forcings,
        "conservative_mixing": args.conservative_mixing,
        "use_one_hot": args.use_one_hot,
        "num_mlps": args.num_mlps,
        "TS": args.TS,
        "zero_output_init": args.zero_output_init,
    }
    model_kwargs = checkpoint_model_kwargs(checkpoint, defaults)
    model_kwargs["nz"] = settings.nz
    model_kwargs["init_eps"] = 1.0
    model_kwargs["train_epsilon"] = False
    args.use_forcings = bool(model_kwargs.get("use_forcings", args.use_forcings))
    with settings.unlock():
        settings.use_forcings = args.use_forcings

    model = VerticalCorrectionMLP(**model_kwargs)
    model_batched_unscaled = make_batched_model(model)

    if checkpoint is not None and "params" in checkpoint:
        params = checkpoint["params"]
        stats = checkpoint.get("stats_features", checkpoint.get("features_stats"))
        print(f"Using NN checkpoint parameters from {args.nn_checkpoint}")
    else:
        n_input_channels = 4 + len(FORCING_FEATURE_NAMES) if args.use_forcings else 4
        x_tmp = jnp.zeros((settings.nz, n_input_channels))
        coords_tmp = jnp.zeros((2, 4))
        dz_tmp = jnp.zeros((settings.nz,))
        params = model.init(jax.random.PRNGKey(args.seed), x_tmp, coords_tmp, dz_tmp)
        stats = None
        print(f"Initialized fresh NN parameters with seed {args.seed}")

    normalize_inputs = None
    scaling_method_features = None
    if checkpoint is not None:
        normalize_inputs = checkpoint.get("normalize_inputs")
        scaling_method_features = checkpoint.get("scaling_method_features")

    stats_file = getattr(args, "feature_stats_file", None)
    if stats_file:
        stats, stats_method, _ = load_feature_stats_file(stats_file)
        normalize_inputs = True
        if stats_method is not None:
            scaling_method_features = stats_method
        print(f"Using NN input feature stats from {stats_file}")

    if stats is not None:
        with settings.unlock():
            if normalize_inputs is not None:
                settings.normalize_inputs = bool(normalize_inputs)
            if scaling_method_features is not None:
                settings.scaling_method_features = scaling_method_features
        setattr(args, "normalize_inputs", bool(settings.normalize_inputs))
        setattr(args, "scaling_method_features", settings.scaling_method_features)
        print(
            "NN input normalization: "
            f"normalize_inputs={settings.normalize_inputs}, "
            f"scaling_method_features={settings.scaling_method_features}"
        )
    print(f"NN forcing inputs: use_forcings={settings.use_forcings}")

    return params, model_batched_unscaled, stats, model_kwargs


def _finite_array_stats(value):
    value = jnp.asarray(value)
    finite = jnp.isfinite(value)
    finite_count = jnp.sum(finite)
    total_count = value.size
    safe_value = jnp.where(finite, value, 0.0)
    denom = jnp.maximum(finite_count, 1)
    min_value = jnp.where(
        finite_count > 0,
        jnp.min(jnp.where(finite, value, jnp.inf)),
        jnp.nan,
    )
    max_value = jnp.where(
        finite_count > 0,
        jnp.max(jnp.where(finite, value, -jnp.inf)),
        jnp.nan,
    )
    mean_value = jnp.sum(safe_value) / denom
    rms_value = jnp.sqrt(jnp.sum(safe_value * safe_value) / denom)
    max_abs_value = jnp.where(
        finite_count > 0,
        jnp.max(jnp.where(finite, jnp.abs(value), -jnp.inf)),
        jnp.nan,
    )
    return {
        "finite": finite_count == total_count,
        "bad_count": total_count - finite_count,
        "total_count": jnp.asarray(total_count),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "rms": rms_value,
        "max_abs": max_abs_value,
    }


def print_nn_correction_stats(simulation, nn_params, model_batched_unscaled, stats, epsilon, label=None):
    def model_batched(params_arg, features, coordinates, dzt):
        return epsilon * model_batched_unscaled(
            params_arg,
            features,
            coordinates,
            dzt,
        )

    state = simulation.state
    dtemp, dsalt, du, dv = predict_nn_forcing(
        state,
        nn_params,
        model_batched,
        stats=stats,
        normalize=state.settings.normalize_inputs,
        scaling_method=state.settings.scaling_method_features,
        use_forcings=state.settings.use_forcings,
    )
    tendency_stats = jax.tree_util.tree_map(
        _finite_array_stats,
        {
            "dtemp": dtemp,
            "dsalt": dsalt,
            "du": du,
            "dv": dv,
        },
    )
    tendency_stats = jax.device_get(tendency_stats)

    if label is None:
        label = "NN correction tendencies"
    print(f"\n{label}")
    for name in ("dtemp", "dsalt", "du", "dv"):
        item = tendency_stats[name]
        print(
            f"  {name}: finite={bool(item['finite'])} "
            f"bad={int(item['bad_count'])}/{int(item['total_count'])} "
            f"min={float(item['min']):.6e} max={float(item['max']):.6e} "
            f"mean={float(item['mean']):.6e} rms={float(item['rms']):.6e} "
            f"max_abs={float(item['max_abs']):.6e}"
        )
    free_buffers((dtemp, dsalt, du, dv))
    return tendency_stats


def key_entry_to_string(entry):
    if hasattr(entry, "key"):
        return str(entry.key)
    if hasattr(entry, "idx"):
        return str(entry.idx)
    if hasattr(entry, "name"):
        return str(entry.name)
    return str(entry)


def param_path_to_string(path):
    return "/".join(key_entry_to_string(entry) for entry in path) or "<root>"


def describe_param_scalar(params, leaf_index=0, flat_index=0):
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(params)
    if not path_leaves:
        raise ValueError("NN parameter tree has no leaves")
    if leaf_index < 0 or leaf_index >= len(path_leaves):
        raise IndexError(
            f"--nn-param-leaf-index {leaf_index} out of range for {len(path_leaves)} parameter leaves"
        )

    path, leaf = path_leaves[leaf_index]
    flat_leaf = jnp.ravel(leaf)
    if flat_index < 0 or flat_index >= flat_leaf.size:
        raise IndexError(
            f"--nn-param-flat-index {flat_index} out of range for selected leaf "
            f"with {flat_leaf.size} scalar values"
        )

    return {
        "leaf_index": leaf_index,
        "flat_index": flat_index,
        "path": param_path_to_string(path),
        "shape": tuple(leaf.shape),
        "dtype": str(leaf.dtype),
        "initial_value": float(flat_leaf[flat_index]),
        "n_leaves": len(path_leaves),
        "leaf_size": int(flat_leaf.size),
    }


def select_random_param_scalars(params, count, seed, path_filter=None):
    if count <= 0:
        raise ValueError("--nn-param-count must be positive")

    path_leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    indexed_path_leaves = [
        (leaf_index, path, leaf)
        for leaf_index, (path, leaf) in enumerate(path_leaves)
        if path_filter is None or path_filter in param_path_to_string(path)
    ]
    if not indexed_path_leaves:
        raise ValueError(f"No NN parameter leaves match --nn-param-path-filter={path_filter!r}")

    leaf_sizes = [int(jnp.ravel(leaf).size) for _, _, leaf in indexed_path_leaves]
    total_size = sum(leaf_sizes)
    if count > total_size:
        raise ValueError(
            f"--nn-param-count {count} is larger than the selected scalar NN parameter count {total_size}"
        )

    rng = np.random.default_rng(seed)
    global_indices = np.sort(rng.choice(total_size, size=count, replace=False))
    offsets = np.cumsum([0] + leaf_sizes)

    selections = []
    for global_index in global_indices:
        selected_leaf_index = int(np.searchsorted(offsets[1:], global_index, side="right"))
        flat_index = int(global_index - offsets[selected_leaf_index])
        leaf_index, path, leaf = indexed_path_leaves[selected_leaf_index]
        flat_leaf = jnp.ravel(leaf)
        selections.append(
            {
                "global_index": int(global_index),
                "leaf_index": leaf_index,
                "flat_index": flat_index,
                "path": param_path_to_string(path),
                "shape": tuple(leaf.shape),
                "dtype": str(leaf.dtype),
                "initial_value": float(flat_leaf[flat_index]),
                "leaf_size": int(flat_leaf.size),
                "path_filter": path_filter,
                "selected_scalar_count": total_size,
            }
        )

    return selections


def get_param_scalar(params, leaf_index, flat_index):
    leaves, _ = jax.tree_util.tree_flatten(params)
    return jnp.ravel(leaves[leaf_index])[flat_index]


def set_param_scalar(params, leaf_index, flat_index, scalar_value):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    leaf = leaves[leaf_index]
    flat_leaf = jnp.ravel(leaf)
    flat_leaf = flat_leaf.at[flat_index].set(scalar_value)
    leaves[leaf_index] = jnp.reshape(flat_leaf, leaf.shape)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def make_param_tangent(params, leaf_index, flat_index):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    tangent_leaves = []
    for current_leaf_index, leaf in enumerate(leaves):
        tangent = jnp.zeros_like(leaf)
        if current_leaf_index == leaf_index:
            flat_tangent = jnp.ravel(tangent)
            flat_tangent = flat_tangent.at[flat_index].set(1.0)
            tangent = jnp.reshape(flat_tangent, leaf.shape)
        tangent_leaves.append(tangent)
    return jax.tree_util.tree_unflatten(treedef, tangent_leaves)


def selected_param_vector(params, selections):
    return jnp.stack(
        [
            get_param_scalar(params, item["leaf_index"], item["flat_index"])
            for item in selections
        ]
    )


def free_buffers(outputs):
    for leaf in jax.tree_util.tree_leaves(outputs):
        if hasattr(leaf, "delete"):
            leaf.delete()


def print_memory_stats(label):
    stats = jax.devices()[0].memory_stats() or {}
    bytes_in_use = stats.get("bytes_in_use")
    peak_bytes_in_use = stats.get("peak_bytes_in_use")
    live_arrays = len(jax.live_arrays())
    if bytes_in_use is None or peak_bytes_in_use is None:
        print(f"  [{label}] memory_stats unavailable  live_arrays={live_arrays}")
        return
    print(
        f"  [{label}] bytes_in_use={bytes_in_use:,}  "
        f"peak={peak_bytes_in_use:,}  live_arrays={live_arrays}"
    )


def jax_memory_snapshot():
    stats = jax.devices()[0].memory_stats() or {}
    bytes_in_use = stats.get("bytes_in_use")
    peak_bytes_in_use = stats.get("peak_bytes_in_use")
    return {
        "bytes_in_use": bytes_in_use if bytes_in_use is not None else np.nan,
        "peak_bytes_in_use": peak_bytes_in_use if peak_bytes_in_use is not None else np.nan,
        "live_arrays": len(jax.live_arrays()),
    }


def _query_current_process_gpu_memory_mib():
    pid = os.getpid()
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        total_mib = 0.0
        found = False
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                row_pid = int(parts[0])
                used_mib = float(parts[1])
            except ValueError:
                continue
            if row_pid == pid:
                total_mib += used_mib
                found = True
        if found:
            return total_mib
    except Exception:
        pass

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    gpu_id = visible_devices[0].strip() or "0"
    commands = [
        [
            "nvidia-smi",
            "-i",
            gpu_id,
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
    ]
    for command in commands:
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
            values = []
            for line in output.splitlines():
                try:
                    values.append(float(line.strip().split()[0]))
                except (IndexError, ValueError):
                    continue
            if values:
                return values[0] if command[1:2] == ["-i"] else max(values)
        except Exception:
            continue
    return np.nan


class GPUMemoryMonitor:
    def __init__(self, interval=0.5):
        self.interval = float(interval)
        self.samples_mib = []
        self.before_mib = np.nan
        self.after_mib = np.nan
        self._stop_event = threading.Event()
        self._thread = None

    def __enter__(self):
        self.before_mib = _query_current_process_gpu_memory_mib()
        self.samples_mib = []
        self._stop_event.clear()

        def poll():
            while not self._stop_event.is_set():
                self.samples_mib.append(_query_current_process_gpu_memory_mib())
                self._stop_event.wait(self.interval)

        self._thread = threading.Thread(target=poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval * 2.0, 1.0))
        self.after_mib = _query_current_process_gpu_memory_mib()

    @property
    def peak_mib(self):
        values = [
            value
            for value in [self.before_mib, self.after_mib, *self.samples_mib]
            if np.isfinite(value)
        ]
        return max(values) if values else np.nan


def format_memory_mib(value):
    if not np.isfinite(value):
        return "n/a"
    return f"{value / 1024.0:.2f} GiB"


def clear_after_ad(label):
    jax.clear_caches()
    gc.collect()
    print_memory_stats(label)


def save_results(results, pkl_path):
    tmp_path = pkl_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp_path, pkl_path)
    print(f"  [checkpoint] saved progress -> {pkl_path}")


def manual_u2_loss(u_array):
    return float(np.mean(np.asarray(u_array, dtype=np.float64) ** 2))


class SampleTrajectoryWriter:
    def __init__(self, path, iterations, sample_steps, u_template, loss_reduction="sum"):
        self.path = path
        self.tmp_path = path + ".tmp"
        self.iterations = int(iterations)
        self.sample_steps = int(sample_steps)
        self.loss_reduction = loss_reduction
        self.sample_indices = np.arange(sample_steps, iterations + 1, sample_steps, dtype=np.int64)
        self.u_shape = tuple(u_template.shape)
        self.u_dtype = np.dtype(u_template.dtype)
        self.h5file = None

    def __enter__(self):
        import h5py

        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.h5file = h5py.File(self.tmp_path, "w")
        self.h5file.attrs["iterations"] = self.iterations
        self.h5file.attrs["sample_steps"] = self.sample_steps
        self.h5file.attrs["loss_reduction"] = self.loss_reduction
        self.h5file.attrs["trajectory_storage_mode"] = "first_callback_value"
        self.h5file.create_dataset("sample_step_indices", data=self.sample_indices)
        for method in ("jvp", "vjp"):
            group = self.h5file.create_group(method)
            u_dataset_kwargs = {}
            if len(self.sample_indices):
                u_dataset_kwargs["chunks"] = (1,) + self.u_shape
            group.create_dataset(
                "u",
                shape=(len(self.sample_indices),) + self.u_shape,
                dtype=self.u_dtype,
                **u_dataset_kwargs,
            )
            group.create_dataset(
                "manual_mean_u2",
                shape=(len(self.sample_indices),),
                dtype=np.float64,
                fillvalue=np.nan,
            )
            group.create_dataset("write_count", shape=(len(self.sample_indices),), dtype=np.int32)
        return self

    def callback(self, method):
        def write_sample(step_idx, u_value):
            sample_index = int(step_idx) // self.sample_steps - 1
            if sample_index < 0 or sample_index >= len(self.sample_indices):
                return
            group = self.h5file[method]
            previous_write_count = int(group["write_count"][sample_index])
            if previous_write_count == 0:
                u_array = np.asarray(u_value)
                group["u"][sample_index, ...] = u_array
                group["manual_mean_u2"][sample_index] = manual_u2_loss(u_array)
            group["write_count"][sample_index] = previous_write_count + 1
            self.h5file.flush()

        return write_sample

    def set_ad_losses(self, jvp_ad_loss, vjp_ad_loss):
        self.h5file.attrs["jvp_ad_loss"] = float(jvp_ad_loss)
        self.h5file.attrs["vjp_ad_loss"] = float(vjp_ad_loss)

    def finalize(self):
        if len(self.sample_indices):
            jvp_losses = self.h5file["jvp/manual_mean_u2"][:]
            vjp_losses = self.h5file["vjp/manual_mean_u2"][:]
            all_samples_written = True
            for method in ("jvp", "vjp"):
                write_counts = self.h5file[f"{method}/write_count"][:]
                self.h5file.attrs[f"{method}_min_write_count"] = int(np.min(write_counts))
                self.h5file.attrs[f"{method}_max_write_count"] = int(np.max(write_counts))
                self.h5file.attrs[f"{method}_all_samples_written"] = bool(np.all(write_counts > 0))
                all_samples_written = all_samples_written and bool(np.all(write_counts > 0))
        else:
            jvp_losses = np.asarray([], dtype=np.float64)
            vjp_losses = np.asarray([], dtype=np.float64)
            all_samples_written = True
            for method in ("jvp", "vjp"):
                self.h5file.attrs[f"{method}_min_write_count"] = 0
                self.h5file.attrs[f"{method}_max_write_count"] = 0
                self.h5file.attrs[f"{method}_all_samples_written"] = True

        jvp_manual_sum = float(np.sum(jvp_losses))
        vjp_manual_sum = float(np.sum(vjp_losses))
        if self.loss_reduction == "mean":
            sample_count = max(len(self.sample_indices), 1)
            jvp_manual_reduced = jvp_manual_sum / sample_count
            vjp_manual_reduced = vjp_manual_sum / sample_count
        else:
            jvp_manual_reduced = jvp_manual_sum
            vjp_manual_reduced = vjp_manual_sum

        self.h5file.attrs["jvp_manual_loss_sum"] = jvp_manual_sum
        self.h5file.attrs["vjp_manual_loss_sum"] = vjp_manual_sum
        self.h5file.attrs["jvp_manual_loss_reduced"] = jvp_manual_reduced
        self.h5file.attrs["vjp_manual_loss_reduced"] = vjp_manual_reduced
        if "jvp_ad_loss" in self.h5file.attrs:
            self.h5file.attrs["jvp_manual_minus_ad_loss"] = (
                float(self.h5file.attrs["jvp_manual_loss_reduced"])
                - float(self.h5file.attrs["jvp_ad_loss"])
            )
        if "vjp_ad_loss" in self.h5file.attrs:
            self.h5file.attrs["vjp_manual_minus_ad_loss"] = (
                float(self.h5file.attrs["vjp_manual_loss_reduced"])
                - float(self.h5file.attrs["vjp_ad_loss"])
            )

        self.h5file.create_dataset("manual_loss_diff_per_sample", data=vjp_losses - jvp_losses)
        if all_samples_written:
            max_abs_u_diff = 0.0
            squared_diff_sum = 0.0
            value_count = 0
            for sample_index in range(len(self.sample_indices)):
                diff = self.h5file["vjp/u"][sample_index, ...] - self.h5file["jvp/u"][sample_index, ...]
                max_abs_u_diff = max(max_abs_u_diff, float(np.nanmax(np.abs(diff))))
                squared_diff_sum += float(np.nansum(np.asarray(diff, dtype=np.float64) ** 2))
                value_count += diff.size
            self.h5file.attrs["max_abs_u_diff"] = max_abs_u_diff
            self.h5file.attrs["rms_u_diff"] = (
                float(np.sqrt(squared_diff_sum / value_count)) if value_count else np.nan
            )
        else:
            self.h5file.attrs["max_abs_u_diff"] = np.nan
            self.h5file.attrs["rms_u_diff"] = np.nan

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.finalize()
        finally:
            self.h5file.close()
        if exc_type is None:
            os.replace(self.tmp_path, self.path)
            print(f"  [trajectory] saved sampled JVP/VJP u trajectories -> {self.path}")
        return False


def maybe_write_sample_u(sample_callback, is_sample_step, step_idx, state):
    if sample_callback is None:
        return

    def do_write(_):
        jax.debug.callback(sample_callback, step_idx, state.variables.u, ordered=True)
        return jnp.asarray(0, dtype=jnp.int8)

    def skip_write(_):
        return jnp.asarray(0, dtype=jnp.int8)

    jax.lax.cond(is_sample_step, do_write, skip_write, operand=None)


class A_hJVPScan:
    def __init__(self, step_function, agg_function, loss_reduction="sum"):
        self.step_function = partial(pure, step=step_function)
        self.agg_function = agg_function
        self.loss_reduction = loss_reduction

    def g(self, state, A_h_value, iterations=1, sample_steps=1):
        def forward_trajectory(var_value):
            init_state = set_var("A_h", state, var_value)

            def scan_step(carry_state, step_idx):
                next_state = self.step_function(carry_state)
                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda s: self.agg_function(s),
                    lambda s: jnp.asarray(0.0),
                    next_state,
                )
                return next_state, step_loss

            step_indices = jnp.arange(1, iterations + 1)
            _, all_step_losses = jax.lax.scan(
                scan_step,
                init_state,
                xs=step_indices,
                length=iterations,
            )
            return reduce_sampled_losses(all_step_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_A_h = jax.jvp(
            forward_trajectory,
            (A_h_value,),
            (jnp.ones_like(A_h_value),),
        )
        return loss, grad_A_h


class A_hChunkedVJP:
    def __init__(self, step_function, agg_function, loss_reduction="sum"):
        self.step_function = partial(pure, step=step_function)
        self.agg_function = agg_function
        self.loss_reduction = loss_reduction

    def g(self, full_state, A_h_value, prognostic_keys, chunk_sizes, sample_steps=1):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size

        def forward_trajectory(var_value):
            init_carry = {
                key: getattr(full_state.variables, key)
                for key in prognostic_keys
            }
            is_prognostic = "A_h" in prognostic_keys
            if is_prognostic:
                init_carry["A_h"] = var_value

            @jax.checkpoint
            def innermost_step(inner_tuple, step_idx):
                dynamic_vars, step_A_h = inner_tuple

                updates = dict(dynamic_vars)
                if not is_prognostic:
                    updates["A_h"] = step_A_h

                curr_state = inject_dynamic_vars(full_state, updates)
                next_state = self.step_function(curr_state)
                next_carry = {
                    key: getattr(next_state.variables, key)
                    for key in prognostic_keys
                }

                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda state: self.agg_function(state),
                    lambda state: jnp.asarray(0.0),
                    next_state,
                )
                return (next_carry, step_A_h), step_loss

            def make_chunk_func(func_to_scan, length):
                def chunk_def(carry, xs_chunk):
                    out, chunk_losses = jax.lax.scan(
                        func_to_scan,
                        carry,
                        xs=xs_chunk,
                        length=length,
                        unroll=1,
                    )
                    return out, chunk_losses

                return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

            current_func = innermost_step
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            step_indices = jnp.arange(1, iterations + 1).reshape(chunk_sizes)
            _, all_chunk_losses = jax.lax.scan(
                current_func,
                (init_carry, var_value),
                xs=step_indices,
                length=chunk_sizes[0],
                unroll=1,
            )
            return reduce_sampled_losses(all_chunk_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_A_h = jax.value_and_grad(forward_trajectory)(A_h_value)
        return loss, grad_A_h


class A_hFullStateChunkedVJP:
    def __init__(self, step_function, agg_function, loss_reduction="sum"):
        self.step_function = partial(pure, step=step_function)
        self.agg_function = agg_function
        self.loss_reduction = loss_reduction

    def g(self, full_state, A_h_value, prognostic_keys, chunk_sizes, sample_steps=1):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size

        def forward_trajectory(var_value):
            init_state = set_var("A_h", full_state, var_value)

            @jax.checkpoint
            def innermost_step(carry_state, step_idx):
                next_state = self.step_function(carry_state)
                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda state: self.agg_function(state),
                    lambda state: jnp.asarray(0.0),
                    next_state,
                )
                return next_state, step_loss

            def make_chunk_func(func_to_scan, length):
                def chunk_def(carry, xs_chunk):
                    out, chunk_losses = jax.lax.scan(
                        func_to_scan,
                        carry,
                        xs=xs_chunk,
                        length=length,
                        unroll=1,
                    )
                    return out, chunk_losses

                return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

            current_func = innermost_step
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            step_indices = jnp.arange(1, iterations + 1).reshape(chunk_sizes)
            _, all_chunk_losses = jax.lax.scan(
                current_func,
                init_state,
                xs=step_indices,
                length=chunk_sizes[0],
                unroll=1,
            )
            return reduce_sampled_losses(all_chunk_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_A_h = jax.value_and_grad(forward_trajectory)(A_h_value)
        return loss, grad_A_h


def nn_epsilon_step(state, simulation, nn_params, model_batched_unscaled, stats, epsilon):
    next_state = state.copy()

    def model_batched(params_arg, features, coordinates, dzt):
        return epsilon * model_batched_unscaled(
            params_arg,
            features,
            coordinates,
            dzt,
        )

    # Traced parameters must not escape through persistent simulation attributes.
    nn_fields = ("_nn_correction_initialized", "_nn_params", "_nn_model_batched", "_nn_stats")
    previous_nn = tuple(getattr(simulation, name, None) for name in nn_fields)
    try:
        simulation._nn_correction_initialized = True
        simulation._nn_params = nn_params
        simulation._nn_model_batched = model_batched
        simulation._nn_stats = stats

        with next_state.settings.unlock():
            next_state.settings.baseline_model = False
            next_state.settings.use_forcings = getattr(state.settings, "use_forcings", False)

        zero_tendency_sources(next_state)

        simulation.step(next_state)
    finally:
        for name, value in zip(nn_fields, previous_nn):
            setattr(simulation, name, value)
    return next_state


class NNEpsilonJVPScan:
    def __init__(
        self,
        simulation,
        nn_params,
        model_batched_unscaled,
        stats,
        loss_function=agg_sum,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.nn_params = nn_params
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction

    def step_function(self, state, epsilon):
        return nn_epsilon_step(
            state,
            self.simulation,
            self.nn_params,
            self.model_batched_unscaled,
            self.stats,
            epsilon,
        )

    def g(self, state, epsilon_value, iterations=1, sample_steps=1):
        def forward_trajectory(epsilon):
            init_state = state.copy()

            def scan_step(carry_state, step_idx):
                next_state = self.step_function(carry_state, epsilon)
                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda s: self.loss_function(s),
                    lambda s: jnp.asarray(0.0),
                    next_state,
                )
                return next_state, step_loss

            step_indices = jnp.arange(1, iterations + 1)
            _, all_step_losses = jax.lax.scan(
                scan_step,
                init_state,
                xs=step_indices,
                length=iterations,
            )
            return reduce_sampled_losses(all_step_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_epsilon = jax.jvp(
            forward_trajectory,
            (epsilon_value,),
            (jnp.ones_like(epsilon_value),),
        )
        return loss, grad_epsilon


class NNEpsilonChunkedVJP:
    def __init__(
        self,
        simulation,
        nn_params,
        model_batched_unscaled,
        stats,
        loss_function=agg_sum,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.nn_params = nn_params
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction

    def step_function(self, state, epsilon):
        return nn_epsilon_step(
            state,
            self.simulation,
            self.nn_params,
            self.model_batched_unscaled,
            self.stats,
            epsilon,
        )

    def g(self, full_state, epsilon_value, prognostic_keys, chunk_sizes, sample_steps=1):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size

        def forward_trajectory(epsilon):
            init_carry = {
                key: getattr(full_state.variables, key)
                for key in prognostic_keys
            }

            @jax.checkpoint
            def innermost_step(inner_tuple, step_idx):
                dynamic_vars, step_epsilon = inner_tuple
                curr_state = inject_dynamic_vars(full_state, dict(dynamic_vars))
                next_state = self.step_function(curr_state, step_epsilon)
                next_carry = {
                    key: getattr(next_state.variables, key)
                    for key in prognostic_keys
                }

                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda state: self.loss_function(state),
                    lambda state: jnp.asarray(0.0),
                    next_state,
                )
                return (next_carry, step_epsilon), step_loss

            def make_chunk_func(func_to_scan, length):
                def chunk_def(carry, xs_chunk):
                    out, chunk_losses = jax.lax.scan(
                        func_to_scan,
                        carry,
                        xs=xs_chunk,
                        length=length,
                        unroll=1,
                    )
                    return out, chunk_losses

                return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

            current_func = innermost_step
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            step_indices = jnp.arange(1, iterations + 1).reshape(chunk_sizes)
            _, all_chunk_losses = jax.lax.scan(
                current_func,
                (init_carry, epsilon),
                xs=step_indices,
                length=chunk_sizes[0],
                unroll=1,
            )
            return reduce_sampled_losses(all_chunk_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_epsilon = jax.value_and_grad(forward_trajectory)(epsilon_value)
        return loss, grad_epsilon


class NNEpsilonFullStateChunkedVJP:
    def __init__(
        self,
        simulation,
        nn_params,
        model_batched_unscaled,
        stats,
        loss_function=agg_sum,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.nn_params = nn_params
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction

    def step_function(self, state, epsilon):
        return nn_epsilon_step(
            state,
            self.simulation,
            self.nn_params,
            self.model_batched_unscaled,
            self.stats,
            epsilon,
        )

    def g(self, full_state, epsilon_value, prognostic_keys, chunk_sizes, sample_steps=1):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size

        def forward_trajectory(epsilon):
            init_state = full_state.copy()

            @jax.checkpoint
            def innermost_step(carry_state, step_idx):
                next_state = self.step_function(carry_state, epsilon)
                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda state: self.loss_function(state),
                    lambda state: jnp.asarray(0.0),
                    next_state,
                )
                return next_state, step_loss

            def make_chunk_func(func_to_scan, length):
                def chunk_def(carry, xs_chunk):
                    out, chunk_losses = jax.lax.scan(
                        func_to_scan,
                        carry,
                        xs=xs_chunk,
                        length=length,
                        unroll=1,
                    )
                    return out, chunk_losses

                return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

            current_func = innermost_step
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            step_indices = jnp.arange(1, iterations + 1).reshape(chunk_sizes)
            _, all_chunk_losses = jax.lax.scan(
                current_func,
                init_state,
                xs=step_indices,
                length=chunk_sizes[0],
                unroll=1,
            )
            return reduce_sampled_losses(all_chunk_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_epsilon = jax.value_and_grad(forward_trajectory)(epsilon_value)
        return loss, grad_epsilon


class NNParamJVPScan:
    def __init__(
        self,
        simulation,
        nn_params,
        model_batched_unscaled,
        stats,
        epsilon,
        leaf_index,
        flat_index,
        loss_function=agg_sum,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.nn_params = nn_params
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.epsilon = epsilon
        self.leaf_index = leaf_index
        self.flat_index = flat_index
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction

    def step_function(self, state, nn_params):
        return nn_epsilon_step(
            state,
            self.simulation,
            nn_params,
            self.model_batched_unscaled,
            self.stats,
            self.epsilon,
        )

    def g(self, state, param_value, iterations=1, sample_steps=1):
        def forward_trajectory(scalar_param):
            step_params = set_param_scalar(
                self.nn_params,
                self.leaf_index,
                self.flat_index,
                scalar_param,
            )
            init_state = state.copy()

            def scan_step(carry_state, step_idx):
                next_state = self.step_function(carry_state, step_params)
                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda s: self.loss_function(s),
                    lambda s: jnp.asarray(0.0),
                    next_state,
                )
                return next_state, step_loss

            step_indices = jnp.arange(1, iterations + 1)
            _, all_step_losses = jax.lax.scan(
                scan_step,
                init_state,
                xs=step_indices,
                length=iterations,
            )
            return reduce_sampled_losses(all_step_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_param = jax.jvp(
            forward_trajectory,
            (param_value,),
            (jnp.ones_like(param_value),),
        )
        return loss, grad_param


class NNParamChunkedVJP:
    def __init__(
        self,
        simulation,
        nn_params,
        model_batched_unscaled,
        stats,
        epsilon,
        leaf_index,
        flat_index,
        loss_function=agg_sum,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.nn_params = nn_params
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.epsilon = epsilon
        self.leaf_index = leaf_index
        self.flat_index = flat_index
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction

    def step_function(self, state, nn_params):
        return nn_epsilon_step(
            state,
            self.simulation,
            nn_params,
            self.model_batched_unscaled,
            self.stats,
            self.epsilon,
        )

    def g(self, full_state, param_value, prognostic_keys, chunk_sizes, sample_steps=1):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size

        def forward_trajectory(scalar_param):
            step_params = set_param_scalar(
                self.nn_params,
                self.leaf_index,
                self.flat_index,
                scalar_param,
            )
            init_carry = {
                key: getattr(full_state.variables, key)
                for key in prognostic_keys
            }

            @jax.checkpoint
            def innermost_step(dynamic_vars, step_idx):
                curr_state = inject_dynamic_vars(full_state, dict(dynamic_vars))
                next_state = self.step_function(curr_state, step_params)
                next_carry = {
                    key: getattr(next_state.variables, key)
                    for key in prognostic_keys
                }

                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda state: self.loss_function(state),
                    lambda state: jnp.asarray(0.0),
                    next_state,
                )
                return next_carry, step_loss

            def make_chunk_func(func_to_scan, length):
                def chunk_def(carry, xs_chunk):
                    out, chunk_losses = jax.lax.scan(
                        func_to_scan,
                        carry,
                        xs=xs_chunk,
                        length=length,
                        unroll=1,
                    )
                    return out, chunk_losses

                return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

            current_func = innermost_step
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            step_indices = jnp.arange(1, iterations + 1).reshape(chunk_sizes)
            _, all_chunk_losses = jax.lax.scan(
                current_func,
                init_carry,
                xs=step_indices,
                length=chunk_sizes[0],
                unroll=1,
            )
            return reduce_sampled_losses(all_chunk_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_param = jax.value_and_grad(forward_trajectory)(param_value)
        return loss, grad_param


class NNParamFullStateChunkedVJP:
    def __init__(
        self,
        simulation,
        nn_params,
        model_batched_unscaled,
        stats,
        epsilon,
        leaf_index,
        flat_index,
        loss_function=agg_sum,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.nn_params = nn_params
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.epsilon = epsilon
        self.leaf_index = leaf_index
        self.flat_index = flat_index
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction

    def step_function(self, state, nn_params):
        return nn_epsilon_step(
            state,
            self.simulation,
            nn_params,
            self.model_batched_unscaled,
            self.stats,
            self.epsilon,
        )

    def g(self, full_state, param_value, prognostic_keys, chunk_sizes, sample_steps=1):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size

        def forward_trajectory(scalar_param):
            step_params = set_param_scalar(
                self.nn_params,
                self.leaf_index,
                self.flat_index,
                scalar_param,
            )
            init_state = full_state.copy()

            @jax.checkpoint
            def innermost_step(carry_state, step_idx):
                next_state = self.step_function(carry_state, step_params)
                is_sample_step = (step_idx % sample_steps) == 0
                maybe_write_sample_u(
                    getattr(self, "sample_callback", None),
                    is_sample_step,
                    step_idx,
                    next_state,
                )
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda state: self.loss_function(state),
                    lambda state: jnp.asarray(0.0),
                    next_state,
                )
                return next_state, step_loss

            def make_chunk_func(func_to_scan, length):
                def chunk_def(carry, xs_chunk):
                    out, chunk_losses = jax.lax.scan(
                        func_to_scan,
                        carry,
                        xs=xs_chunk,
                        length=length,
                        unroll=1,
                    )
                    return out, chunk_losses

                return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

            current_func = innermost_step
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            step_indices = jnp.arange(1, iterations + 1).reshape(chunk_sizes)
            _, all_chunk_losses = jax.lax.scan(
                current_func,
                init_state,
                xs=step_indices,
                length=chunk_sizes[0],
                unroll=1,
            )
            return reduce_sampled_losses(all_chunk_losses, iterations, sample_steps, self.loss_reduction)

        loss, grad_param = jax.value_and_grad(forward_trajectory)(param_value)
        return loss, grad_param


class NNParamsJVPDirections:
    def __init__(
        self,
        simulation,
        model_batched_unscaled,
        stats,
        epsilon,
        loss_function,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.epsilon = epsilon
        self.loss_function = loss_function
        self.loss_reduction = loss_reduction

    def step_function(self, state, nn_params):
        return nn_epsilon_step(
            state,
            self.simulation,
            nn_params,
            self.model_batched_unscaled,
            self.stats,
            self.epsilon,
        )

    def trajectory_loss(self, state, nn_params, iterations=1, sample_steps=1):
        init_state = state.copy()

        def scan_step(carry_state, step_idx):
            next_state = self.step_function(carry_state, nn_params)
            is_sample_step = (step_idx % sample_steps) == 0
            maybe_write_sample_u(
                getattr(self, "sample_callback", None),
                is_sample_step,
                step_idx,
                next_state,
            )
            step_loss = jax.lax.cond(
                is_sample_step,
                lambda s: self.loss_function(s),
                lambda s: jnp.asarray(0.0),
                next_state,
            )
            return next_state, step_loss

        step_indices = jnp.arange(1, iterations + 1)
        _, all_step_losses = jax.lax.scan(
            scan_step,
            init_state,
            xs=step_indices,
            length=iterations,
        )
        return reduce_sampled_losses(all_step_losses, iterations, sample_steps, self.loss_reduction)

    def g_direction(self, state, nn_params, tangent, iterations=1, sample_steps=1):
        def forward_trajectory(params_arg):
            return self.trajectory_loss(
                state,
                params_arg,
                iterations=iterations,
                sample_steps=sample_steps,
            )

        return jax.jvp(forward_trajectory, (nn_params,), (tangent,))


class NNParamsChunkedVJP:
    def __init__(
        self,
        simulation,
        model_batched_unscaled,
        stats,
        epsilon,
        loss_function,
        freeze_vars=True,
        loss_reduction="sum",
    ):
        self.simulation = simulation
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.epsilon = epsilon
        self.loss_function = loss_function
        self.freeze_vars = freeze_vars
        self.loss_reduction = loss_reduction

    def step_function(self, state, nn_params):
        return nn_epsilon_step(
            state,
            self.simulation,
            nn_params,
            self.model_batched_unscaled,
            self.stats,
            self.epsilon,
        )

    def g(self, full_state, nn_params, prognostic_keys, chunk_sizes, sample_steps=1):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size

        def make_chunk_func(func_to_scan, length):
            def chunk_def(carry, xs_chunk):
                out, chunk_losses = jax.lax.scan(
                    func_to_scan,
                    carry,
                    xs=xs_chunk,
                    length=length,
                    unroll=1,
                )
                return out, chunk_losses

            return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

        def forward_trajectory(params_arg):
            if self.freeze_vars:
                init_carry = {
                    key: getattr(full_state.variables, key)
                    for key in prognostic_keys
                }

                @jax.checkpoint
                def innermost_step(dynamic_vars, step_idx):
                    curr_state = inject_dynamic_vars(full_state, dict(dynamic_vars))
                    next_state = self.step_function(curr_state, params_arg)
                    next_carry = {
                        key: getattr(next_state.variables, key)
                        for key in prognostic_keys
                    }
                    is_sample_step = (step_idx % sample_steps) == 0
                    maybe_write_sample_u(
                        getattr(self, "sample_callback", None),
                        is_sample_step,
                        step_idx,
                        next_state,
                    )
                    step_loss = jax.lax.cond(
                        is_sample_step,
                        lambda state: self.loss_function(state),
                        lambda state: jnp.asarray(0.0),
                        next_state,
                    )
                    return next_carry, step_loss

                initial_scan_carry = init_carry
            else:
                init_state = full_state.copy()

                @jax.checkpoint
                def innermost_step(carry_state, step_idx):
                    next_state = self.step_function(carry_state, params_arg)
                    is_sample_step = (step_idx % sample_steps) == 0
                    maybe_write_sample_u(
                        getattr(self, "sample_callback", None),
                        is_sample_step,
                        step_idx,
                        next_state,
                    )
                    step_loss = jax.lax.cond(
                        is_sample_step,
                        lambda state: self.loss_function(state),
                        lambda state: jnp.asarray(0.0),
                        next_state,
                    )
                    return next_state, step_loss

                initial_scan_carry = init_state

            current_func = innermost_step
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            step_indices = jnp.arange(1, iterations + 1).reshape(chunk_sizes)
            _, all_chunk_losses = jax.lax.scan(
                current_func,
                initial_scan_carry,
                xs=step_indices,
                length=chunk_sizes[0],
                unroll=1,
            )
            return reduce_sampled_losses(all_chunk_losses, iterations, sample_steps, self.loss_reduction)

        return jax.value_and_grad(forward_trajectory)(nn_params)


def make_benchmark_loss_function(args, simulation, target_file=None):
    if args.loss_mode == "u2":
        return agg_sum

    if args.loss_mode == "glorys_target":
        if target_file is None:
            raise ValueError("--loss-mode glorys_target requires --glorys-target-file")
        target = load_glorys_snapshot(
            simulation,
            target_file,
            tuple(args.keys_loss),
            time_index=args.glorys_time_index,
        )

        def loss_function(state):
            return compute_loss(
                state,
                target,
                keys=tuple(args.keys_loss),
                tau=state.variables.tau,
            )

        return loss_function

    raise ValueError(f"Unsupported loss mode: {args.loss_mode}")


def cosine_similarity(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return np.nan
    return float(np.dot(left, right) / denom)


def relative_vector_error(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(right)
    if denom == 0:
        return np.nan
    return float(np.linalg.norm(left - right) / denom)


def run_comparison(
    simulation,
    prognostic_keys,
    steps_list,
    sample_steps,
    chunk_base,
    control_mode,
    control_value,
    method_pair="jvp-vjp",
    freeze_vars=True,
    nn_params=None,
    model_batched_unscaled=None,
    nn_stats=None,
    nn_param_leaf_index=0,
    nn_param_flat_index=0,
    epsilon=None,
    loss_function=agg_sum,
    loss_reduction="sum",
    save_sample_trajectories=False,
    trajectory_folder=None,
    trajectory_prefix=None,
    results=None,
    pkl_path=None,
):
    if results is None:
        results = {
            "steps": [],
            "jvp_grad": [],
            "jvp_loss": [],
            "jvp_time": [],
            "vjp_grad": [],
            "vjp_loss": [],
            "vjp_time": [],
        }

    already_done = set(results["steps"])
    for n_steps in steps_list:
        chunk_sizes = make_chunk_sizes(n_steps, base=chunk_base)
        if n_steps in already_done:
            print(f"\n=== n_steps = {n_steps} : already in results, skipping (resume) ===")
            if save_sample_trajectories:
                trajectory_path = os.path.join(
                    trajectory_folder,
                    f"{trajectory_prefix}_steps_{n_steps}.h5",
                )
                if os.path.exists(trajectory_path):
                    print(f"  [Trajectory] already exists -> {trajectory_path}")
                else:
                    print(
                        "  [Trajectory] skipped step has no live forward pass to stream; "
                        "use a fresh output prefix/folder or remove this step from the pickle to regenerate it."
                    )
            continue

        print(f"\n=== n_steps = {n_steps} | chunk_sizes = {chunk_sizes} ===")
        if hasattr(simulation, "prepare_ERA5_forcing_window"):
            simulation.prepare_ERA5_forcing_window(simulation.state, n_steps)
        print_memory_stats("Baseline")

        jax.clear_caches()
        gc.collect()
        trajectory_writer = None
        if save_sample_trajectories:
            trajectory_path = os.path.join(
                trajectory_folder,
                f"{trajectory_prefix}_steps_{n_steps}.h5",
            )
            if os.path.exists(trajectory_path):
                print(f"  [Trajectory] already exists -> {trajectory_path}")
            else:
                print("  [Trajectory] streaming sampled u during measured AD runs...")
                trajectory_writer = SampleTrajectoryWriter(
                    trajectory_path,
                    n_steps,
                    sample_steps,
                    simulation.state.variables.u,
                    loss_reduction,
                ).__enter__()
                trajectory_writer.h5file.attrs["method_pair"] = method_pair
                trajectory_writer.h5file.attrs["control_mode"] = control_mode

        ad_jvp = None
        ad_vjp = None
        try:
            if control_mode == "A_h":
                ad_jvp = A_hJVPScan(simulation.step, loss_function, loss_reduction)
                jvp_kwargs = {"A_h_value": control_value}
            elif control_mode == "nn_epsilon":
                ad_jvp = NNEpsilonJVPScan(
                    simulation,
                    nn_params,
                    model_batched_unscaled,
                    nn_stats,
                    loss_function,
                    loss_reduction,
                )
                jvp_kwargs = {"epsilon_value": control_value}
            elif control_mode == "nn_param":
                ad_jvp = NNParamJVPScan(
                    simulation,
                    nn_params,
                    model_batched_unscaled,
                    nn_stats,
                    epsilon,
                    nn_param_leaf_index,
                    nn_param_flat_index,
                    loss_function,
                    loss_reduction,
                )
                jvp_kwargs = {"param_value": control_value}
            else:
                raise ValueError(f"Unsupported control_mode: {control_mode}")

            print("  [JVP] Compiling...")
            warmup_jvp = ad_jvp.g(
                simulation.state,
                iterations=n_steps,
                sample_steps=sample_steps,
                **jvp_kwargs,
            )
            jax.block_until_ready(warmup_jvp)
            free_buffers(warmup_jvp)

            print("  [JVP] Running...")
            if trajectory_writer is not None:
                ad_jvp.sample_callback = trajectory_writer.callback("jvp")
            try:
                start_time = time.time()
                n_diff_jvp = ad_jvp.g(
                    simulation.state,
                    iterations=n_steps,
                    sample_steps=sample_steps,
                    **jvp_kwargs,
                )
                jax.block_until_ready(n_diff_jvp)
            finally:
                ad_jvp.sample_callback = None
            elapsed_jvp = time.time() - start_time
            loss_jvp = float(n_diff_jvp[0])
            grad_jvp = float(jnp.mean(n_diff_jvp[1]))
            print(f"  -> jvp_diff_scanned : grad={grad_jvp:.6e} loss={loss_jvp:.6e} time={elapsed_jvp:.4f}s")

            if method_pair == "jvp-jvp":
                print("  [JVP repeat] Running...")
                if trajectory_writer is not None:
                    ad_jvp.sample_callback = trajectory_writer.callback("vjp")
                try:
                    start_time = time.time()
                    n_diff_repeat = ad_jvp.g(
                        simulation.state,
                        iterations=n_steps,
                        sample_steps=sample_steps,
                        **jvp_kwargs,
                    )
                    jax.block_until_ready(n_diff_repeat)
                finally:
                    ad_jvp.sample_callback = None
                elapsed_vjp = time.time() - start_time
                loss_vjp = float(n_diff_repeat[0])
                grad_vjp = float(jnp.mean(n_diff_repeat[1]))
                print(
                    "  -> jvp_diff_scanned_repeat : "
                    f"grad={grad_vjp:.6e} loss={loss_vjp:.6e} time={elapsed_vjp:.4f}s"
                )

                free_buffers(n_diff_jvp)
                free_buffers(n_diff_repeat)
                del warmup_jvp, n_diff_jvp, n_diff_repeat
                jax.clear_caches()
                gc.collect()
            elif method_pair == "jvp-vjp":
                free_buffers(n_diff_jvp)
                del warmup_jvp, n_diff_jvp
                jax.clear_caches()
                gc.collect()
                print_memory_stats("Post-JVP")

                if control_mode == "A_h":
                    ad_vjp = (
                        A_hChunkedVJP(simulation.step, loss_function, loss_reduction)
                        if freeze_vars
                        else A_hFullStateChunkedVJP(simulation.step, loss_function, loss_reduction)
                    )
                    vjp_kwargs = {"A_h_value": control_value}
                elif control_mode == "nn_epsilon":
                    vjp_class = NNEpsilonChunkedVJP if freeze_vars else NNEpsilonFullStateChunkedVJP
                    ad_vjp = vjp_class(
                        simulation,
                        nn_params,
                        model_batched_unscaled,
                        nn_stats,
                        loss_function,
                        loss_reduction,
                    )
                    vjp_kwargs = {"epsilon_value": control_value}
                elif control_mode == "nn_param":
                    vjp_class = NNParamChunkedVJP if freeze_vars else NNParamFullStateChunkedVJP
                    ad_vjp = vjp_class(
                        simulation,
                        nn_params,
                        model_batched_unscaled,
                        nn_stats,
                        epsilon,
                        nn_param_leaf_index,
                        nn_param_flat_index,
                        loss_function,
                        loss_reduction,
                    )
                    vjp_kwargs = {"param_value": control_value}
                else:
                    raise ValueError(f"Unsupported control_mode: {control_mode}")

                print("  [VJP] Compiling...")
                warmup_vjp = ad_vjp.g(
                    simulation.state,
                    prognostic_keys=prognostic_keys,
                    chunk_sizes=chunk_sizes,
                    sample_steps=sample_steps,
                    **vjp_kwargs,
                )
                jax.block_until_ready(warmup_vjp)
                free_buffers(warmup_vjp)

                print("  [VJP] Running...")
                if trajectory_writer is not None:
                    ad_vjp.sample_callback = trajectory_writer.callback("vjp")
                try:
                    start_time = time.time()
                    n_diff_vjp = ad_vjp.g(
                        simulation.state,
                        prognostic_keys=prognostic_keys,
                        chunk_sizes=chunk_sizes,
                        sample_steps=sample_steps,
                        **vjp_kwargs,
                    )
                    jax.block_until_ready(n_diff_vjp)
                finally:
                    ad_vjp.sample_callback = None
                elapsed_vjp = time.time() - start_time
                loss_vjp = float(n_diff_vjp[0])
                grad_vjp = float(jnp.mean(n_diff_vjp[1]))
                print(f"  -> vjp_grad_chunked_Nlvls : grad={grad_vjp:.6e} loss={loss_vjp:.6e} time={elapsed_vjp:.4f}s")

                free_buffers(n_diff_vjp)
                del warmup_vjp, n_diff_vjp
                clear_after_ad("Post-AD")
            else:
                raise ValueError(f"Unsupported method_pair: {method_pair}")

            if trajectory_writer is not None:
                trajectory_writer.set_ad_losses(loss_jvp, loss_vjp)
                trajectory_writer.__exit__(None, None, None)
                trajectory_writer = None

            del ad_jvp
            ad_jvp = None
            if ad_vjp is not None:
                del ad_vjp
                ad_vjp = None
            print_memory_stats("Post-VJP" if method_pair == "jvp-vjp" else "Post-JVP-repeat")
        except Exception as exc:
            if ad_jvp is not None:
                ad_jvp.sample_callback = None
            if ad_vjp is not None:
                ad_vjp.sample_callback = None
            if trajectory_writer is not None:
                trajectory_writer.__exit__(type(exc), exc, exc.__traceback__)
            raise

        results["steps"].append(n_steps)
        results["jvp_grad"].append(grad_jvp)
        results["jvp_loss"].append(loss_jvp)
        results["jvp_time"].append(elapsed_jvp)
        results["vjp_grad"].append(grad_vjp)
        results["vjp_loss"].append(loss_vjp)
        results["vjp_time"].append(elapsed_vjp)

        if pkl_path is not None:
            save_results(results, pkl_path)

    return results


def run_param_vector_comparison(
    simulation,
    prognostic_keys,
    steps_list,
    sample_steps,
    chunk_base,
    freeze_vars,
    nn_params,
    model_batched_unscaled,
    nn_stats,
    epsilon,
    selected_params,
    loss_function,
    loss_reduction="sum",
    save_sample_trajectories=False,
    trajectory_folder=None,
    trajectory_prefix=None,
    results=None,
    pkl_path=None,
):
    if results is None:
        results = {
            "steps": [],
            "jvp_grad_vector": [],
            "vjp_grad_vector": [],
            "jvp_loss": [],
            "vjp_loss": [],
            "jvp_time": [],
            "vjp_time": [],
            "cosine_similarity": [],
            "relative_error": [],
            "jvp_norm": [],
            "vjp_norm": [],
            "loss_abs_diff": [],
        }

    tangents = [
        make_param_tangent(nn_params, item["leaf_index"], item["flat_index"])
        for item in selected_params
    ]

    already_done = set(results["steps"])
    for n_steps in steps_list:
        chunk_sizes = make_chunk_sizes(n_steps, base=chunk_base)
        if n_steps in already_done:
            print(f"\n=== n_steps = {n_steps} : already in results, skipping (resume) ===")
            if save_sample_trajectories:
                trajectory_path = os.path.join(
                    trajectory_folder,
                    f"{trajectory_prefix}_steps_{n_steps}.h5",
                )
                if os.path.exists(trajectory_path):
                    print(f"  [Trajectory] already exists -> {trajectory_path}")
                else:
                    print(
                        "  [Trajectory] skipped step has no live forward pass to stream; "
                        "use a fresh output prefix/folder or remove this step from the pickle to regenerate it."
                    )
            continue

        print(f"\n=== n_steps = {n_steps} | chunk_sizes = {chunk_sizes} | {len(selected_params)} JVP directions ===")
        if hasattr(simulation, "prepare_ERA5_forcing_window"):
            simulation.prepare_ERA5_forcing_window(simulation.state, n_steps)
        print_memory_stats("Baseline")

        jax.clear_caches()
        gc.collect()
        trajectory_writer = None
        if save_sample_trajectories:
            trajectory_path = os.path.join(
                trajectory_folder,
                f"{trajectory_prefix}_steps_{n_steps}.h5",
            )
            if os.path.exists(trajectory_path):
                print(f"  [Trajectory] already exists -> {trajectory_path}")
            else:
                print("  [Trajectory] streaming sampled u during measured JVP/VJP runs...")
                trajectory_writer = SampleTrajectoryWriter(
                    trajectory_path,
                    n_steps,
                    sample_steps,
                    simulation.state.variables.u,
                    loss_reduction,
                ).__enter__()
                trajectory_writer.h5file.attrs["method_pair"] = "jvp-vjp"
                trajectory_writer.h5file.attrs["control_mode"] = "nn_param_vector"
                trajectory_writer.h5file.attrs["jvp_streamed_direction_index"] = 0

        ad_jvp = None
        ad_vjp = None
        try:
            ad_jvp = NNParamsJVPDirections(
                simulation,
                model_batched_unscaled,
                nn_stats,
                epsilon,
                loss_function,
                loss_reduction,
            )

            print("  [JVP vector] Compiling first direction...")
            warmup_jvp = ad_jvp.g_direction(
                simulation.state,
                nn_params,
                tangents[0],
                iterations=n_steps,
                sample_steps=sample_steps,
            )
            jax.block_until_ready(warmup_jvp)
            free_buffers(warmup_jvp)

            print("  [JVP vector] Running one-hot directions...")
            start_time = time.time()
            jvp_losses = []
            jvp_grads = []
            for direction_idx, tangent in enumerate(tangents):
                if trajectory_writer is not None and direction_idx == 0:
                    ad_jvp.sample_callback = trajectory_writer.callback("jvp")
                try:
                    out = ad_jvp.g_direction(
                        simulation.state,
                        nn_params,
                        tangent,
                        iterations=n_steps,
                        sample_steps=sample_steps,
                    )
                    jax.block_until_ready(out)
                finally:
                    ad_jvp.sample_callback = None
                jvp_losses.append(float(out[0]))
                jvp_grads.append(float(out[1]))
                free_buffers(out)
                print(
                    f"    dir {direction_idx + 1:02d}/{len(tangents)} "
                    f"grad={jvp_grads[-1]:.6e} loss={jvp_losses[-1]:.6e}"
                )
            elapsed_jvp = time.time() - start_time
            jvp_vector = np.asarray(jvp_grads, dtype=np.float64)
            loss_jvp = float(jvp_losses[0])
            loss_jvp_spread = float(np.max(np.abs(np.asarray(jvp_losses) - loss_jvp)))
            print(
                f"  -> JVP vector done: loss={loss_jvp:.6e} "
                f"max_loss_spread={loss_jvp_spread:.3e} time={elapsed_jvp:.4f}s"
            )

            del ad_jvp
            ad_jvp = None
            jax.clear_caches()
            gc.collect()
            print_memory_stats("Post-JVP-vector")

            ad_vjp = NNParamsChunkedVJP(
                simulation,
                model_batched_unscaled,
                nn_stats,
                epsilon,
                loss_function,
                freeze_vars=freeze_vars,
                loss_reduction=loss_reduction,
            )

            print("  [VJP full params] Compiling...")
            warmup_vjp = ad_vjp.g(
                simulation.state,
                nn_params,
                prognostic_keys=prognostic_keys,
                chunk_sizes=chunk_sizes,
                sample_steps=sample_steps,
            )
            jax.block_until_ready(warmup_vjp)
            free_buffers(warmup_vjp)

            print("  [VJP full params] Running...")
            if trajectory_writer is not None:
                ad_vjp.sample_callback = trajectory_writer.callback("vjp")
            try:
                start_time = time.time()
                vjp_loss_value, vjp_grad_tree = ad_vjp.g(
                    simulation.state,
                    nn_params,
                    prognostic_keys=prognostic_keys,
                    chunk_sizes=chunk_sizes,
                    sample_steps=sample_steps,
                )
                jax.block_until_ready((vjp_loss_value, vjp_grad_tree))
            finally:
                ad_vjp.sample_callback = None
            elapsed_vjp = time.time() - start_time
            loss_vjp = float(vjp_loss_value)
            vjp_vector = np.asarray(selected_param_vector(vjp_grad_tree, selected_params), dtype=np.float64)

            cos_sim = cosine_similarity(jvp_vector, vjp_vector)
            rel_err = relative_vector_error(jvp_vector, vjp_vector)
            jvp_norm = float(np.linalg.norm(jvp_vector))
            vjp_norm = float(np.linalg.norm(vjp_vector))
            loss_abs_diff = float(abs(loss_vjp - loss_jvp))

            print(
                "  -> VJP vector comparison: "
                f"loss={loss_vjp:.6e} cos={cos_sim:.6f} "
                f"rel_err={rel_err:.6e} "
                f"||jvp||={jvp_norm:.6e} ||vjp||={vjp_norm:.6e} "
                f"time={elapsed_vjp:.4f}s"
            )

            free_buffers((vjp_loss_value, vjp_grad_tree))
            del ad_vjp, vjp_loss_value, vjp_grad_tree
            ad_vjp = None
            clear_after_ad("Post-AD")

            if trajectory_writer is not None:
                trajectory_writer.set_ad_losses(loss_jvp, loss_vjp)
                trajectory_writer.__exit__(None, None, None)
                trajectory_writer = None
        except Exception as exc:
            if ad_jvp is not None:
                ad_jvp.sample_callback = None
            if ad_vjp is not None:
                ad_vjp.sample_callback = None
            if trajectory_writer is not None:
                trajectory_writer.__exit__(type(exc), exc, exc.__traceback__)
            raise

        print_memory_stats("Post-VJP-vector")

        results["steps"].append(n_steps)
        results["jvp_grad_vector"].append(jvp_vector.tolist())
        results["vjp_grad_vector"].append(vjp_vector.tolist())
        results["jvp_loss"].append(loss_jvp)
        results["vjp_loss"].append(loss_vjp)
        results["jvp_time"].append(elapsed_jvp)
        results["vjp_time"].append(elapsed_vjp)
        results["cosine_similarity"].append(cos_sim)
        results["relative_error"].append(rel_err)
        results["jvp_norm"].append(jvp_norm)
        results["vjp_norm"].append(vjp_norm)
        results["loss_abs_diff"].append(loss_abs_diff)

        if pkl_path is not None:
            save_results(results, pkl_path)

    return results


def run_param_vector_vjp_only(
    simulation,
    prognostic_keys,
    steps_list,
    sample_steps,
    chunk_base,
    freeze_vars,
    nn_params,
    model_batched_unscaled,
    nn_stats,
    epsilon,
    selected_params,
    loss_function,
    loss_reduction="sum",
    results=None,
    pkl_path=None,
):
    if results is None:
        results = {
            "steps": [],
            "vjp_grad_vector": [],
            "vjp_loss": [],
            "vjp_time": [],
            "vjp_grad_norm": [],
            "vjp_grad_min": [],
            "vjp_grad_max": [],
            "vjp_grad_max_abs": [],
            "vjp_peak_gpu_memory_mib": [],
            "vjp_before_gpu_memory_mib": [],
            "vjp_after_gpu_memory_mib": [],
            "vjp_jax_bytes_in_use": [],
            "vjp_jax_peak_bytes_in_use": [],
            "vjp_jax_live_arrays": [],
        }

    already_done = set(results["steps"])
    for n_steps in steps_list:
        chunk_sizes = make_chunk_sizes(n_steps, base=chunk_base)
        if n_steps in already_done:
            print(f"\n=== n_steps = {n_steps} : already in results, skipping (resume) ===")
            continue

        print(f"\n=== n_steps = {n_steps} | chunk_sizes = {chunk_sizes} | VJP only ===")
        if hasattr(simulation, "prepare_ERA5_forcing_window"):
            simulation.prepare_ERA5_forcing_window(simulation.state, n_steps)
        print_memory_stats("Baseline")

        jax.clear_caches()
        gc.collect()

        ad_vjp = NNParamsChunkedVJP(
            simulation,
            model_batched_unscaled,
            nn_stats,
            epsilon,
            loss_function,
            freeze_vars=freeze_vars,
            loss_reduction=loss_reduction,
        )

        print("  [VJP full params] Compiling...")
        warmup_vjp = ad_vjp.g(
            simulation.state,
            nn_params,
            prognostic_keys=prognostic_keys,
            chunk_sizes=chunk_sizes,
            sample_steps=sample_steps,
        )
        jax.block_until_ready(warmup_vjp)
        free_buffers(warmup_vjp)

        print("  [VJP full params] Running...")
        jax_memory_before = jax_memory_snapshot()
        with GPUMemoryMonitor() as memory_monitor:
            start_time = time.time()
            vjp_loss_value, vjp_grad_tree = ad_vjp.g(
                simulation.state,
                nn_params,
                prognostic_keys=prognostic_keys,
                chunk_sizes=chunk_sizes,
                sample_steps=sample_steps,
            )
            jax.block_until_ready((vjp_loss_value, vjp_grad_tree))
        elapsed_vjp = time.time() - start_time
        jax_memory_after = jax_memory_snapshot()

        loss_vjp = float(vjp_loss_value)
        vjp_vector = np.asarray(selected_param_vector(vjp_grad_tree, selected_params), dtype=np.float64)
        vjp_norm = float(np.linalg.norm(vjp_vector))
        vjp_min = float(np.nanmin(vjp_vector)) if vjp_vector.size else np.nan
        vjp_max = float(np.nanmax(vjp_vector)) if vjp_vector.size else np.nan
        vjp_max_abs = float(np.nanmax(np.abs(vjp_vector))) if vjp_vector.size else np.nan

        print(
            "  -> VJP only: "
            f"loss={loss_vjp:.6e} "
            f"||grad||={vjp_norm:.6e} "
            f"grad_min={vjp_min:.6e} grad_max={vjp_max:.6e} "
            f"grad_max_abs={vjp_max_abs:.6e} "
            f"time={elapsed_vjp:.4f}s "
            f"gpu_peak={format_memory_mib(memory_monitor.peak_mib)}"
        )
        print("  selected VJP gradient entries:")
        for item, grad_value in zip(selected_params, vjp_vector):
            print(
                f"    global={item['global_index']} path={item['path']} "
                f"leaf={item['leaf_index']} flat_index={item['flat_index']} "
                f"grad={grad_value:.12e}"
            )

        free_buffers((vjp_loss_value, vjp_grad_tree))
        del ad_vjp, vjp_loss_value, vjp_grad_tree
        clear_after_ad("Post-VJP-only")

        results["steps"].append(n_steps)
        results["vjp_grad_vector"].append(vjp_vector.tolist())
        results["vjp_loss"].append(loss_vjp)
        results["vjp_time"].append(elapsed_vjp)
        results["vjp_grad_norm"].append(vjp_norm)
        results["vjp_grad_min"].append(vjp_min)
        results["vjp_grad_max"].append(vjp_max)
        results["vjp_grad_max_abs"].append(vjp_max_abs)
        results["vjp_peak_gpu_memory_mib"].append(float(memory_monitor.peak_mib))
        results["vjp_before_gpu_memory_mib"].append(float(memory_monitor.before_mib))
        results["vjp_after_gpu_memory_mib"].append(float(memory_monitor.after_mib))
        results["vjp_jax_bytes_in_use"].append(float(jax_memory_after["bytes_in_use"]))
        results["vjp_jax_peak_bytes_in_use"].append(float(jax_memory_after["peak_bytes_in_use"]))
        results["vjp_jax_live_arrays"].append(int(jax_memory_after["live_arrays"]))
        results.setdefault("vjp_jax_bytes_in_use_before", []).append(float(jax_memory_before["bytes_in_use"]))
        results.setdefault("vjp_jax_peak_bytes_in_use_before", []).append(float(jax_memory_before["peak_bytes_in_use"]))

        if pkl_path is not None:
            save_results(results, pkl_path)

    return results


def run_nn_param_vjp_only(
    simulation,
    prognostic_keys,
    steps_list,
    sample_steps,
    chunk_base,
    freeze_vars,
    nn_params,
    model_batched_unscaled,
    nn_stats,
    epsilon,
    control_value,
    nn_param_leaf_index,
    nn_param_flat_index,
    nn_param_description,
    loss_function,
    loss_reduction="sum",
    results=None,
    pkl_path=None,
):
    if results is None:
        results = {
            "steps": [],
            "vjp_grad": [],
            "vjp_grad_vector": [],
            "vjp_loss": [],
            "vjp_time": [],
            "vjp_grad_norm": [],
            "vjp_grad_min": [],
            "vjp_grad_max": [],
            "vjp_grad_max_abs": [],
            "vjp_peak_gpu_memory_mib": [],
            "vjp_before_gpu_memory_mib": [],
            "vjp_after_gpu_memory_mib": [],
            "vjp_jax_bytes_in_use": [],
            "vjp_jax_peak_bytes_in_use": [],
            "vjp_jax_live_arrays": [],
        }

    already_done = set(results["steps"])
    for n_steps in steps_list:
        chunk_sizes = make_chunk_sizes(n_steps, base=chunk_base)
        if n_steps in already_done:
            print(f"\n=== n_steps = {n_steps} : already in results, skipping (resume) ===")
            continue

        print(f"\n=== n_steps = {n_steps} | chunk_sizes = {chunk_sizes} | scalar NN-param VJP only ===")
        if hasattr(simulation, "prepare_ERA5_forcing_window"):
            simulation.prepare_ERA5_forcing_window(simulation.state, n_steps)
        print_memory_stats("Baseline")

        jax.clear_caches()
        gc.collect()

        vjp_class = NNParamChunkedVJP if freeze_vars else NNParamFullStateChunkedVJP
        ad_vjp = vjp_class(
            simulation,
            nn_params,
            model_batched_unscaled,
            nn_stats,
            epsilon,
            nn_param_leaf_index,
            nn_param_flat_index,
            loss_function,
            loss_reduction,
        )

        print("  [VJP scalar param] Compiling...")
        warmup_vjp = ad_vjp.g(
            simulation.state,
            control_value,
            prognostic_keys=prognostic_keys,
            chunk_sizes=chunk_sizes,
            sample_steps=sample_steps,
        )
        jax.block_until_ready(warmup_vjp)
        free_buffers(warmup_vjp)

        print("  [VJP scalar param] Running...")
        jax_memory_before = jax_memory_snapshot()
        with GPUMemoryMonitor() as memory_monitor:
            start_time = time.time()
            vjp_loss_value, vjp_grad_value = ad_vjp.g(
                simulation.state,
                control_value,
                prognostic_keys=prognostic_keys,
                chunk_sizes=chunk_sizes,
                sample_steps=sample_steps,
            )
            jax.block_until_ready((vjp_loss_value, vjp_grad_value))
        elapsed_vjp = time.time() - start_time
        jax_memory_after = jax_memory_snapshot()

        loss_vjp = float(vjp_loss_value)
        grad_vjp = float(vjp_grad_value)
        grad_abs = float(abs(grad_vjp))

        print(
            "  -> VJP only: "
            f"loss={loss_vjp:.6e} "
            f"grad={grad_vjp:.12e} "
            f"time={elapsed_vjp:.4f}s "
            f"gpu_peak={format_memory_mib(memory_monitor.peak_mib)}"
        )
        if nn_param_description is not None:
            print(
                "  selected VJP gradient entry: "
                f"path={nn_param_description['path']} "
                f"leaf={nn_param_description['leaf_index']} "
                f"flat_index={nn_param_description['flat_index']} "
                f"grad={grad_vjp:.12e}"
            )

        free_buffers((vjp_loss_value, vjp_grad_value))
        del ad_vjp, vjp_loss_value, vjp_grad_value
        clear_after_ad("Post-VJP-only")

        results["steps"].append(n_steps)
        results["vjp_grad"].append(grad_vjp)
        results["vjp_grad_vector"].append([grad_vjp])
        results["vjp_loss"].append(loss_vjp)
        results["vjp_time"].append(elapsed_vjp)
        results["vjp_grad_norm"].append(grad_abs)
        results["vjp_grad_min"].append(grad_vjp)
        results["vjp_grad_max"].append(grad_vjp)
        results["vjp_grad_max_abs"].append(grad_abs)
        results["vjp_peak_gpu_memory_mib"].append(float(memory_monitor.peak_mib))
        results["vjp_before_gpu_memory_mib"].append(float(memory_monitor.before_mib))
        results["vjp_after_gpu_memory_mib"].append(float(memory_monitor.after_mib))
        results["vjp_jax_bytes_in_use"].append(float(jax_memory_after["bytes_in_use"]))
        results["vjp_jax_peak_bytes_in_use"].append(float(jax_memory_after["peak_bytes_in_use"]))
        results["vjp_jax_live_arrays"].append(int(jax_memory_after["live_arrays"]))
        results.setdefault("vjp_jax_bytes_in_use_before", []).append(float(jax_memory_before["bytes_in_use"]))
        results.setdefault("vjp_jax_peak_bytes_in_use_before", []).append(float(jax_memory_before["peak_bytes_in_use"]))

        if pkl_path is not None:
            save_results(results, pkl_path)

    return results


def ordered_result_arrays(results):
    order = np.argsort(np.array(results["steps"]))
    return {
        key: np.array(values)[order]
        for key, values in results.items()
        if isinstance(values, list) and len(values) == len(results["steps"])
    }


def set_power_of_three_ticks(ax, steps):
    ticks = np.array(steps, dtype=int)
    labels = []
    for step in ticks:
        power = round(np.log(step) / np.log(3))
        labels.append(rf"$3^{power}$" if 3**power == step else str(step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def set_symlog_for_signed_values(ax, values):
    values = np.asarray(values, dtype=np.float64)
    finite_abs = np.abs(values[np.isfinite(values)])
    finite_abs = finite_abs[finite_abs > 0]
    linthresh = float(np.min(finite_abs) / 10.0) if finite_abs.size else 1e-30
    ax.set_yscale("symlog", linthresh=linthresh)


def value_tag(name, value):
    safe_value = f"{value:.0e}" if value != 0 else "0"
    safe_value = safe_value.replace("+", "").replace("-", "m").replace(".", "p")
    return f"{name}_{safe_value}"


def result_suffix_for(args, control_value_float):
    parts = []
    if args.control_mode == "nn_epsilon":
        parts.append(value_tag("eps", control_value_float))
    elif args.control_mode == "nn_param":
        parts.append(f"param_l{args.nn_param_leaf_index}_i{args.nn_param_flat_index}")
    elif args.control_mode == "nn_param_vector":
        param_seed = args.nn_param_seed if args.nn_param_seed is not None else args.seed
        parts.append(f"param_vector_n{args.nn_param_count}_seed{param_seed}")
        if args.nn_param_path_filter:
            parts.append("filter_" + re.sub(r"[^A-Za-z0-9_.-]+", "-", args.nn_param_path_filter).strip("-"))
    if args.method_pair == "jvp-jvp":
        parts.append("jvp_jvp")
    elif args.method_pair == "vjp-only":
        parts.append("vjp_only")
    if not args.freeze_vars:
        parts.append("nofreeze")
    return "_" + "_".join(parts) if parts else ""


def result_pickle_path(args, gpu_tag, control_value_float):
    result_suffix = result_suffix_for(args, control_value_float)
    return os.path.join(
        args.output_folder,
        f"{args.output_prefix}{result_suffix}_results_{gpu_tag}.pkl",
    )


def result_plot_path(args, gpu_tag, control_value_float):
    result_suffix = result_suffix_for(args, control_value_float)
    return os.path.join(
        args.output_folder,
        f"{args.output_prefix}{result_suffix}_{gpu_tag}.png",
    )


def print_summary(results, grad_err_pct, loss_err_pct):
    arrays = ordered_result_arrays(results)
    method_pair = results.get("method_pair", "jvp-vjp")
    second_label = "jvp2" if method_pair == "jvp-jvp" else "vjp"
    print("\n=== Summary ===")
    for i, n_steps in enumerate(arrays["steps"]):
        print(
            f"steps={n_steps:5d} | "
            f"grad_jvp={arrays['jvp_grad'][i]:.4e} grad_{second_label}={arrays['vjp_grad'][i]:.4e} "
            f"grad_err%={grad_err_pct[i]:.4f} | "
            f"loss_jvp={arrays['jvp_loss'][i]:.4e} loss_{second_label}={arrays['vjp_loss'][i]:.4e} "
            f"loss_err%={loss_err_pct[i]:.4f} | "
            f"t_jvp={arrays['jvp_time'][i]:.2f}s t_{second_label}={arrays['vjp_time'][i]:.2f}s"
        )


def plot_results(results, gpu_name="unknown", save_path="ad_comparison.png"):
    arrays = ordered_result_arrays(results)
    steps = arrays["steps"]
    jvp_grad = arrays["jvp_grad"]
    vjp_grad = arrays["vjp_grad"]
    jvp_loss = arrays["jvp_loss"]
    vjp_loss = arrays["vjp_loss"]
    jvp_time = arrays["jvp_time"]
    vjp_time = arrays["vjp_time"]

    grad_err_pct = np.abs(vjp_grad - jvp_grad) / np.abs(jvp_grad) * 100.0
    loss_err_pct = np.abs(vjp_loss - jvp_loss) / np.abs(jvp_loss) * 100.0
    control_mode = results.get("control_mode", "A_h")
    method_pair = results.get("method_pair", "jvp-vjp")
    gradient_labels = {
        "A_h": "dLoss/dA_h",
        "nn_epsilon": "dLoss/dEpsilon",
        "nn_param": "dLoss/dNNParam",
    }
    gradient_label = gradient_labels.get(control_mode, "Gradient")
    first_method_label = "JVP (run 1)" if method_pair == "jvp-jvp" else "JVP (scanned)"
    second_method_label = "JVP (run 2)" if method_pair == "jvp-jvp" else "VJP (chunked, N-levels)"
    error_title = "JVP repeat error" if method_pair == "jvp-jvp" else "Gradient error between methods"
    loss_error_title = "JVP repeat loss error" if method_pair == "jvp-jvp" else "Loss error between methods"

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"AD method comparison — GPU: {gpu_name}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(steps, jvp_grad, marker="o", label=first_method_label)
    ax.plot(steps, vjp_grad, marker="s", label=second_method_label)
    ax.set_xscale("log", base=3)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel(gradient_label)
    ax.set_title(f"{gradient_label} vs number of steps")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[0, 1]
    ax.plot(steps, grad_err_pct, marker="o", color="firebrick")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Gradient error (% of JVP)")
    ax.set_title(error_title)
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[1, 0]
    ax.plot(steps, jvp_loss, marker="o", label=first_method_label)
    ax.plot(steps, vjp_loss, marker="s", label=second_method_label)
    ax.set_xscale("log", base=3)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs number of steps")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[1, 1]
    ax.plot(steps, loss_err_pct, marker="o", color="firebrick")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Loss error (% of JVP)")
    ax.set_title(loss_error_title)
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[2, 0]
    ax.plot(steps, jvp_time, marker="o", label=first_method_label)
    ax.plot(steps, vjp_time, marker="s", label=second_method_label)
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Compute time (s)")
    ax.set_title("Compute time vs number of steps")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    axes[2, 1].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved figure to {save_path}")
    return grad_err_pct, loss_err_pct


def print_param_vector_summary(results):
    arrays = ordered_result_arrays(results)
    print("\n=== NN Parameter Vector Summary ===")
    for i, n_steps in enumerate(arrays["steps"]):
        print(
            f"steps={n_steps:5d} | "
            f"cos={arrays['cosine_similarity'][i]:.6f} "
            f"rel_err={arrays['relative_error'][i]:.4e} | "
            f"||jvp||={arrays['jvp_norm'][i]:.4e} "
            f"||vjp||={arrays['vjp_norm'][i]:.4e} | "
            f"loss_jvp={arrays['jvp_loss'][i]:.4e} "
            f"loss_vjp={arrays['vjp_loss'][i]:.4e} "
            f"loss_abs_diff={arrays['loss_abs_diff'][i]:.4e} | "
            f"t_jvp10={arrays['jvp_time'][i]:.2f}s "
            f"t_vjp={arrays['vjp_time'][i]:.2f}s"
        )


def print_vjp_only_summary(results):
    arrays = ordered_result_arrays(results)
    control_mode = results.get("control_mode", "nn_param_vector")
    title = "NN Scalar Parameter VJP-Only Summary" if control_mode == "nn_param" else "NN Parameter Vector VJP-Only Summary"
    print(f"\n=== {title} ===")
    for i, n_steps in enumerate(arrays["steps"]):
        memory_mib = arrays.get("vjp_peak_gpu_memory_mib", np.full_like(arrays["steps"], np.nan, dtype=float))[i]
        print(
            f"steps={n_steps:5d} | "
            f"loss={arrays['vjp_loss'][i]:.4e} "
            f"||grad||={arrays['vjp_grad_norm'][i]:.4e} "
            f"grad_min={arrays['vjp_grad_min'][i]:.4e} "
            f"grad_max={arrays['vjp_grad_max'][i]:.4e} "
            f"grad_max_abs={arrays['vjp_grad_max_abs'][i]:.4e} | "
            f"t_vjp={arrays['vjp_time'][i]:.2f}s "
            f"gpu_peak={format_memory_mib(memory_mib)}"
        )


def _vjp_only_memory_series(arrays):
    gpu_mib = arrays.get("vjp_peak_gpu_memory_mib")
    if gpu_mib is not None and np.any(np.isfinite(gpu_mib)):
        return gpu_mib / 1024.0, "Peak GPU memory from nvidia-smi (GiB)"
    jax_peak = arrays.get("vjp_jax_peak_bytes_in_use")
    if jax_peak is not None:
        return jax_peak / (1024.0 ** 3), "JAX peak bytes in use (GiB)"
    return np.full_like(arrays["steps"], np.nan, dtype=float), "Memory (GiB)"


def plot_vjp_only_results(results, gpu_name="unknown", save_path="ad_vjp_only.png"):
    arrays = ordered_result_arrays(results)
    steps = arrays["steps"]
    grad_vector = np.asarray(arrays["vjp_grad_vector"], dtype=np.float64)
    memory_gib, memory_label = _vjp_only_memory_series(arrays)
    control_mode = results.get("control_mode", "nn_param_vector")
    title = "NN scalar parameter VJP only" if control_mode == "nn_param" else "NN parameter-vector VJP only"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title} — GPU: {gpu_name}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(steps, memory_gib, marker="o")
    ax.set_xscale("log", base=3)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel(memory_label)
    ax.set_title("Memory Usage")
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[0, 1]
    ax.plot(steps, arrays["vjp_time"], marker="o")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Compute time (s)")
    ax.set_title("VJP Time")
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[1, 0]
    if grad_vector.ndim == 1:
        grad_vector = grad_vector[:, None]
    for param_index in range(grad_vector.shape[1]):
        label = "selected param" if grad_vector.shape[1] == 1 else f"param {param_index}"
        ax.plot(steps, grad_vector[:, param_index], marker="o", label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xscale("log", base=3)
    set_symlog_for_signed_values(ax, grad_vector)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Selected VJP gradient value")
    ax.set_title("Gradient Entries")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    set_power_of_three_ticks(ax, steps)

    ax = axes[1, 1]
    ax.plot(steps, arrays["vjp_loss"], marker="o")
    ax.set_xscale("log", base=3)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved VJP-only figure to {save_path}")


def plot_param_vector_results(results, gpu_name="unknown", save_path="ad_param_vector.png"):
    arrays = ordered_result_arrays(results)
    steps = arrays["steps"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"NN parameter-vector AD comparison — GPU: {gpu_name}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(steps, arrays["cosine_similarity"], marker="o")
    ax.set_xscale("log", base=3)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Gradient direction agreement")
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[0, 1]
    ax.plot(steps, arrays["relative_error"], marker="o", color="firebrick")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel(r"$||g_{JVP} - g_{VJP}|| / ||g_{VJP}||$")
    ax.set_title("Relative vector error")
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[1, 0]
    ax.plot(steps, arrays["jvp_norm"], marker="o", label="10 JVP directions")
    ax.plot(steps, arrays["vjp_norm"], marker="s", label="VJP selected entries")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Gradient vector norm")
    ax.set_title("Gradient norm")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[1, 1]
    ax.plot(steps, arrays["jvp_loss"], marker="o", label="JVP primal")
    ax.plot(steps, arrays["vjp_loss"], marker="s", label="VJP primal")
    ax.set_xscale("log", base=3)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[2, 0]
    ax.plot(steps, arrays["loss_abs_diff"], marker="o", color="firebrick")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Absolute loss difference")
    ax.set_title("Primal loss mismatch")
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    ax = axes[2, 1]
    ax.plot(steps, arrays["jvp_time"], marker="o", label="10 JVP directions")
    ax.plot(steps, arrays["vjp_time"], marker="s", label="1 VJP full params")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Compute time (s)")
    ax.set_title("Compute time")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    set_power_of_three_ticks(ax, steps)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved parameter-vector figure to {save_path}")


def plot_epsilon_sweep_results(results_by_epsilon, gpu_name="unknown", save_path="ad_comparison_epsilons.png"):
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"NN epsilon AD comparison — GPU: {gpu_name}", fontsize=13)

    for epsilon, results in results_by_epsilon:
        arrays = ordered_result_arrays(results)
        steps = arrays["steps"]
        jvp_grad = arrays["jvp_grad"]
        vjp_grad = arrays["vjp_grad"]
        jvp_loss = arrays["jvp_loss"]
        vjp_loss = arrays["vjp_loss"]
        jvp_time = arrays["jvp_time"]
        vjp_time = arrays["vjp_time"]
        grad_err_pct = np.abs(vjp_grad - jvp_grad) / np.abs(jvp_grad) * 100.0
        loss_err_pct = np.abs(vjp_loss - jvp_loss) / np.abs(jvp_loss) * 100.0
        label = f"eps={epsilon:g}"

        ax = axes[0, 0]
        ax.plot(steps, jvp_grad, marker="o", label=f"JVP {label}")
        ax.plot(steps, vjp_grad, marker="s", linestyle="--", label=f"VJP {label}")

        ax = axes[0, 1]
        ax.plot(steps, grad_err_pct, marker="o", label=label)

        ax = axes[1, 0]
        ax.plot(steps, jvp_loss, marker="o", label=f"JVP {label}")
        ax.plot(steps, vjp_loss, marker="s", linestyle="--", label=f"VJP {label}")

        ax = axes[1, 1]
        ax.plot(steps, loss_err_pct, marker="o", label=label)

        ax = axes[2, 0]
        ax.plot(steps, jvp_time, marker="o", label=f"JVP {label}")
        ax.plot(steps, vjp_time, marker="s", linestyle="--", label=f"VJP {label}")

    panels = [
        (axes[0, 0], "dLoss/dEpsilon vs number of steps", "dLoss/dEpsilon", False),
        (axes[0, 1], "Gradient error between methods", "Gradient error (% of JVP)", True),
        (axes[1, 0], "Loss vs number of steps", "Loss", False),
        (axes[1, 1], "Loss error between methods", "Loss error (% of JVP)", True),
        (axes[2, 0], "Compute time vs number of steps", "Compute time (s)", True),
    ]
    for ax, title, ylabel, log_y in panels:
        ax.set_xscale("log", base=3)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Number of steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        if results_by_epsilon:
            set_power_of_three_ticks(ax, ordered_result_arrays(results_by_epsilon[0][1])["steps"])

    axes[2, 1].axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved epsilon sweep figure to {save_path}")


def main():
    args = parse_args()
    if args.control_mode == "nn_param_vector" and args.method_pair not in ("jvp-vjp", "vjp-only"):
        raise ValueError("--control-mode nn_param_vector supports --method-pair jvp-vjp or vjp-only")
    if args.method_pair == "vjp-only" and args.control_mode not in ("nn_param", "nn_param_vector"):
        raise ValueError("--method-pair vjp-only is currently implemented for --control-mode nn_param or nn_param_vector")
    if args.method_pair == "vjp-only" and args.save_sample_trajectories:
        print("Ignoring --save-sample-trajectories for --method-pair vjp-only.")
        args.save_sample_trajectories = False
    if args.loss_mode != "u2" and args.control_mode not in ("nn_param", "nn_param_vector"):
        raise ValueError("--loss-mode glorys_target is currently implemented for --control-mode nn_param or nn_param_vector")

    output_loss_suffix = "_target" if args.loss_mode == "glorys_target" else ""
    output_reduction_suffix = "_mean" if args.loss_reduction == "mean" else ""
    if args.output_folder is None:
        args.output_folder = f"output_folder_{args.control_mode}{output_loss_suffix}{output_reduction_suffix}"
    if args.output_prefix is None:
        args.output_prefix = f"ad_comparison_{args.control_mode}{output_loss_suffix}{output_reduction_suffix}"
    if args.trajectory_folder is None:
        args.trajectory_folder = os.path.join(args.output_folder, "sample_trajectories")
    os.makedirs(args.output_folder, exist_ok=True)

    steps_list = make_steps_list(args)
    max_steps = max(steps_list)
    gpu_name = get_gpu_name()
    gpu_tag = get_gpu_tag()

    print(f"Running on device: {gpu_name}")
    print(f"GPU tag for output files: {gpu_tag}")
    print(f"n_steps sweep: {steps_list}")
    print(f"sample_steps: {args.sample_steps}")
    print(f"control_mode: {args.control_mode}")
    print(f"method_pair: {args.method_pair}")
    print(f"loss_mode: {args.loss_mode}")
    print(f"loss_reduction: {args.loss_reduction}")
    print(f"veros_diskless_mode: {getattr(runtime_settings, 'diskless_mode', 'n/a')}")
    print(f"use_forcings: {getattr(args, 'use_forcings', False)}")
    print(f"freeze_vars_in_chunked_vjp: {args.freeze_vars}")
    print(f"save_sample_trajectories: {args.save_sample_trajectories}")
    if args.save_sample_trajectories:
        print(f"trajectory_folder: {args.trajectory_folder}")
    if args.control_mode == "A_h":
        control_value = jnp.asarray(args.A_h)
        control_values = [args.A_h]
        print(f"A_h: {args.A_h}")
    elif args.control_mode == "nn_epsilon":
        control_values = args.epsilons if args.epsilons is not None else [args.epsilon]
        print(f"epsilons: {control_values}")
    elif args.control_mode == "nn_param":
        control_values = [0.0]
        print(f"fixed epsilon for NN parameter derivative: {args.epsilon}")
        print(
            "NN parameter selector: "
            f"leaf={args.nn_param_leaf_index}, flat_index={args.nn_param_flat_index}"
        )
    else:
        control_values = [0.0]
        param_seed = args.nn_param_seed if args.nn_param_seed is not None else args.seed
        print(f"fixed epsilon for NN parameter derivative: {args.epsilon}")
        print(f"NN parameter vector: count={args.nn_param_count}, seed={param_seed}")

    if args.plot_only:
        results_by_epsilon = []
        for control_value_float in control_values:
            pkl_path = result_pickle_path(args, gpu_tag, control_value_float)
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Missing result file for --plot-only: {pkl_path}")
            print(f"Loading existing results file: {pkl_path}")
            with open(pkl_path, "rb") as f:
                results = pickle.load(f)

            plot_path = result_plot_path(args, gpu_tag, control_value_float)
            if args.method_pair == "vjp-only":
                plot_vjp_only_results(results, gpu_name=gpu_name, save_path=plot_path)
                print_vjp_only_summary(results)
            elif args.control_mode == "nn_param_vector":
                plot_param_vector_results(results, gpu_name=gpu_name, save_path=plot_path)
                print_param_vector_summary(results)
            else:
                grad_err_pct, loss_err_pct = plot_results(results, gpu_name=gpu_name, save_path=plot_path)
                print_summary(results, grad_err_pct, loss_err_pct)

            if args.control_mode == "nn_epsilon":
                results_by_epsilon.append((control_value_float, results))

        if args.control_mode == "nn_epsilon" and len(results_by_epsilon) > 1:
            sweep_plot_path = os.path.join(
                args.output_folder,
                f"{args.output_prefix}_epsilon_sweep_{gpu_tag}.png",
            )
            plot_epsilon_sweep_results(
                results_by_epsilon,
                gpu_name=gpu_name,
                save_path=sweep_plot_path,
            )
        return

    restart_file = resolve_restart_file(args.restart_file, args.restart_index)
    glorys_file = resolve_input_file(args.glorys_file, args.glorys_index, "GLORYS initial")
    glorys_target_file = resolve_input_file(
        args.glorys_target_file,
        args.glorys_target_index,
        "GLORYS target",
    )
    if args.loss_mode == "glorys_target" and glorys_target_file is None:
        raise ValueError("--loss-mode glorys_target requires --glorys-target-file")

    print(f"restart_file: {restart_file or 'none (setup initial condition)'}")
    print(f"glorys_file: {glorys_file or 'none'}")
    print(f"glorys_target_file: {glorys_target_file or 'none'}")
    simulation = init_simulation(
        warmup=args.warmup,
        restart_file=restart_file,
        glorys_file=glorys_file,
        glorys_time_index=args.glorys_time_index,
        glorys_month_index=args.glorys_month_index,
        glorys_start_month=args.glorys_start_month,
    )
    print("simulation initialized")
    print(
        "setup time parameters: "
        f"dt_mom={simulation.state.settings.dt_mom}, "
        f"dt_tracer={simulation.state.settings.dt_tracer}, "
        f"dt_forcing={getattr(simulation.state.settings, 'dt_forcing', 'n/a')}, "
        f"runlen={simulation.state.settings.runlen}"
    )
    if args.control_mode in ("nn_epsilon", "nn_param", "nn_param_vector"):
        prognostic_keys = get_all_active_keys(simulation.state)
        print("Using all active variables in the chunked VJP carry for NN mode.")
    else:
        prognostic_keys = get_prognostic_keys(simulation.state)
    print(prognostic_keys)

    loss_function = make_benchmark_loss_function(
        args,
        simulation,
        target_file=glorys_target_file,
    )

    nn_params = None
    model_batched_unscaled = None
    nn_stats = None
    nn_model_kwargs = None
    nn_param_description = None
    selected_params = None
    if args.control_mode in ("nn_epsilon", "nn_param", "nn_param_vector"):
        nn_params, model_batched_unscaled, nn_stats, nn_model_kwargs = init_nn_benchmark(
            simulation,
            args,
        )
        if args.control_mode == "nn_param":
            nn_param_description = describe_param_scalar(
                nn_params,
                args.nn_param_leaf_index,
                args.nn_param_flat_index,
            )
            control_values = [nn_param_description["initial_value"]]
            print("Selected NN parameter scalar:")
            print(
                f"  path={nn_param_description['path']} "
                f"shape={nn_param_description['shape']} dtype={nn_param_description['dtype']}"
            )
            print(
                f"  leaf={nn_param_description['leaf_index']}/"
                f"{nn_param_description['n_leaves'] - 1} "
                f"flat_index={nn_param_description['flat_index']}/"
                f"{nn_param_description['leaf_size'] - 1} "
                f"value={nn_param_description['initial_value']:.12e}"
            )
            if args.method_pair == "vjp-only":
                print_nn_correction_stats(
                    simulation,
                    nn_params,
                    model_batched_unscaled,
                    nn_stats,
                    jnp.asarray(args.epsilon),
                    label=(
                        "NN correction tendency stats before VJP-only run "
                        f"(nn_seed={args.seed}, leaf={args.nn_param_leaf_index}, "
                        f"flat_index={args.nn_param_flat_index})"
                    ),
                )
                jax.clear_caches()
                gc.collect()
        elif args.control_mode == "nn_param_vector":
            param_seed = args.nn_param_seed if args.nn_param_seed is not None else args.seed
            selected_params = select_random_param_scalars(
                nn_params,
                args.nn_param_count,
                param_seed,
                path_filter=args.nn_param_path_filter,
            )
            print("Selected NN parameter vector:")
            if args.nn_param_path_filter:
                print(f"  path_filter={args.nn_param_path_filter!r}")
            for i, item in enumerate(selected_params):
                print(
                    f"  [{i:02d}] global={item['global_index']} "
                    f"path={item['path']} shape={item['shape']} "
                    f"leaf={item['leaf_index']} flat_index={item['flat_index']} "
                    f"value={item['initial_value']:.12e}"
                )
            if args.method_pair == "vjp-only":
                print_nn_correction_stats(
                    simulation,
                    nn_params,
                    model_batched_unscaled,
                    nn_stats,
                    jnp.asarray(args.epsilon),
                    label=(
                        "NN correction tendency stats before VJP-only run "
                        f"(nn_seed={args.seed}, param_seed={param_seed})"
                    ),
                )
                jax.clear_caches()
                gc.collect()

    results_by_epsilon = []
    last_errors = None
    last_results = None
    for control_value_float in control_values:
        if args.clear_between_epsilons:
            print("\nClearing JAX caches before next epsilon...")
            jax.clear_caches()
            gc.collect()

        control_value = jnp.asarray(control_value_float)
        if args.control_mode == "nn_epsilon":
            print(f"\n### epsilon = {control_value_float:g} ###")

        pkl_path = result_pickle_path(args, gpu_tag, control_value_float)
        trajectory_prefix = f"{args.output_prefix}{result_suffix_for(args, control_value_float)}_{gpu_tag}"
        existing_results = None
        if os.path.exists(pkl_path):
            print(f"Found existing results file: {pkl_path}")
            with open(pkl_path, "rb") as f:
                existing_results = pickle.load(f)
            print(f"  -> resuming; already have results for steps: {existing_results.get('steps', [])}")

        if args.control_mode == "nn_param" and args.method_pair == "vjp-only":
            results = run_nn_param_vjp_only(
                simulation,
                prognostic_keys,
                steps_list,
                sample_steps=args.sample_steps,
                chunk_base=args.chunk_base,
                freeze_vars=args.freeze_vars,
                nn_params=nn_params,
                model_batched_unscaled=model_batched_unscaled,
                nn_stats=nn_stats,
                epsilon=jnp.asarray(args.epsilon),
                control_value=control_value,
                nn_param_leaf_index=args.nn_param_leaf_index,
                nn_param_flat_index=args.nn_param_flat_index,
                nn_param_description=nn_param_description,
                loss_function=loss_function,
                loss_reduction=args.loss_reduction,
                results=existing_results,
                pkl_path=pkl_path,
            )
        elif args.control_mode == "nn_param_vector" and args.method_pair == "vjp-only":
            results = run_param_vector_vjp_only(
                simulation,
                prognostic_keys,
                steps_list,
                sample_steps=args.sample_steps,
                chunk_base=args.chunk_base,
                freeze_vars=args.freeze_vars,
                nn_params=nn_params,
                model_batched_unscaled=model_batched_unscaled,
                nn_stats=nn_stats,
                epsilon=jnp.asarray(args.epsilon),
                selected_params=selected_params,
                loss_function=loss_function,
                loss_reduction=args.loss_reduction,
                results=existing_results,
                pkl_path=pkl_path,
            )
        elif args.control_mode == "nn_param_vector":
            results = run_param_vector_comparison(
                simulation,
                prognostic_keys,
                steps_list,
                sample_steps=args.sample_steps,
                chunk_base=args.chunk_base,
                freeze_vars=args.freeze_vars,
                nn_params=nn_params,
                model_batched_unscaled=model_batched_unscaled,
                nn_stats=nn_stats,
                epsilon=jnp.asarray(args.epsilon),
                selected_params=selected_params,
                loss_function=loss_function,
                loss_reduction=args.loss_reduction,
                save_sample_trajectories=args.save_sample_trajectories,
                trajectory_folder=args.trajectory_folder,
                trajectory_prefix=trajectory_prefix,
                results=existing_results,
                pkl_path=pkl_path,
            )
        else:
            results = run_comparison(
                simulation,
                prognostic_keys,
                steps_list,
                sample_steps=args.sample_steps,
                chunk_base=args.chunk_base,
                control_mode=args.control_mode,
                control_value=control_value,
                method_pair=args.method_pair,
                freeze_vars=args.freeze_vars,
                nn_params=nn_params,
                model_batched_unscaled=model_batched_unscaled,
                nn_stats=nn_stats,
                nn_param_leaf_index=args.nn_param_leaf_index,
                nn_param_flat_index=args.nn_param_flat_index,
                epsilon=jnp.asarray(args.epsilon),
                loss_function=loss_function,
                loss_reduction=args.loss_reduction,
                save_sample_trajectories=args.save_sample_trajectories,
                trajectory_folder=args.trajectory_folder,
                trajectory_prefix=trajectory_prefix,
                results=existing_results,
                pkl_path=pkl_path,
            )
        results["gpu_name"] = gpu_name
        results["gpu_tag"] = gpu_tag
        results["control_mode"] = args.control_mode
        results["method_pair"] = args.method_pair
        results["freeze_vars"] = args.freeze_vars
        results["A_h"] = args.A_h
        results["epsilon"] = control_value_float if args.control_mode == "nn_epsilon" else args.epsilon
        results["epsilon_values"] = control_values if args.control_mode == "nn_epsilon" else None
        results["nn_checkpoint"] = args.nn_checkpoint
        results["nn_model_kwargs"] = nn_model_kwargs
        results["use_forcings"] = args.use_forcings
        results["nn_param"] = nn_param_description
        results["nn_param_vector"] = selected_params
        results["nn_param_path_filter"] = args.nn_param_path_filter
        results["nn_param_seed"] = args.nn_param_seed if args.nn_param_seed is not None else args.seed
        results["sample_steps"] = args.sample_steps
        results["restart_file"] = restart_file
        results["glorys_file"] = glorys_file
        results["glorys_target_file"] = glorys_target_file
        results["glorys_time_index"] = args.glorys_time_index
        results["glorys_month_index"] = args.glorys_month_index
        results["glorys_start_month"] = args.glorys_start_month
        results["loss_mode"] = args.loss_mode
        results["loss_reduction"] = args.loss_reduction
        results["keys_loss"] = list(args.keys_loss)
        results["save_sample_trajectories"] = args.save_sample_trajectories
        results["trajectory_folder"] = args.trajectory_folder
        save_results(results, pkl_path)

        plot_path = result_plot_path(args, gpu_tag, control_value_float)
        if args.method_pair == "vjp-only":
            plot_vjp_only_results(results, gpu_name=gpu_name, save_path=plot_path)
            print_vjp_only_summary(results)
        elif args.control_mode == "nn_param_vector":
            plot_param_vector_results(results, gpu_name=gpu_name, save_path=plot_path)
            print_param_vector_summary(results)
        else:
            grad_err_pct, loss_err_pct = plot_results(results, gpu_name=gpu_name, save_path=plot_path)
            last_errors = (grad_err_pct, loss_err_pct)
            print_summary(results, grad_err_pct, loss_err_pct)
        last_results = results

        if args.control_mode == "nn_epsilon":
            results_by_epsilon.append((control_value_float, results))

    if args.control_mode == "nn_epsilon" and len(results_by_epsilon) > 1:
        sweep_plot_path = os.path.join(
            args.output_folder,
            f"{args.output_prefix}_epsilon_sweep_{gpu_tag}.png",
        )
        plot_epsilon_sweep_results(
            results_by_epsilon,
            gpu_name=gpu_name,
            save_path=sweep_plot_path,
        )

    del last_errors, last_results


if __name__ == "__main__":
    main()

