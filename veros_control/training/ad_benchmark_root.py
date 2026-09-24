import argparse
from contextlib import contextmanager
import gc
import glob
import os
import pickle
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

import jax
import jax.numpy as jnp
import optax
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

from veros_control.training.ad_benchmark import (
    agg_sum,
    describe_param_scalar,
    get_gpu_name,
    get_gpu_tag,
    get_all_active_keys,
    init_nn_benchmark,
    init_simulation,
    make_chunk_sizes,
    make_steps_list,
    plot_results,
    plot_param_vector_results,
    plot_vjp_only_results,
    print_summary,
    print_param_vector_summary,
    print_nn_correction_stats,
    print_vjp_only_summary,
    resolve_input_file,
    resolve_restart_file,
    result_pickle_path,
    result_plot_path,
    result_suffix_for,
    reduce_sampled_losses,
    run_comparison,
    run_nn_param_vjp_only,
    nn_epsilon_step,
    run_param_vector_comparison,
    run_param_vector_vjp_only,
    save_results,
    select_random_param_scalars,
)
from veros_control.training.loss import compute_loss
from veros_control.training.train import get_prognostic_keys, inject_dynamic_vars, load_glorys_snapshot
from veros_control.core.checkpointing import load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Root AD benchmark: compare JVP and chunked VJP from restart or GLORYS "
            "initial conditions with the old u**2 loss or a next-GLORYS temp/salt loss."
        )
    )
    parser.add_argument(
        "--run-mode",
        choices=("benchmark", "train"),
        default="benchmark",
        help="Run the usual benchmark, or train the NN parameters with the benchmark VJP loss.",
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
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=2,
        help="Compute the selected loss every N steps. Default: 2, matching the old benchmark parser.",
    )
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
        help="Compare JVP against chunked VJP, repeat the same JVP twice, or run VJP only.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("A_h", "nn_epsilon", "nn_param", "nn_param_vector"),
        default="nn_param",
        help="Scalar sensitivity to benchmark. Default: nn_param.",
    )
    parser.add_argument("--A-h", type=float, default=5e4)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="NN correction amplitude for nn_epsilon or fixed amplitude for nn_param.",
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=None,
        help="Epsilon values to sweep in nn_epsilon mode. Defaults to --epsilon.",
    )
    parser.add_argument(
        "--restart-file",
        default=None,
        help="Restart file, or glob pattern, used as initial state.",
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
        help="GLORYS initial-condition file, or glob pattern, used as initial state.",
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
        help=(
            "Optional GLORYS target file, or glob pattern, for --loss-mode glorys_target. "
            "If omitted, the target is selected from --glorys-file with index --glorys-index + 1."
        ),
    )
    parser.add_argument(
        "--glorys-target-index",
        type=int,
        default=None,
        help=(
            "Index to select when --glorys-target-file is a glob pattern. "
            "Default: 0 for --glorys-target-file, or --glorys-index + 1 for automatic next target."
        ),
    )
    parser.add_argument(
        "--glorys-linear-interpolation",
        action="store_true",
        help=(
            "Interpolate the GLORYS target with alpha=n_steps/720 between the initial "
            "and next snapshots. Requires glorys_target loss and a single rollout length "
            "in [0, 720]. Uses this fixed target at every sampled trajectory step."
        ),
    )
    parser.add_argument(
        "--loss-mode",
        choices=("u2", "glorys_target"),
        default="u2",
        help="Loss used by the root AD benchmark. Default: old u**2 loss.",
    )
    parser.add_argument(
        "--balance-loss-terms",
        action="store_true",
        help=(
            "Train with mean(L_key / (initial_rollout_L_key + 1e-12)). "
            "GLORYS target loss only; fixed denominators are saved and restored on resume."
        ),
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
    parser.add_argument("--warmup", type=int, default=0)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--zero-output-init",
        action="store_true",
        help="Initialize NN output Dense layers with zero kernel and bias for fresh parameters.",
    )
    parser.add_argument(
        "--nn-param-leaf-index",
        type=int,
        default=0,
        help="Parameter pytree leaf for --control-mode nn_param. Default: 0.",
    )
    parser.add_argument(
        "--nn-param-flat-index",
        type=int,
        default=0,
        help="Flat scalar index inside the selected NN parameter leaf. Default: 0.",
    )
    parser.add_argument(
        "--nn-param-count",
        type=int,
        default=10,
        help="Number of random NN scalar parameters for --control-mode nn_param_vector. Default: 10.",
    )
    parser.add_argument(
        "--nn-param-seed",
        type=int,
        default=None,
        help="Seed used to select random NN scalar parameters. Default: --seed.",
    )
    parser.add_argument(
        "--nn-param-path-filter",
        default=None,
        help="Restrict random NN scalar parameters to leaves whose path contains this text.",
    )
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load existing result pickle files and regenerate figures.",
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
        help="Carry the full Veros state in chunked VJP.",
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
    parser.add_argument(
        "--train-n-steps",
        type=int,
        default=None,
        help=(
            "Rollout length for --run-mode train. If omitted, training requires "
            "the step selection to contain exactly one n_steps value."
        ),
    )
    parser.add_argument(
        "--train-updates",
        type=int,
        default=1,
        help="Number of optimizer updates for --run-mode train. Default: 1.",
    )
    parser.add_argument(
        "--train-lr",
        type=float,
        default=1e-8,
        help="Learning rate for --run-mode train. Default: 1e-8.",
    )
    parser.add_argument(
        "--train-lr-final",
        type=float,
        default=None,
        help="Optional final learning rate for --run-mode train. Enables per-update LR scheduling.",
    )
    parser.add_argument(
        "--train-lr-schedule",
        choices=("linear", "geometric"),
        default="linear",
        help="Schedule used between --train-lr and --train-lr-final. Default: linear.",
    )
    parser.add_argument(
        "--train-epsilon-final",
        type=float,
        default=None,
        help="Optional final epsilon for --run-mode train. Enables per-update epsilon scheduling.",
    )
    parser.add_argument(
        "--train-epsilon-schedule",
        choices=("linear", "geometric"),
        default="linear",
        help="Schedule used between --epsilon and --train-epsilon-final. Default: linear.",
    )
    parser.add_argument(
        "--train-grad-clip-norm",
        type=float,
        default=None,
        help="Optional global gradient clipping norm for --run-mode train.",
    )
    parser.add_argument(
        "--train-output-file",
        default=None,
        help="Checkpoint path for --run-mode train. Default: <output-folder>/<output-prefix>_train_<gpu>.pkl.",
    )
    parser.add_argument(
        "--train-save-every",
        type=int,
        default=1,
        help="Save the train checkpoint every N updates. Default: 1.",
    )
    parser.add_argument(
        "--train-overwrite",
        action="store_true",
        help="Start a fresh train checkpoint even if --train-output-file already exists.",
    )
    parser.add_argument(
        "--train-fail-on-nonfinite",
        action="store_true",
        default=True,
        help="Stop training when loss, gradients, updates, or parameters are non-finite.",
    )
    parser.add_argument(
        "--train-allow-nonfinite",
        dest="train_fail_on_nonfinite",
        action="store_false",
        help="Do not stop when train diagnostics contain non-finite values.",
    )
    parser.set_defaults(freeze_vars=True)

    parser.add_argument('--basin-mean-loss', action='store_true', help='Compare area-weighted T/S means separately for each of six basins and shallow level')
    parser.add_argument('--basin-levels', type=int, choices=(5, 10), default=5, help='Number of shallowest levels for basin means (default 5)')
    args = parser.parse_args()
    args.keys_loss = tuple(args.keys_loss)
    return args


def _print_precision():
    requested_float64 = jnp.asarray(1.0, dtype=jnp.float64).dtype
    print(f"jax_enable_x64: {jax.config.read('jax_enable_x64')}")
    print(f"float default: {jnp.asarray(1.0).dtype}")
    print(f"float64 requested: {requested_float64}")


def _load_existing_results(args, gpu_tag, control_values):
    loaded = []
    for control_value_float in control_values:
        pkl_path = result_pickle_path(args, gpu_tag, control_value_float)
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing result file for --plot-only: {pkl_path}")
        print(f"Loading existing results file: {pkl_path}")
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
        loaded.append((control_value_float, results, pkl_path))
    return loaded


def _resolve_glorys_target_file(args):
    if args.loss_mode != "glorys_target":
        return None

    if args.glorys_target_file is not None:
        target_index = args.glorys_target_index
        if target_index is None:
            target_index = 0
        return resolve_input_file(args.glorys_target_file, target_index, "GLORYS target")

    if args.glorys_file is None:
        raise ValueError(
            "--loss-mode glorys_target requires --glorys-file for automatic next-month "
            "target selection, or an explicit --glorys-target-file"
        )

    if not glob.has_magic(args.glorys_file):
        raise ValueError(
            "Automatic next-GLORYS target selection requires --glorys-file to be a glob pattern. "
            "Pass --glorys-target-file when using a single initial GLORYS file."
        )

    target_index = args.glorys_target_index
    if target_index is None:
        target_index = args.glorys_index + 1
    return resolve_input_file(args.glorys_file, target_index, "next GLORYS target")


def _glorys_interpolation_alpha(args, steps_list):
    if not args.glorys_linear_interpolation:
        return None
    if args.loss_mode != "glorys_target" or args.glorys_file is None:
        raise ValueError(
            "--glorys-linear-interpolation requires --loss-mode glorys_target and --glorys-file"
        )
    if args.run_mode == "train":
        n_steps = _resolve_train_n_steps(args, steps_list)
    elif len(steps_list) == 1:
        n_steps = int(steps_list[0])
    else:
        raise ValueError("--glorys-linear-interpolation requires one rollout length")
    if not 0 <= n_steps <= 720:
        raise ValueError("GLORYS linear interpolation requires 0 <= n_steps <= 720")
    return n_steps / 720.0


def make_root_loss_functions(args, simulation, target_file=None, initial_file=None):
    if args.loss_mode == "u2":
        return agg_sum, (), None

    if args.loss_mode == "glorys_target":
        if target_file is None:
            raise ValueError("--loss-mode glorys_target requires --glorys-target-file")

        loss_keys = tuple(args.keys_loss)
        target = load_glorys_snapshot(
            simulation,
            target_file,
            loss_keys,
            time_index=args.glorys_time_index,
        )

        if args.glorys_linear_interpolation:
            if initial_file is None:
                raise ValueError("GLORYS interpolation requires the initial GLORYS snapshot")
            initial_target = load_glorys_snapshot(
                simulation, initial_file, loss_keys, time_index=args.glorys_time_index,
            )
            # Interpolate GLORYS snapshots at the rollout time (alpha = n_steps / 720).
            # For 81 steps, the next snapshot contributes 11.25% of the target.
            alpha = args.glorys_interpolation_alpha
            target = {
                key: initial_target[key] * (1.0 - alpha) + target[key] * alpha
                for key in loss_keys
            }

        if getattr(args, "basin_mean_loss", False):
            from veros_control.training.basin_loss import make_basin_loss
            scalar, names, terms, metadata = make_basin_loss(
                simulation.state.variables, target, loss_keys, args.basin_levels,
            )
            args.basin_loss_metadata = metadata
            print(f"Basin means: {metadata['names']}; depths (m): {metadata['depths_m']}")
            return scalar, names, terms

        def loss_function(state):
            return compute_loss(
                state,
                target,
                keys=loss_keys,
                tau=state.variables.tau,
            )

        def loss_terms_function(state):
            return jnp.stack(
                [
                    compute_loss(
                        state,
                        target,
                        keys=(key,),
                        tau=state.variables.tau,
                    )
                    for key in loss_keys
                ]
            )

        return loss_function, loss_keys, loss_terms_function

    raise ValueError(f"Unsupported loss mode: {args.loss_mode}")


def _reduce_sampled_loss_terms(loss_terms, iterations, sample_steps, loss_reduction):
    flat_terms = jnp.reshape(loss_terms, (-1, loss_terms.shape[-1]))
    total = jnp.sum(flat_terms, axis=0)
    if loss_reduction == "sum":
        return total
    if loss_reduction == "mean":
        sample_count = max(int(iterations) // int(sample_steps), 1)
        return total / jnp.asarray(sample_count, dtype=total.dtype)
    raise ValueError(f"Unsupported loss reduction: {loss_reduction}")


def _safe_get_variable(variables, name):
    try:
        return getattr(variables, name)
    except (AttributeError, RuntimeError):
        return None


def _field_at_tau(value, tau):
    value = jnp.asarray(value)
    if value.ndim == 4:
        return value[..., tau]
    return value


def _masked_rms_max_abs(value, mask):
    value = jnp.asarray(value)
    mask = jnp.asarray(mask, dtype=bool)
    finite = jnp.isfinite(value)
    valid = mask & finite
    count = jnp.sum(valid)
    safe = jnp.where(valid, value, 0.0)
    rms = jnp.sqrt(jnp.sum(safe * safe) / jnp.maximum(count, 1))
    max_abs = jnp.where(
        count > 0,
        jnp.max(jnp.where(valid, jnp.abs(value), 0.0)),
        jnp.nan,
    )
    return rms, max_abs


@contextmanager
def _temporary_solver_cache():
    """Release solver entries for temporary diagnostic / AD state copies only.

    Existing entries (including a solver for the persistent simulation state) are
    preserved. Removing a cache entry does not change or invalidate the solver.
    """
    from veros.core.external.solvers import get_linear_solver

    cache = getattr(get_linear_solver, "cache", None)
    if not isinstance(cache, dict):
        # Other Veros versions may manage solver lifetime without this cache.
        yield
        return
    existing_keys = set(cache)
    try:
        yield
    finally:
        for key in list(cache):
            if key not in existing_keys:
                del cache[key]


@_temporary_solver_cache()
def tendency_ratio_diagnostics(
    simulation,
    params,
    model_batched_unscaled,
    nn_stats,
    epsilon,
):
    base_state = simulation.state
    tau = int(jax.device_get(base_state.variables.tau))

    physics_state = base_state.copy()
    with physics_state.settings.unlock():
        physics_state.settings.baseline_model = True
    from veros_control.training.train import zero_tendency_sources

    zero_tendency_sources(physics_state)
    simulation.step(physics_state)
    jax.block_until_ready(physics_state)

    specs = (
        ("temp", "dtemp_nn", "dtemp", "maskT"),
        ("salt", "dsalt_nn", "dsalt", "maskT"),
        ("u", "du_nn", "du", "maskU"),
        ("v", "dv_nn", "dv", "maskV"),
    )

    physics_stats = {}
    for name, _, phys_name, mask_name in specs:
        phys_value = _safe_get_variable(physics_state.variables, phys_name)
        mask = _safe_get_variable(base_state.variables, mask_name)
        if phys_value is None or mask is None:
            physics_stats[name] = {
                "phys_rms": np.nan,
                "phys_max_abs": np.nan,
            }
            continue

        phys = _field_at_tau(phys_value, tau)
        phys_rms, phys_max_abs = _masked_rms_max_abs(phys, mask)
        phys_rms, phys_max_abs = jax.device_get((phys_rms, phys_max_abs))
        physics_stats[name] = {
            "phys_rms": float(phys_rms),
            "phys_max_abs": float(phys_max_abs),
        }

    del physics_state
    gc.collect()

    nn_state = nn_epsilon_step(
        base_state,
        simulation,
        params,
        model_batched_unscaled,
        nn_stats,
        jnp.asarray(epsilon),
    )
    jax.block_until_ready(nn_state)

    out = {}
    for name, corr_name, _, mask_name in specs:
        corr_value = _safe_get_variable(nn_state.variables, corr_name)
        mask = _safe_get_variable(base_state.variables, mask_name)
        if corr_value is None or mask is None:
            corr_rms = np.nan
            corr_max_abs = np.nan
        else:
            corr = _field_at_tau(corr_value, tau)
            corr_rms, corr_max_abs = _masked_rms_max_abs(corr, mask)
            corr_rms, corr_max_abs = jax.device_get((corr_rms, corr_max_abs))
            corr_rms = float(corr_rms)
            corr_max_abs = float(corr_max_abs)

        phys_rms = physics_stats[name]["phys_rms"]
        phys_max_abs = physics_stats[name]["phys_max_abs"]
        rms_ratio = corr_rms / phys_rms if phys_rms > 0 else np.nan
        max_ratio = corr_max_abs / phys_max_abs if phys_max_abs > 0 else np.nan
        out[name] = {
            "rms_ratio": rms_ratio,
            "max_ratio": max_ratio,
            "corr_rms": corr_rms,
            "phys_rms": phys_rms,
            "corr_max_abs": corr_max_abs,
            "phys_max_abs": phys_max_abs,
        }

    del nn_state
    gc.collect()
    return out


def _format_named_values(values):
    return " ".join(f"{name}={value:.6e}" for name, value in values.items())


def _format_tendency_ratio_field(diag, field):
    return " ".join(f"{name}={diag[name][field]:.6e}" for name in ("temp", "salt", "u", "v"))


def _tree_float_leaves(tree):
    leaves = []
    for leaf in jax.tree_util.tree_leaves(tree):
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or dtype == jax.dtypes.float0:
            continue
        if jnp.issubdtype(dtype, jnp.inexact):
            leaves.append(jnp.asarray(leaf))
    return leaves


def _tree_stats(tree):
    leaves = _tree_float_leaves(tree)
    if not leaves:
        return {
            "finite": True,
            "norm": 0.0,
            "max_abs": 0.0,
            "bad_count": 0,
        }

    finite = jnp.asarray(True)
    norm_sq = jnp.asarray(0.0)
    max_abs = jnp.asarray(0.0)
    bad_count = jnp.asarray(0)
    for leaf in leaves:
        leaf_finite = jnp.isfinite(leaf)
        abs_leaf = jnp.abs(leaf)
        finite = finite & jnp.all(leaf_finite)
        norm_sq = norm_sq + jnp.sum(jnp.where(leaf_finite, abs_leaf * abs_leaf, 0.0))
        max_abs = jnp.maximum(
            max_abs,
            jnp.max(jnp.where(leaf_finite, abs_leaf, jnp.inf)),
        )
        bad_count = bad_count + jnp.sum(~leaf_finite)

    finite, norm, max_abs, bad_count = jax.device_get((
        finite,
        jnp.sqrt(norm_sq),
        max_abs,
        bad_count,
    ))
    return {
        "finite": bool(finite),
        "norm": float(norm),
        "max_abs": float(max_abs),
        "bad_count": int(bad_count),
    }


def _tree_path_name(path):
    if hasattr(jax.tree_util, "keystr"):
        return jax.tree_util.keystr(path)
    return "/".join(str(entry) for entry in path)


def _bad_tree_leaf_summaries(tree, limit=8):
    summaries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        dtype = getattr(leaf, "dtype", None)
        if dtype is None or dtype == jax.dtypes.float0:
            continue
        if not jnp.issubdtype(dtype, jnp.inexact):
            continue

        arr = jnp.asarray(leaf)
        finite = jnp.isfinite(arr)
        if bool(jax.device_get(jnp.all(finite))):
            continue

        bad_count, has_nan, has_pos_inf, has_neg_inf = jax.device_get((
            jnp.sum(~finite),
            jnp.any(jnp.isnan(arr)),
            jnp.any(arr == jnp.inf),
            jnp.any(arr == -jnp.inf),
        ))
        flags = []
        if bool(has_nan):
            flags.append("nan")
        if bool(has_pos_inf):
            flags.append("+inf")
        if bool(has_neg_inf):
            flags.append("-inf")
        summaries.append(
            f"{_tree_path_name(path)} shape={getattr(arr, 'shape', ())} "
            f"bad={int(bad_count)}/{arr.size} ({','.join(flags)})"
        )
        if len(summaries) >= limit:
            break
    return summaries


def _tree_scalar_value(tree, leaf_index, flat_index):
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    if leaf_index < 0 or leaf_index >= len(path_leaves):
        return np.nan
    _, leaf = path_leaves[leaf_index]
    flat_leaf = jnp.ravel(leaf)
    if flat_index < 0 or flat_index >= flat_leaf.size:
        return np.nan
    return float(jax.device_get(flat_leaf[flat_index]))


def _make_train_optimizer(args):
    if args.train_grad_clip_norm is None:
        return optax.adamw(1.0)
    return optax.chain(
        optax.clip_by_global_norm(args.train_grad_clip_norm),
        optax.adamw(1.0),
    )


def _scheduled_value_for_update(start_value, final_value, schedule, update_idx, start_update, final_update, name):
    if final_value is None:
        return float(start_value)
    n_scheduled = max(final_update - start_update - 1, 1)
    progress = (update_idx - start_update) / n_scheduled
    progress = min(max(progress, 0.0), 1.0)

    if schedule == "linear":
        return float(start_value + progress * (final_value - start_value))

    if start_value <= 0 or final_value <= 0:
        raise ValueError(f"--train-{name}-schedule geometric requires positive start and final values")
    return float(start_value * (final_value / start_value) ** progress)


def _train_lr_for_update(args, update_idx, start_update, final_update):
    return _scheduled_value_for_update(
        args.train_lr,
        args.train_lr_final,
        args.train_lr_schedule,
        update_idx,
        start_update,
        final_update,
        "lr",
    )


def _train_epsilon_for_update(args, update_idx, start_update, final_update):
    return _scheduled_value_for_update(
        args.epsilon,
        args.train_epsilon_final,
        args.train_epsilon_schedule,
        update_idx,
        start_update,
        final_update,
        "epsilon",
    )


def _scale_update(update, lr):
    dtype = getattr(update, "dtype", None)
    if dtype is None or dtype == jax.dtypes.float0:
        return update
    return update * jnp.asarray(lr, dtype=dtype)


def _resolve_train_n_steps(args, steps_list):
    if args.train_n_steps is not None:
        return args.train_n_steps
    if len(steps_list) == 1:
        return int(steps_list[0])
    raise ValueError(
        "--run-mode train needs one rollout length. Pass --train-n-steps 729, "
        "--n-steps 729, or --n-step-powers 6 6."
    )


def _train_checkpoint_path(args, gpu_tag):
    if args.train_output_file is not None:
        return args.train_output_file
    result_suffix = result_suffix_for(args, 0.0)
    return os.path.join(
        args.output_folder,
        f"{args.output_prefix}{result_suffix}_train_{gpu_tag}.pkl",
    )


def _train_checkpoint_payload(
    args,
    params,
    opt_state,
    history,
    update_index,
    gpu_name,
    gpu_tag,
    restart_file,
    glorys_file,
    glorys_target_file,
    nn_model_kwargs,
    nn_stats,
    train_n_steps,
):
    return {
        "params": params,
        "opt_state": opt_state,
        "train_history": history,
        "losses": [item["loss"] for item in history],
        "update": update_index,
        "train_mode": "ad_benchmark_root",
        "run_mode": args.run_mode,
        "gpu_name": gpu_name,
        "gpu_tag": gpu_tag,
        "hidden_dim_nn": args.hidden_dim,
        "depth_nn": args.depth,
        "epsilon": args.epsilon,
        "epsilon_final": args.train_epsilon_final,
        "epsilon_schedule": args.train_epsilon_schedule,
        "train_epsilon": False,
        "use_lon_lat": args.use_lon_lat,
        "use_forcings": args.use_forcings,
        "conservative_mixing": args.conservative_mixing,
        "use_one_hot": args.use_one_hot,
        "num_mlps": args.num_mlps,
        "TS": args.TS,
        "zero_output_init": args.zero_output_init,
        "stats_features": nn_stats,
        "feature_stats_file": args.feature_stats_file,
        "normalize_inputs": getattr(args, "normalize_inputs", nn_stats is not None),
        "scaling_method_features": getattr(args, "scaling_method_features", None),
        "nn_model_kwargs": nn_model_kwargs,
        "control_mode": args.control_mode,
        "method_pair": "vjp-only",
        "n_steps": train_n_steps,
        "sample_steps": args.sample_steps,
        "chunk_base": args.chunk_base,
        "freeze_vars": args.freeze_vars,
        "balance_loss_terms": args.balance_loss_terms,
        "loss_term_denominators": args.loss_term_denominators,
        "glorys_linear_interpolation": args.glorys_linear_interpolation,
        "glorys_interpolation_alpha": args.glorys_interpolation_alpha,
        "loss_mode": args.loss_mode,
        "loss_reduction": args.loss_reduction,
        "keys_loss": list(args.keys_loss),
        "basin_mean_loss": getattr(args, "basin_mean_loss", False),
        "basin_levels": getattr(args, "basin_levels", 5) if getattr(args, "basin_mean_loss", False) else None,
        "basin_loss_metadata": getattr(args, "basin_loss_metadata", None),
        "restart_file": restart_file,
        "glorys_file": glorys_file,
        "glorys_target_file": glorys_target_file,
        "glorys_time_index": args.glorys_time_index,
        "glorys_month_index": args.glorys_month_index,
        "glorys_start_month": args.glorys_start_month,
        "lr": args.train_lr,
        "lr_final": args.train_lr_final,
        "lr_schedule": args.train_lr_schedule,
        "grad_clip_norm": args.train_grad_clip_norm,
        "seed": args.seed,
    }


def _validate_loss_denominators(denominators, keys):
    if not isinstance(denominators, dict) or set(denominators) != set(keys):
        raise ValueError("Loss balancing denominators must match --keys-loss")
    values = {key: float(denominators[key]) for key in keys}
    if any(not np.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("Loss balancing denominators must be finite and positive")
    return values


def _initial_loss_denominators(loss_terms, keys):
    values = np.asarray(jax.device_get(loss_terms), dtype=np.float64)
    if values.shape != (len(keys),) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Initial rollout loss terms must be finite and nonnegative")
    return _validate_loss_denominators(
        {key: float(values[index]) + 1e-12 for index, key in enumerate(keys)}, keys,
    )


def _load_or_init_train_state(args, checkpoint_path, params, optimizer):
    args.loss_term_denominators = None
    checkpoint = None
    if os.path.exists(checkpoint_path) and not args.train_overwrite:
        checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is not None and "params" in checkpoint:
        basin_enabled = getattr(args, "basin_mean_loss", False)
        if checkpoint.get("basin_mean_loss", False) != basin_enabled or (basin_enabled and checkpoint.get("basin_levels") != args.basin_levels):
            raise ValueError("Checkpoint basin objective differs; start a separate training output (optionally initialize NN weights with --nn-checkpoint)")
        if (
            checkpoint.get("glorys_linear_interpolation", False) != args.glorys_linear_interpolation
            or (
                args.glorys_linear_interpolation
                and checkpoint.get("glorys_interpolation_alpha") != args.glorys_interpolation_alpha
            )
        ):
            raise ValueError("Checkpoint GLORYS interpolation differs; use a new --train-output-file")
        if checkpoint.get("balance_loss_terms", False) != args.balance_loss_terms:
            raise ValueError("Checkpoint loss balancing differs; use a new --train-output-file")
        if args.balance_loss_terms:
            for key, expected in (
                ("n_steps", _resolve_train_n_steps(args, make_steps_list(args))),
                ("sample_steps", args.sample_steps),
                ("loss_reduction", args.loss_reduction),
            ):
                if checkpoint.get(key) != expected:
                    raise ValueError(f"Balanced checkpoint {key} differs; use a new --train-output-file")
            args.loss_term_denominators = _validate_loss_denominators(
                checkpoint.get("loss_term_denominators"), args.keys_loss,
            )
        params = checkpoint["params"]
        opt_state = checkpoint.get("opt_state")
        if opt_state is None:
            opt_state = optimizer.init(params)
        history = list(checkpoint.get("train_history", []))
        start_update = int(checkpoint.get("update", len(history)))
        print(f"Resuming train mode from update {start_update}: {checkpoint_path}")
        return params, opt_state, history, start_update

    if os.path.exists(checkpoint_path) and args.train_overwrite:
        print(f"Overwriting existing train checkpoint: {checkpoint_path}")
    else:
        print(f"Starting train mode from current NN parameters: {checkpoint_path}")

    return params, optimizer.init(params), [], 0


class NNParamsTrainChunkedVJP:
    def __init__(
        self,
        simulation,
        model_batched_unscaled,
        stats,
        loss_function,
        freeze_vars=True,
        loss_reduction="sum",
        loss_terms_function=None,
        loss_term_count=0,
    ):
        self.simulation = simulation
        self.model_batched_unscaled = model_batched_unscaled
        self.stats = stats
        self.loss_function = loss_function
        self.freeze_vars = freeze_vars
        self.loss_reduction = loss_reduction
        self.loss_terms_function = loss_terms_function
        self.loss_term_count = loss_term_count
        self.loss_term_denominators = None

    def step_function(self, state, nn_params, epsilon):
        return nn_epsilon_step(
            state,
            self.simulation,
            nn_params,
            self.model_batched_unscaled,
            self.stats,
            epsilon,
        )

    @_temporary_solver_cache()
    def g(
        self,
        full_state,
        nn_params,
        prognostic_keys,
        chunk_sizes,
        sample_steps=1,
        epsilon=1.0,
        return_aux=False,
        forward_only=False,
    ):
        iterations = 1
        for chunk_size in chunk_sizes:
            iterations *= chunk_size
        collect_loss_terms = (return_aux or self.loss_term_denominators is not None) and self.loss_terms_function is not None
        if self.loss_term_denominators is not None and not collect_loss_terms:
            raise ValueError("Loss balancing requires per-variable loss terms")

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

        def sampled_loss_output(state):
            if not collect_loss_terms:
                return self.loss_function(state)
            return self.loss_function(state), self.loss_terms_function(state)

        def zero_loss_output():
            if not collect_loss_terms:
                return jnp.asarray(0.0)
            return (
                jnp.asarray(0.0),
                jnp.zeros((self.loss_term_count,), dtype=jnp.asarray(0.0).dtype),
            )

        def forward_trajectory(params_arg, epsilon_arg):
            if self.freeze_vars:
                init_carry = {
                    key: getattr(full_state.variables, key)
                    for key in prognostic_keys
                }

                @jax.checkpoint
                def innermost_step(dynamic_vars, step_idx):
                    curr_state = inject_dynamic_vars(full_state, dict(dynamic_vars))
                    next_state = self.step_function(curr_state, params_arg, epsilon_arg)
                    next_carry = {
                        key: getattr(next_state.variables, key)
                        for key in prognostic_keys
                    }
                    is_sample_step = (step_idx % sample_steps) == 0
                    step_loss_output = jax.lax.cond(
                        is_sample_step,
                        sampled_loss_output,
                        lambda state: zero_loss_output(),
                        next_state,
                    )
                    return next_carry, step_loss_output

                initial_scan_carry = init_carry
            else:
                init_state = full_state.copy()

                @jax.checkpoint
                def innermost_step(carry_state, step_idx):
                    next_state = self.step_function(carry_state, params_arg, epsilon_arg)
                    is_sample_step = (step_idx % sample_steps) == 0
                    step_loss_output = jax.lax.cond(
                        is_sample_step,
                        sampled_loss_output,
                        lambda state: zero_loss_output(),
                        next_state,
                    )
                    return next_state, step_loss_output

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
            if not collect_loss_terms:
                return reduce_sampled_losses(
                    all_chunk_losses,
                    iterations,
                    sample_steps,
                    self.loss_reduction,
                )

            all_chunk_scalar_losses, all_chunk_loss_terms = all_chunk_losses
            scalar_loss = reduce_sampled_losses(
                all_chunk_scalar_losses,
                iterations,
                sample_steps,
                self.loss_reduction,
            )
            loss_terms = _reduce_sampled_loss_terms(
                all_chunk_loss_terms,
                iterations,
                sample_steps,
                self.loss_reduction,
            )
            if self.loss_term_denominators is not None:
                # Constants from the first rollout, never differentiated or recomputed per update.
                scalar_loss = jnp.mean(loss_terms / self.loss_term_denominators)
            return scalar_loss, loss_terms

        if forward_only:
            return forward_trajectory(nn_params, epsilon)
        if collect_loss_terms:
            (loss, loss_terms), grads = jax.value_and_grad(
                forward_trajectory, argnums=0, has_aux=True,
            )(nn_params, epsilon)
            if return_aux:
                return loss, grads, loss_terms
            return loss, grads
        if return_aux:
            loss, grads = jax.value_and_grad(forward_trajectory, argnums=0)(nn_params, epsilon)
            return loss, grads, None
        return jax.value_and_grad(forward_trajectory, argnums=0)(nn_params, epsilon)


def run_train_mode(
    simulation,
    args,
    prognostic_keys,
    steps_list,
    nn_params,
    model_batched_unscaled,
    nn_stats,
    nn_model_kwargs,
    loss_function,
    loss_term_names,
    loss_terms_function,
    gpu_name,
    gpu_tag,
    restart_file,
    glorys_file,
    glorys_target_file,
):
    if args.control_mode != "nn_param":
        raise ValueError("--run-mode train currently expects --control-mode nn_param")
    if args.train_updates <= 0:
        raise ValueError("--train-updates must be positive")
    if args.train_save_every <= 0:
        raise ValueError("--train-save-every must be positive")

    train_n_steps = _resolve_train_n_steps(args, steps_list)
    chunk_sizes = make_chunk_sizes(train_n_steps, base=args.chunk_base)
    checkpoint_path = _train_checkpoint_path(args, gpu_tag)
    optimizer = _make_train_optimizer(args)
    params, opt_state, history, start_update = _load_or_init_train_state(
        args,
        checkpoint_path,
        nn_params,
        optimizer,
    )

    print(
        "\nTraining NN parameters with benchmark loss: "
        f"n_steps={train_n_steps}, chunk_sizes={chunk_sizes}, "
        f"sample_steps={args.sample_steps}, updates={args.train_updates}, "
        f"lr={args.train_lr:g}, lr_final={args.train_lr_final}, "
        f"lr_schedule={args.train_lr_schedule}, epsilon={args.epsilon:g}, "
        f"epsilon_final={args.train_epsilon_final}, "
        f"epsilon_schedule={args.train_epsilon_schedule}, "
        f"grad_clip_norm={args.train_grad_clip_norm}"
    )
    print("Training uses the same sampled loss as the benchmark run.")

    if hasattr(simulation, "prepare_ERA5_forcing_window"):
        simulation.prepare_ERA5_forcing_window(simulation.state, train_n_steps)

    ad_train = NNParamsTrainChunkedVJP(
        simulation,
        model_batched_unscaled,
        nn_stats,
        loss_function,
        freeze_vars=args.freeze_vars,
        loss_reduction=args.loss_reduction,
        loss_terms_function=loss_terms_function,
        loss_term_count=len(loss_term_names),
    )

    candidate_validator = getattr(args, "candidate_validator", None)
    lr_backoff = getattr(args, "lr_backoff", 1.0)
    if candidate_validator is not None:
        valid, diagnostic = candidate_validator(params)
        print(f"Initial rollout validation: valid={valid}: {diagnostic}")
        if not valid:
            raise FloatingPointError("Starting checkpoint rollout is invalid under the requested corrections; no optimizer step attempted")

    final_update = start_update + args.train_updates
    if args.balance_loss_terms:
        if not loss_term_names or args.sample_steps > train_n_steps:
            raise ValueError("Loss balancing requires at least one sampled GLORYS loss")
        if args.loss_term_denominators is None:
            print("Calibrating fixed loss denominators from the initial rollout (no optimizer update).")
            initial_epsilon = _train_epsilon_for_update(args, start_update, start_update, final_update)
            baseline_loss, baseline_terms = ad_train.g(
                simulation.state, params, prognostic_keys=prognostic_keys,
                chunk_sizes=chunk_sizes, sample_steps=args.sample_steps,
                epsilon=jnp.asarray(initial_epsilon), return_aux=True, forward_only=True,
            )
            args.loss_term_denominators = _initial_loss_denominators(baseline_terms, loss_term_names)
            del baseline_loss, baseline_terms
        ad_train.loss_term_denominators = jnp.asarray(
            [args.loss_term_denominators[name] for name in loss_term_names]
        )
        print(f"Fixed loss denominators: {_format_named_values(args.loss_term_denominators)}")

    for update_idx in range(start_update, final_update):
        print(f"\n=== train update {update_idx + 1}/{final_update} ===")
        current_epsilon = _train_epsilon_for_update(args, update_idx, start_update, final_update)
        tendency_ratios = tendency_ratio_diagnostics(
            simulation,
            params,
            model_batched_unscaled,
            nn_stats,
            current_epsilon,
        )
        start_time = time.time()
        loss, grads, loss_terms = ad_train.g(
            simulation.state,
            params,
            prognostic_keys=prognostic_keys,
            chunk_sizes=chunk_sizes,
            sample_steps=args.sample_steps,
            epsilon=jnp.asarray(current_epsilon),
            return_aux=True,
        )
        if loss_terms is None:
            jax.block_until_ready((loss, grads))
        else:
            jax.block_until_ready((loss, grads, loss_terms))
        grad_time = time.time() - start_time

        current_lr = _train_lr_for_update(args, update_idx, start_update, final_update)
        if candidate_validator is not None and not _tree_stats(grads)["finite"]:
            raise FloatingPointError("Nonfinite gradient at current parameters; reducing the candidate learning rate cannot repair this gradient")
        raw_updates, candidate_opt_state = optimizer.update(grads, opt_state, params)
        candidate_retries = 0
        candidate_report = None
        if candidate_validator is not None:
            if not _tree_stats(candidate_opt_state)["finite"]:
                raise FloatingPointError("Nonfinite candidate Adam state; previous optimizer state preserved")
            from veros_control.training.stability import backtrack_candidate
            scheduled_lr = current_lr
            params_after_update, updates, current_lr, candidate_retries, candidate_report = backtrack_candidate(
                params, raw_updates, scheduled_lr * lr_backoff, candidate_validator,
                max_retries=getattr(args, "candidate_max_retries", 8),
            )
            lr_backoff *= 0.5 ** candidate_retries
            args.lr_backoff = lr_backoff
        else:
            updates = jax.tree_util.tree_map(
                lambda update: _scale_update(update, current_lr), raw_updates,
            )
            params_after_update = optax.apply_updates(params, updates)
        jax.block_until_ready((updates, params_after_update))

        loss_value = float(loss)
        grad_stats = _tree_stats(grads)
        update_stats = _tree_stats(updates)
        param_stats = _tree_stats(params_after_update)
        selected_grad = _tree_scalar_value(
            grads,
            args.nn_param_leaf_index,
            args.nn_param_flat_index,
        )
        loss_term_values = {}
        if loss_terms is not None:
            loss_term_array = np.asarray(jax.device_get(loss_terms), dtype=np.float64)
            loss_term_values = {
                name: float(loss_term_array[index])
                for index, name in enumerate(loss_term_names)
            }

        normalized_loss_terms = {}
        if args.balance_loss_terms:
            normalized_loss_terms = {
                name: value / args.loss_term_denominators[name]
                for name, value in loss_term_values.items()
            }

        print(
            f"  loss={loss_value:.12e} "
            f"epsilon={current_epsilon:.6e} "
            f"lr={current_lr:.6e} "
            f"grad_norm={grad_stats['norm']:.6e} grad_max={grad_stats['max_abs']:.6e} "
            f"update_norm={update_stats['norm']:.6e} update_max={update_stats['max_abs']:.6e} "
            f"params_norm={param_stats['norm']:.6e} params_max={param_stats['max_abs']:.6e} "
            f"selected_grad={selected_grad:.12e} "
            f"time={grad_time:.4f}s"
        )
        if loss_term_values:
            print(f"  loss_terms={_format_named_values(loss_term_values)}")
        if normalized_loss_terms:
            print(f"  normalized_loss_terms={_format_named_values(normalized_loss_terms)}")
        print(
            "  tendency_ratio_rms(correction/physics)="
            f"{_format_tendency_ratio_field(tendency_ratios, 'rms_ratio')}"
        )
        print(
            "  tendency_ratio_max(correction/physics)="
            f"{_format_tendency_ratio_field(tendency_ratios, 'max_ratio')}"
        )

        bad = []
        if not np.isfinite(loss_value):
            bad.append("loss")
        if not grad_stats["finite"]:
            bad.append("grads")
            print("    bad grad leaves: " + "; ".join(_bad_tree_leaf_summaries(grads)))
        if not update_stats["finite"]:
            bad.append("updates")
            print("    bad update leaves: " + "; ".join(_bad_tree_leaf_summaries(updates)))
        if not param_stats["finite"]:
            bad.append("params_after_update")
            print("    bad param leaves: " + "; ".join(_bad_tree_leaf_summaries(params_after_update)))

        if bad and args.train_fail_on_nonfinite:
            raise FloatingPointError(
                f"Non-finite values during train update {update_idx + 1}: {', '.join(bad)}"
            )

        params = params_after_update
        opt_state = candidate_opt_state
        history.append(
            {
                "update": update_idx + 1,
                "loss": loss_value,
                "lr": current_lr,
                "epsilon": current_epsilon,
                "grad_norm": grad_stats["norm"],
                "grad_max_abs": grad_stats["max_abs"],
                "grad_bad_count": grad_stats["bad_count"],
                "update_norm": update_stats["norm"],
                "update_max_abs": update_stats["max_abs"],
                "param_norm": param_stats["norm"],
                "param_max_abs": param_stats["max_abs"],
                "selected_grad": selected_grad,
                **({"candidate_retries": candidate_retries, "candidate_validation": candidate_report,
                    "lr_backoff": lr_backoff} if candidate_validator is not None else {}),
                "loss_terms": loss_term_values,
                **({"normalized_loss_terms": normalized_loss_terms} if args.balance_loss_terms else {}),
                "tendency_ratio_rms": {
                    name: tendency_ratios[name]["rms_ratio"]
                    for name in ("temp", "salt", "u", "v")
                },
                "tendency_ratio_max": {
                    name: tendency_ratios[name]["max_ratio"]
                    for name in ("temp", "salt", "u", "v")
                },
                "tendency_correction_rms": {
                    name: tendency_ratios[name]["corr_rms"]
                    for name in ("temp", "salt", "u", "v")
                },
                "tendency_physics_rms": {
                    name: tendency_ratios[name]["phys_rms"]
                    for name in ("temp", "salt", "u", "v")
                },
                "time": grad_time,
            }
        )

        if (update_idx + 1) % args.train_save_every == 0 or update_idx + 1 == final_update:
            save_checkpoint(
                checkpoint_path,
                _train_checkpoint_payload(
                    args,
                    params,
                    opt_state,
                    history,
                    update_idx + 1,
                    gpu_name,
                    gpu_tag,
                    restart_file,
                    glorys_file,
                    glorys_target_file,
                    nn_model_kwargs,
                    nn_stats,
                    train_n_steps,
                ),
            )
            print(f"  saved train checkpoint -> {checkpoint_path}")

        del loss, grads, loss_terms, raw_updates, updates, params_after_update
        gc.collect()
        print(f"  memory after update: live_arrays={len(jax.live_arrays())}")

    print(f"\nTrain mode complete. Checkpoint: {checkpoint_path}")
    return checkpoint_path, history


def main():
    args = parse_args()
    if args.basin_mean_loss and (args.loss_mode != "glorys_target" or any(k not in ('temp', 'salt') for k in args.keys_loss)):
        raise ValueError('--basin-mean-loss requires GLORYS targets and temp/salt loss keys')
    if args.balance_loss_terms and (args.run_mode != "train" or args.loss_mode != "glorys_target"):
        raise ValueError("--balance-loss-terms requires --run-mode train --loss-mode glorys_target")
    if args.balance_loss_terms and (args.sample_steps <= 0 or len(set(args.keys_loss)) != len(args.keys_loss)):
        raise ValueError("Loss balancing requires positive --sample-steps and unique --keys-loss")
    if args.run_mode == "train":
        if args.control_mode != "nn_param":
            raise ValueError("--run-mode train currently expects --control-mode nn_param")
        if args.plot_only:
            raise ValueError("--plot-only is only available for --run-mode benchmark")
        if args.method_pair != "vjp-only":
            print("Train mode uses VJP gradients; setting method_pair to vjp-only.")
            args.method_pair = "vjp-only"
    if args.control_mode == "nn_param_vector" and args.method_pair not in ("jvp-vjp", "vjp-only"):
        raise ValueError("--control-mode nn_param_vector supports --method-pair jvp-vjp or vjp-only")
    if args.method_pair == "vjp-only" and args.control_mode not in ("nn_param", "nn_param_vector"):
        raise ValueError("--method-pair vjp-only is currently implemented for --control-mode nn_param or nn_param_vector")
    if args.method_pair == "vjp-only" and args.save_sample_trajectories:
        print("Ignoring --save-sample-trajectories for --method-pair vjp-only.")
        args.save_sample_trajectories = False

    steps_list = make_steps_list(args)
    args.glorys_interpolation_alpha = _glorys_interpolation_alpha(args, steps_list)

    initialization_mode = (
        "glorys"
        if args.glorys_file is not None
        else "restart"
        if args.restart_file is not None
        else "setup"
    )
    output_mode_suffix = "_glorys" if initialization_mode == "glorys" else ""
    output_loss_suffix = "_target" if args.loss_mode == "glorys_target" else ""
    if args.glorys_linear_interpolation:
        interpolation_steps = round(args.glorys_interpolation_alpha * 720)
        output_loss_suffix += f"_lininterp{interpolation_steps}of720"
    if args.basin_mean_loss:
        output_loss_suffix += f"_basin_top{args.basin_levels}"
    if args.balance_loss_terms:
        output_loss_suffix += "_balanced"
    output_reduction_suffix = "_mean" if args.loss_reduction == "mean" else ""
    if args.output_folder is None:
        args.output_folder = (
            f"output_folder_{args.control_mode}{output_mode_suffix}"
            f"{output_loss_suffix}{output_reduction_suffix}"
        )
    if args.output_prefix is None:
        args.output_prefix = (
            f"ad_comparison_{args.control_mode}{output_mode_suffix}"
            f"{output_loss_suffix}{output_reduction_suffix}"
        )
    if args.trajectory_folder is None:
        args.trajectory_folder = os.path.join(args.output_folder, "sample_trajectories")
    os.makedirs(args.output_folder, exist_ok=True)

    gpu_name = get_gpu_name()
    gpu_tag = get_gpu_tag()

    _print_precision()
    print(f"Running on device: {gpu_name}")
    print(f"GPU tag for output files: {gpu_tag}")
    print(f"n_steps sweep: {steps_list}")
    print(f"sample_steps: {args.sample_steps}")
    print(f"chunk_base: {args.chunk_base}")
    print(f"control_mode: {args.control_mode}")
    print(f"run_mode: {args.run_mode}")
    print(f"method_pair: {args.method_pair}")
    print(f"loss_mode: {args.loss_mode}")
    print(f"loss_reduction: {args.loss_reduction}")
    print(f"veros_diskless_mode: {getattr(runtime_settings, 'diskless_mode', 'n/a')}")
    print(f"use_forcings: {getattr(args, 'use_forcings', False)}")
    if args.loss_mode == "glorys_target":
        print(f"keys_loss: {args.keys_loss}")
        if args.glorys_linear_interpolation:
            print(f"GLORYS linear interpolation: alpha={args.glorys_interpolation_alpha:g} (n_steps/720)")
    print(f"initialization_mode: {initialization_mode}")
    print(f"freeze_vars_in_chunked_vjp: {args.freeze_vars}")
    print(f"save_sample_trajectories: {args.save_sample_trajectories}")
    if args.save_sample_trajectories:
        print(f"trajectory_folder: {args.trajectory_folder}")

    if args.control_mode == "A_h":
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
        for control_value_float, results, _ in _load_existing_results(args, gpu_tag, control_values):
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
        return

    restart_file = resolve_restart_file(args.restart_file, args.restart_index)
    glorys_file = resolve_input_file(args.glorys_file, args.glorys_index, "GLORYS initial")
    glorys_target_file = _resolve_glorys_target_file(args)
    if restart_file is not None and glorys_file is not None:
        raise ValueError("Use either --restart-file or --glorys-file, not both")

    print(f"restart_file: {restart_file or 'none (setup initial condition)'}")
    print(f"glorys_file: {glorys_file or 'none'}")
    print(f"glorys_target_file: {glorys_target_file or 'none'}")
    if glorys_file is not None:
        print(
            "glorys initialization indices: "
            f"time_index={args.glorys_time_index}, "
            f"month_index={args.glorys_month_index}, "
            f"start_month={args.glorys_start_month}"
        )

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
        print("Using restart/prognostic variables in the chunked VJP carry.")
    print(prognostic_keys)

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

    loss_function, loss_term_names, loss_terms_function = make_root_loss_functions(
        args,
        simulation,
        target_file=glorys_target_file,
        initial_file=glorys_file,
    )

    if args.run_mode == "train":
        run_train_mode(
            simulation,
            args,
            prognostic_keys,
            steps_list,
            nn_params,
            model_batched_unscaled,
            nn_stats,
            nn_model_kwargs,
            loss_function,
            loss_term_names,
            loss_terms_function,
            gpu_name,
            gpu_tag,
            restart_file,
            glorys_file,
            glorys_target_file,
        )
        return

    for control_value_float in control_values:
        if args.clear_between_epsilons:
            print("\nClearing JAX caches before next epsilon...")
            jax.clear_caches()
            gc.collect()

        if args.control_mode == "nn_epsilon":
            print(f"\n### epsilon = {control_value_float:g} ###")

        control_value = jnp.asarray(control_value_float)
        pkl_path = result_pickle_path(args, gpu_tag, control_value_float)
        existing_results = None
        if os.path.exists(pkl_path):
            print(f"Found existing results file: {pkl_path}")
            with open(pkl_path, "rb") as f:
                existing_results = pickle.load(f)
            print(f"  -> resuming; already have results for steps: {existing_results.get('steps', [])}")
        else:
            print(f"No existing results file for this GPU/VRAM combo -> starting fresh: {pkl_path}")

        trajectory_prefix = f"{args.output_prefix}{result_suffix_for(args, control_value_float)}_{gpu_tag}"

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
        results["nn_checkpoint"] = args.nn_checkpoint
        results["nn_model_kwargs"] = nn_model_kwargs
        results["use_forcings"] = args.use_forcings
        results["nn_param"] = nn_param_description
        results["nn_param_vector"] = selected_params
        results["nn_param_count"] = args.nn_param_count
        results["nn_param_seed"] = args.nn_param_seed if args.nn_param_seed is not None else args.seed
        results["nn_param_path_filter"] = args.nn_param_path_filter
        results["sample_steps"] = args.sample_steps
        results["restart_file"] = restart_file
        results["glorys_file"] = glorys_file
        results["glorys_target_file"] = glorys_target_file
        results["glorys_time_index"] = args.glorys_time_index
        results["glorys_month_index"] = args.glorys_month_index
        results["glorys_start_month"] = args.glorys_start_month
        results["glorys_linear_interpolation"] = args.glorys_linear_interpolation
        results["glorys_interpolation_alpha"] = args.glorys_interpolation_alpha
        results["loss_mode"] = args.loss_mode
        results["loss_reduction"] = args.loss_reduction
        results["keys_loss"] = list(args.keys_loss)
        results["basin_mean_loss"] = args.basin_mean_loss
        results["basin_levels"] = args.basin_levels if args.basin_mean_loss else None
        results["basin_loss_metadata"] = getattr(args, "basin_loss_metadata", None)
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
            print_summary(results, grad_err_pct, loss_err_pct)


if __name__ == "__main__":
    main()

