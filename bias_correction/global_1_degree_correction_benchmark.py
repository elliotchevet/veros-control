import sys
from pathlib import Path
import os
import re
import time
from operator import attrgetter

# Configure environment
PRP = str(Path.cwd().resolve().parents[0]) + '/'
print(PRP)
sys.path.extend([PRP, PRP + 'utils/'])
sys.path.append("/scratch/so2495/code/OceanHybridModels/VerosLearning/veros/")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Use full GPU memory instead of JAX's default 75% cap ---
# These MUST be set before jax (or anything that imports jax, like veros) is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Run bias correction training")
    parser.add_argument('--baseline_model', action='store_true', help='no nn correction, run the biased model')
    parser.add_argument('--ref_model', action='store_true', help='run the reference model (TKE based)')
    parser.add_argument('--eval_files', action='store_true', help='only runs the diagnostic nc files')
    parser.add_argument('--hidden_dim_nn', type=int, default=128)
    parser.add_argument('--depth_nn', type=int, default=3)
    parser.add_argument('--nb_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nb_days_training', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_epsilon', action='store_true')
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--viz_mlp_freq', type=float, default=60*60*24*365)
    parser.add_argument('--normalize_inputs', action='store_true')
    parser.add_argument('--normalize_loss', action='store_true')
    parser.add_argument('--scaling_method_features', type=str, default='minmax')
    parser.add_argument('--scaling_method_loss', type=str, default='minmax')
    parser.add_argument('--avg_file', type=str, default="/scratch/so2495/data/numerical_simulations/veros_output/acc_quarter_pressure_formulation_gt/acc_quarter_pressure_formulation_gt.averages.nc")
    parser.add_argument('--use-lon-lat', action='store_true', 
                       help='use longitude and latitude as inputs of the MLP')
    parser.add_argument('--conservative-mixing', action='store_true', 
                       help='constraint mixing conservation in the vertical')
    parser.add_argument('--use_one_hot', action='store_true', 
                       help='use one hot encoding for features of MLP when multiple features per mlp')
    parser.add_argument('--seed', type=int, default=0, 
                       help='Random seed for initialization')

    parser.add_argument(
        '--num_mlps',
        type=int,
        choices=[1, 2, 4],
        default=1,
        help='Number of MLPs to use: 1, 2, or 4'
    )  
    parser.add_argument(
        "--loss_type",
        type=str,
        default="zonal_mean",
        choices=["zonal_mean", "pointwise"],
        help="Type of loss function to use. Options: 'zonal_mean' or 'pointwise'. Default: 'pointwise'."
    )    
    parser.add_argument("--TS", action="store_true",
                        help="If set, use only T,S as inputs but still predict T,S,U,V")    
                        
    parser.add_argument("--reweight_loss", action="store_true",
                        help="If set, use only T,S as inputs but still predict T,S,U,V")
    parser.add_argument('--folder', type=str, default='results/')
    parser.add_argument('--keys_loss', nargs='+', type=str, default=['temp', 'salt', 'ssh'], help='List of variable names to include in the loss computation')
    # Resume training parameters
    parser.add_argument('--resume_training', action='store_true', 
                       help='Resume training from existing checkpoint')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None,
                       help='Specific path to checkpoint file to resume from')
    parser.add_argument('--overwrite_training', action='store_true',
                       help='Overwrite existing training (start fresh)')
    parser.add_argument('--suffix', type=str, default='', 
                            help='Optional suffix for output file/folder names')                       
    
    return parser.parse_args()

# Set Veros runtime settings
from veros import runtime_settings
setattr(runtime_settings, 'backend', 'jax')
setattr(runtime_settings, 'device', 'gpu')
setattr(runtime_settings, 'force_overwrite', False)
setattr(runtime_settings, 'linear_solver', 'scipy_jax')

import pickle
import glob
import numpy as np
import gc
import jax
import jax.numpy as jnp
from jax import vjp, vmap, jit
from functools import partial
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import optax
import matplotlib.pyplot as plt
from veros import restart
from veros.core import utilities
from veros.core.operators import update, update_add, at
from veros.io_tools import hdf5 as h5tools
from utils_train import Config, init_simulation_ssh
from tqdm import tqdm
# Scale all weights and biases in the parameter tree
from flax import traverse_util
from flax.core import freeze, unfreeze
# Define MLP model
from veros.state import veros_state_pytree_flatten, veros_variables_pytree_flatten
from veros.state import veros_state_pytree_unflatten, veros_variables_pytree_unflatten


def get_gpu_name():
    """
    Return a human-readable name of the (first) accelerator JAX is running on,
    e.g. 'NVIDIA A100-SXM4-80GB'. Falls back gracefully to CPU/unknown.
    """
    try:
        devices = jax.devices()
        if not devices:
            return "unknown (no JAX devices found)"
        dev = devices[0]
        # device_kind is the most reliable cross-platform field (e.g. GPU model name)
        name = getattr(dev, "device_kind", None) or str(dev)
        platform = getattr(dev, "platform", "unknown")
        return f"{name} ({platform}, {len(devices)} device(s) visible)"
    except Exception as e:
        return f"unknown (error querying JAX devices: {e})"


def get_gpu_tag():
    """
    Build a filesystem-safe tag encoding GPU model + VRAM size, e.g.
    'NVIDIA_A100-SXM4-80GB_80GB'. Used to key the output pkl file so that
    re-running on the same GPU/VRAM combo can find and resume it, while a
    different GPU/VRAM combo gets its own separate results file.
    """
    try:
        devices = jax.devices()
        if not devices:
            return "unknown_gpu"
        dev = devices[0]
        raw_name = getattr(dev, "device_kind", None) or str(dev)

        vram_gb = None
        try:
            stats = dev.memory_stats()
            if stats and "bytes_limit" in stats and stats["bytes_limit"]:
                vram_gb = round(stats["bytes_limit"] / (1024 ** 3))
        except Exception:
            pass

        # Sanitize: keep alnum, dash, underscore; collapse everything else to '_'
        safe_name = re.sub(r'[^A-Za-z0-9\-]+', '_', raw_name).strip('_')

        if vram_gb is not None:
            return f"{safe_name}_{vram_gb}GB"
        return safe_name
    except Exception as e:
        return f"unknown_gpu_error_{type(e).__name__}"

 
def make_chunk_sizes(iterations, base=3):
    """
    Factor `iterations` into a tuple of `base`-sized chunks for
    vjp_grad_scanned_Nlvls (product of chunk_sizes must equal iterations).
    Falls back to a single chunk of size `iterations` if it isn't a
    clean power of `base`.
    """
    n = iterations
    chunks = []
    while n > 1:
        if n % base == 0:
            chunks.append(base)
            n //= base
        else:
            # not divisible cleanly -> just use remaining n as one chunk
            chunks.append(n)
            n = 1
    if not chunks:
        chunks = [iterations]
    return tuple(chunks)
 
import jax

def free_buffers(outputs):
    """Recursively free JAX arrays within a PyTree to release memory to the XLA pool."""
    jax.tree_util.tree_map(lambda x: x.delete() if hasattr(x, 'delete') else None, outputs)


def save_results(results, pkl_path):
    """Persist results dict to disk. Called after every step so progress
    survives a crash/preemption, not just at the very end of the run."""
    tmp_path = pkl_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp_path, pkl_path)  # atomic on POSIX, avoids truncated files
    print(f"  [checkpoint] saved progress -> {pkl_path}")


def run_comparison(simulation, prognostic_keys, step_function, agg_function,
                   var_name, var_value, steps_list, results=None, pkl_path=None):
    if results is None:
        results = {
            "steps": [],
            "jvp_grad": [], "jvp_loss": [], "jvp_time": [],
            "vjp_grad": [], "vjp_loss": [], "vjp_time": [],
        }

    already_done = set(results["steps"])

    for iteration_grad in steps_list:
        if iteration_grad in already_done:
            print(f"\n=== steps = {iteration_grad} : already in results, skipping (resume) ===")
            continue

        chunk_sizes = make_chunk_sizes(iteration_grad, base=3)
        print(f"\n=== steps = {iteration_grad} | chunk_sizes = {chunk_sizes} ===")

        # --- Check memory baseline before we do anything ---
        stats = jax.devices()[0].memory_stats()
        print(f"  [Baseline] bytes_in_use={stats.get('bytes_in_use', 0):,}  peak={stats.get('peak_bytes_in_use', 0):,}  live_arrays={len(jax.live_arrays())}")

        # ==========================================
        # 1. JVP (Scanned)
        # ==========================================
        jax.clear_caches()
        gc.collect()
        
        ad_jvp = jvp_grad_scan(step_function, agg_function, var_name)
        
        # --- Warmup (Compilation) ---
        print("  [JVP] Compiling...")
        warmup_jvp = ad_jvp.g(simulation.state, var_value=var_value, iterations=iteration_grad)
        jax.block_until_ready(warmup_jvp)
        free_buffers(warmup_jvp)
        
        # --- Execution (Timing) ---
        print("  [JVP] Running...")
        start_time = time.time()
        n_diff_jvp = ad_jvp.g(simulation.state, var_value=var_value, iterations=iteration_grad)
        jax.block_until_ready(n_diff_jvp)
        elapsed_jvp = time.time() - start_time
        
        loss_jvp = float(n_diff_jvp[0])
        grad_jvp = float(jnp.mean(n_diff_jvp[1]))
        print(f"  -> jvp_diff_scanned : grad={grad_jvp:.6e} loss={loss_jvp:.6e} time={elapsed_jvp:.4f}s")
        
        # --- Cleanup ---
        free_buffers(n_diff_jvp)
        
        # Explicitly delete variable bindings that could hold closures/arrays
        del ad_jvp, warmup_jvp, n_diff_jvp
        jax.clear_caches()
        gc.collect()

        # --- Check memory after JVP cleanup ---
        stats = jax.devices()[0].memory_stats()
        print(f"  [Post-JVP] bytes_in_use={stats.get('bytes_in_use', 0):,}  peak={stats.get('peak_bytes_in_use', 0):,}  live_arrays={len(jax.live_arrays())}")

        # ==========================================
        # 2. VJP (Chunked, N-levels)
        # ==========================================
        ad_vjp = vjp_grad_scanned_Nlvls(step_function, agg_function, var_name)
        
        # --- Warmup (Compilation) ---
        print("  [VJP] Compiling...")
        warmup_vjp = ad_vjp.g(
            simulation.state,
            var_value=var_value,
            prognostic_keys=prognostic_keys,
            iterations=iteration_grad,
            chunk_sizes=chunk_sizes,
        )
        jax.block_until_ready(warmup_vjp)
        free_buffers(warmup_vjp)
        
        # --- Execution (Timing) ---
        print("  [VJP] Running...")
        start_time = time.time()
        n_diff_vjp = ad_vjp.g(
            simulation.state,
            var_value=var_value,
            prognostic_keys=prognostic_keys,
            iterations=iteration_grad,
            chunk_sizes=chunk_sizes,
        )
        jax.block_until_ready(n_diff_vjp)
        elapsed_vjp = time.time() - start_time
        
        loss_vjp = float(n_diff_vjp[0])
        grad_vjp = float(jnp.mean(n_diff_vjp[1]))
        print(f"  -> vjp_grad_chunked_Nlvls : grad={grad_vjp:.6e} loss={loss_vjp:.6e} time={elapsed_vjp:.4f}s")
        
        # --- Cleanup ---
        free_buffers(n_diff_vjp)

        # Explicitly delete variable bindings that could hold closures/arrays
        del ad_vjp, warmup_vjp, n_diff_vjp
        jax.clear_caches()
        gc.collect()

        # --- Check memory after VJP cleanup ---
        stats = jax.devices()[0].memory_stats()
        print(f"  [Post-VJP] bytes_in_use={stats.get('bytes_in_use', 0):,}  peak={stats.get('peak_bytes_in_use', 0):,}  live_arrays={len(jax.live_arrays())}")

        # ==========================================
        # 3. Store Results
        # ==========================================
        results["steps"].append(iteration_grad)
        results["jvp_grad"].append(grad_jvp)
        results["jvp_loss"].append(loss_jvp)
        results["jvp_time"].append(elapsed_jvp)
        results["vjp_grad"].append(grad_vjp)
        results["vjp_loss"].append(loss_vjp)
        results["vjp_time"].append(elapsed_vjp)
        already_done.add(iteration_grad)

        # --- Incremental checkpoint: survive a crash mid-run ---
        if pkl_path is not None:
            save_results(results, pkl_path)

    return results
 
def plot_results(results, gpu_name="unknown", save_path="ad_comparison.png"):
    steps = np.array(results["steps"])
    jvp_grad = np.array(results["jvp_grad"])
    vjp_grad = np.array(results["vjp_grad"])
    jvp_loss = np.array(results["jvp_loss"])
    vjp_loss = np.array(results["vjp_loss"])
 
    # Errors expressed as a percentage of the JVP value (reference)
    grad_err_pct = np.abs(vjp_grad - jvp_grad) / np.abs(jvp_grad) * 100.0
    loss_err_pct = np.abs(vjp_loss - jvp_loss) / np.abs(jvp_loss) * 100.0
 
    jvp_time = np.array(results["jvp_time"])
    vjp_time = np.array(results["vjp_time"])
 
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"AD method comparison — GPU: {gpu_name}", fontsize=13)
 
    # --- Panel 1: gradient value vs steps, both methods ---
    ax = axes[0, 0]
    ax.plot(steps, jvp_grad, marker="o", label="JVP (scanned)")
    ax.plot(steps, vjp_grad, marker="s", label="VJP (chunked, N-levels)")
    ax.set_xscale("log", base=3)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Gradient")
    ax.set_title("Gradient vs number of steps")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
 
    # --- Panel 2: gradient error (%) vs steps ---
    ax = axes[0, 1]
    ax.plot(steps, grad_err_pct, marker="o", color="firebrick")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Gradient error (% of JVP)")
    ax.set_title("Gradient error between methods")
    ax.grid(True, which="both", alpha=0.3)
 
    # --- Panel 3: loss value vs steps, both methods ---
    ax = axes[1, 0]
    ax.plot(steps, jvp_loss, marker="o", label="JVP (scanned)")
    ax.plot(steps, vjp_loss, marker="s", label="VJP (chunked, N-levels)")
    ax.set_xscale("log", base=3)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs number of steps")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
 
    # --- Panel 4: loss error (%) vs steps ---
    ax = axes[1, 1]
    ax.plot(steps, loss_err_pct, marker="o", color="firebrick")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Loss error (% of JVP)")
    ax.set_title("Loss error between methods")
    ax.grid(True, which="both", alpha=0.3)
 
    # --- Panel 5: compute time vs steps, both methods ---
    ax = axes[2, 0]
    ax.plot(steps, jvp_time, marker="o", label="JVP (scanned)")
    ax.plot(steps, vjp_time, marker="s", label="VJP (chunked, N-levels)")
    ax.set_xscale("log", base=3)
    ax.set_yscale("log")
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Compute time (s)")
    ax.set_title("Compute time vs number of steps")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
 
    # Unused 6th slot in the 3x2 grid
    axes[2, 1].axis("off")
 
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    print(f"\nSaved figure to {save_path}")
    plt.show()
 
    return grad_err_pct, loss_err_pct
 
 
 

def pure(state, step):
    n_state = state.copy()
    step(n_state)  # This is a function that modifies state object inplace    
    return n_state


def loss_function_unmasked(state, gt, iter, tau, keys, stats=None, normalize=False, scaling_method='meanstd'):
    error = 0.0
    for key in keys:
        data = getattr(state.variables, key)
        if key == 'ssh':
            agg_pred = (data[2:-2, 2:-2])
            agg_true = (gt[iter][key][2:-2, 2:-2])     
        else:
            agg_pred = jnp.mean(data[2:-2, 2:-2, :, tau], axis=0)
            agg_true = jnp.mean(gt[iter][key][2:-2, 2:-2, :, tau], axis=0)

        if normalize and stats is not None:
            if scaling_method == 'meanstd':
                mean, std = stats
                denom = jnp.where(std[key] < 1e-6, 1.0, std[key])
                
                agg_pred = (agg_pred - mean[key]) / denom
                agg_true = (agg_true - mean[key]) / denom

            elif scaling_method == 'minmax':
                min_val, max_val = stats
                denom = jnp.where((max_val[key] - min_val[key]) < 1e-6, 1.0, max_val[key] - min_val[key])

                agg_pred = (agg_pred - min_val[key]) / denom
                agg_true = (agg_true - min_val[key]) / denom

            else:
                raise ValueError(f"Unsupported scaling_method: {scaling_method}")

        error += jnp.mean((agg_pred - agg_true) ** 2)
    return error

def loss_function(state, gt, iter, tau, keys, stats=None, normalize=False, scaling_method='meanstd', loss_type = 'zonal_mean', reweight_loss = False):
    error = 0.0
    mask = getattr(state.variables, "maskT")  # shape: (252, 328, 15), dtype: bool
    for key in keys:
        data = getattr(state.variables, key)

        if key == 'ssh':
            # SSH is 2D: (252, 328), use mask at last level
            pred_crop = data[2:-2, 2:-2]                      # shape: (248, 324)
            true_crop = gt[iter][key][2:-2, 2:-2]
            mask_crop = mask[2:-2, 2:-2, -1]                  # shape: (248, 324), bool
            
            # Apply mask
            agg_pred = pred_crop[mask_crop]
            agg_true = true_crop[mask_crop]

            if normalize and stats is not None:
                
                if scaling_method == 'meanstd':
                    mean, std_or_range = stats
                    denom = jnp.where(std_or_range[key] < 1e-6, 1.0, std_or_range[key])
                    mean_crop = mean[key][mask_crop]
                    denom_crop = denom[mask_crop]
                elif scaling_method == 'minmax':
                    min_val, max_val = stats  # reusing variable names
                    denom = jnp.where((max_val[key] - min_val[key]) < 1e-6, 1.0, max_val[key] - min_val[key])
                    mean_crop = min_val[key][mask_crop]
                    denom_crop = denom[mask_crop]
                else:
                    raise ValueError(f"Unsupported scaling_method: {scaling_method}")

                agg_pred = (agg_pred - mean_crop) / denom_crop
                agg_true = (agg_true - mean_crop) / denom_crop

        else:
            # Tracers: data shape (252, 328, 15)
            if loss_type == 'zonal_mean':
                agg_pred = jnp.mean(data[2:-2, 2:-2, :, tau], axis=0)
                agg_true = jnp.mean(gt[iter][key][2:-2, 2:-2, :, tau], axis=0)
            elif loss_type == 'pointwise':
                print("loss is pointwise")
                agg_pred = (data[2:-2, 2:-2, :, tau])
                agg_true = (gt[iter][key][2:-2, 2:-2, :, tau])
            
            mask_crop = mask[2:-2, 2:-2, :]                   # same shape, bool
            if loss_type == 'zonal_mean':
                mask_avg = jnp.any(mask_crop, axis=0)
            elif loss_type == 'pointwise':
                mask_avg = mask_crop
                
            
            agg_pred = agg_pred[mask_avg]                  # shape: (N,)
            agg_true = agg_true[mask_avg]

            if normalize and stats is not None:
                if scaling_method == 'meanstd':
                    mean, std_or_range = stats
                    denom = jnp.where(std_or_range[key] < 1e-6, 1.0, std_or_range[key])
                    mean_crop = mean[key][mask_avg]
                    denom_crop = denom[mask_avg]
                elif scaling_method == 'minmax':
                    min_val, max_val = stats
                    denom = jnp.where((max_val[key] - min_val[key]) < 1e-6, 1.0, max_val[key] - min_val[key])
                    mean_crop = min_val[key][mask_avg]
                    denom_crop = denom[mask_avg]
                else:
                    raise ValueError(f"Unsupported scaling_method: {scaling_method}")

                agg_pred = (agg_pred - mean_crop) / denom_crop
                agg_true = (agg_true - mean_crop) / denom_crop

        # Mean Squared Error over valid points
        squared_error = (agg_pred - agg_true) ** 2
        loss_val = jnp.mean(squared_error)
        if reweight_loss:
            # Coefficients derived from inverse of logged losses to normalize magnitude to ~1.0
            # temp: 1/0.002345 ≈ 426.3
            # salt: 1/3.065e-05 ≈ 32622
            # ssh:  1/2.132e-06 ≈ 468893
            
            weight = 1.0
            if key == 'temp':
                weight = 1#426.3
            elif key == 'salt':
                weight = 76.5#32622.0
            elif key == 'ssh':
                weight = 1099.0#468893.0
            
            loss_val = loss_val * weight
            jax.debug.print(f"loss_type: {loss_type} | Key: {key} | Weighted Loss: {{x}}", x=loss_val)
        
        error += loss_val
    return error        

def safe_add(x, y):
    is_float0_x = getattr(x, 'dtype', None) == jax.dtypes.float0
    is_float0_y = getattr(y, 'dtype', None) == jax.dtypes.float0

    if is_float0_x and is_float0_y:
        print('both x, y, : ', x, x.shape,  y, y.shape)
        return jnp.zeros([], dtype=jnp.float32)  # both are float0 → return scalar zero
    elif is_float0_x:
        print('only x, : ', x, x.shape)
        return y
    elif is_float0_y:
        print('only y, : ', y, y.shape)
        return x
    else:
        return x + y
    
def safe_add_with_path(path, x, y):
    is_float0_x = getattr(x, 'dtype', None) == jax.dtypes.float0
    is_float0_y = getattr(y, 'dtype', None) == jax.dtypes.float0
    name = "/".join(str(p) for p in path)

    if is_float0_x and is_float0_y:
        print(f"[{name}] both x and y are float0 — skipping")
        return x  # return either
    elif is_float0_x:
        print(f"[{name}] x is float0 — returning y")
        return y
    elif is_float0_y:
        print(f"[{name}] y is float0 — returning x")
        return x
    else:
        return x + y    

def run_vjp_correction(state_init, params, model, step_func, step_function_inplace,
                                                 loss_func, n_iter, sample_steps, var_meta, keys):
    sampled_indices = []
    vjp_funcs_loss = []
    losses = []

    itr_days = 0
    state = state_init.copy()

    # === FORWARD PASS ===
    for iters in range(n_iter):
        state = step_func(state)
        state = model(state, params)
        with state.variables.unlock():
            state.variables.u = utilities.enforce_boundaries(state.variables.u, state.settings.enable_cyclic_x)
            state.variables.v = utilities.enforce_boundaries(state.variables.v, state.settings.enable_cyclic_x)
            
        if jnp.isnan(state.variables.temp).any():
            print("NaNs found in temp at iteration", iters)

        if (iters + 1) % sample_steps == 0:
            loss_partial = partial(loss_func, iter=itr_days, tau=state.variables.tau, keys=keys)
            loss_val, vjp_loss = jax.vjp(loss_partial, state)
            losses.append(loss_val)
            vjp_funcs_loss.append(vjp_loss)
            sampled_indices.append(iters)
            itr_days += 1

    total_loss = jnp.mean(jnp.stack(losses))
    dl_dvar_total = jax.tree_map(jnp.zeros_like, params)
    dl_dxn = None

    # === SINGLE BACKWARD PASS ===
    for iters in range(n_iter - 1, -1, -1):
        # Check if this is a sampled step to inject gradient from the loss
        if iters in sampled_indices:
            idx = sampled_indices.index(iters)
            dl_dloss, = vjp_funcs_loss[idx](1.0)
            if dl_dxn is None:
                dl_dxn = dl_dloss
            else:
                dl_dxn = jax.tree_util.tree_map_with_path(safe_add_with_path, dl_dxn, dl_dloss)
                #dl_dxn = jax.tree_map(safe_add_with_path, dl_dxn, dl_dloss)

        # Skip backprop if no gradient to propagate yet
        if dl_dxn is None:
            continue

        # === Recompute forward state at iteration `iters - 1` ===
        state_nn = state_init.copy()
        for k in range(iters):
            state_tmp = step_func(state_nn)
            state_nn = model(state_tmp, params)
            with state.variables.unlock():
                state.variables.u = utilities.enforce_boundaries(state.variables.u, state.settings.enable_cyclic_x)
                state.variables.v = utilities.enforce_boundaries(state.variables.v, state.settings.enable_cyclic_x)
        state_k = step_func(state_nn)  # final state at `iters`

        # === Backprop through model and step_func ===
        _, vjp_model = jax.vjp(model, state_k, params)
        dl_dxhn, dl_dtheta = vjp_model(dl_dxn)

        _, vjp_step = jax.vjp(step_func, state_nn)
        dl_dxn, = vjp_step(dl_dxhn)

        # Accumulate gradient wrt parameters
        dl_dvar_total = jax.tree_util.tree_map(jnp.add, dl_dvar_total, dl_dtheta)

    # Normalize by number of samples
    dl_dvar_total = jax.tree_map(lambda x: x / len(sampled_indices), dl_dvar_total)
    return total_loss, dl_dvar_total


def compute_loss_and_grad(simulation, gt_files, params, model_apply_batched_ckpt, step_function_ckpt, loss_func, n_timesteps, sample_steps, var_meta, keys):
    return run_vjp_correction(
        simulation.state, 
        params,
        model_apply_batched_ckpt,
        step_function_ckpt, 
        simulation.step,
        loss_func, 
        n_iter=n_timesteps,#n_timesteps,
        sample_steps=sample_steps,#sample_steps,
        var_meta=var_meta,
        keys=keys
    )


def forward_unroll(params, state, model, step_func, loss_func, n_iter, sample_steps, keys):
    n_samples = n_iter // sample_steps
    losses = jnp.zeros((n_samples,))
    itr_days = 0

    for i in range(n_iter):
        state = step_func(state)
        state = model(state, params)
        
        with state.variables.unlock():
            state.variables.u = utilities.enforce_boundaries(state.variables.u, state.settings.enable_cyclic_x)
            state.variables.v = utilities.enforce_boundaries(state.variables.v, state.settings.enable_cyclic_x)
        
        if (i + 1) % sample_steps == 0:
            loss_val = loss_func(
                state,
                iter=itr_days,
                tau=state.variables.tau,
                keys=keys
            )
            losses = losses.at[itr_days].set(loss_val)
            itr_days = itr_days+1

    total_loss = jnp.mean(losses)
    return total_loss

def run_vjp_correction_builtin(params, state_init, forward_checkpoint):
    loss_fn = value_and_grad(forward_checkpoint)
    total_loss, dl_dvar_total = loss_fn(params,state_init)
    return total_loss, dl_dvar_total

def compute_loss_and_grad_builtin(simulation, params, model_apply_batched_ckpt, step_function_ckpt, loss_func, n_timesteps, sample_steps, keys):
    forward_checkpoint = jax.checkpoint(partial(
    forward_unroll,
    model=model_apply_batched_ckpt,
    step_func=step_function_ckpt,
    loss_func=loss_func,
    n_iter=n_timesteps,
    sample_steps=sample_steps,
    keys = keys), policy=jax.checkpoint_policies.nothing_saveable)
    return run_vjp_correction_builtin(params, simulation.state, forward_checkpoint)

# Initialize training
def create_optimizer(lr=1e-6):
    return optax.chain(
        #optax.clip_by_global_norm(1.0),
        optax.adamw(lr)
    )

def scale_params(params, epsilon):
    params = unfreeze(params)
    scaled = {k: v * epsilon for k, v in flatten_dict(params).items()}
    return freeze(unflatten_dict(scaled))

def load_checkpoint(output_file):
    """Load training checkpoint if exists, return None otherwise"""
    if os.path.exists(output_file):
        print(f"Loading checkpoint from {output_file}")
        with open(output_file, 'rb') as f:
            return pickle.load(f)
    return None

def agg_sum(state, key_sum='u'):
    return ((getattr(state.variables, key_sum)[:, :, :] - 0) ** 2).mean()


class autodiff() :
    def __init__(self, step_function, agg_function,  var_name) :
        """
            Computes derivative dL/dvar with L in R and var in R
            step_function is the function done n iterations
            agg_function is computed at the end to go from R^space -> R
            var_name : name of the variable in the state to differentiatiat w.r.t
        """
        self.agg_function = agg_function
        self.step_function = partial(autodiff.pure, step=step_function)
        self.var_name = var_name

    @staticmethod
    def pure(state, step) :
        """
            Convert the state function into a "pure step" copying the input state
        """
        n_state = state.copy()
        step(n_state)  # This is a function that modifies state object inplace
        return n_state

    @staticmethod
    def set_var(var_name, state, var_value):
        n_state = state.copy()
        vs = n_state.variables
        with n_state.variables.unlock():
            setattr(vs, var_name, var_value)
        return n_state


    @staticmethod
    def wrapper(var_value, state, step_fun, var_name, agg_func, iter):
        n_state = autodiff.set_var(var_name, state, var_value)

        for i in range(iter) :
            n_state = step_fun(n_state)

        return agg_func(n_state)

class vjp_grad(autodiff) :

    def g(self, state, var_value, iterations=1, **kwargs) :

        n_state = autodiff.set_var(self.var_name, state, var_value)

        # Forward pass through all iterations
        current_state = n_state
        funs = []
        for i in range(iterations):
            #print(f'a {i}')
            current_state, vjp_fun  = jax.vjp(self.step_function, current_state)
            funs.append(vjp_fun)

        # Compute final output
        l, vjp_agg = jax.vjp(self.agg_function, current_state)

        # Backward pass using VJP
        cotangent = jnp.ones_like(l)

        # Backpropagate through self.agg_function
        ds, = vjp_agg(cotangent)

        # Backpropagate through all steps
        for vjp_fun in reversed(funs):
            ds, = vjp_fun(ds)
        return l, attrgetter(f'variables.{self.var_name}')(ds)

    def __str__(self) :
        return "vjp_diff"

class vjp_grad_scan(autodiff):

    def g(self, state, var_value, iterations=1, **kwargs):
        
        # 1. Define the complete forward trajectory as a single pure function
        def forward_trajectory(var_val):
            # Initialize state with the target variable injected
            init_state = autodiff.set_var(self.var_name, state, var_val)
            
            # 2. Define the body of the scan loop
            # jax.lax.scan requires a signature of: body_fun(carry, x) -> (new_carry, y)
            def scan_step(carry_state, _):
                # Apply jax.checkpoint here to prevent saving intermediate activations
                # inside the Veros step, strictly bounding memory usage.
                next_state = jax.checkpoint(
                    self.step_function, 
                    policy=jax.checkpoint_policies.nothing_saveable
                )(carry_state)
                
                # We don't need to accumulate per-step outputs (y) right now, so we return None
                return next_state, None
            
            # 3. Execute the XLA-optimized loop
            # xs=None and length=iterations tells scan to just loop N times
            final_state, _ = jax.lax.scan(
                scan_step, 
                init_state, 
                xs=None, 
                length=iterations
            )
            
            # 4. Return the aggregated scalar loss
            return self.agg_function(final_state)

        # 5. Let JAX natively handle the reverse-mode autodiff through the scan block
        # value_and_grad returns a tuple: (loss, gradient_wrt_var_val)
        loss, grad_val = jax.value_and_grad(forward_trajectory)(var_value)
        
        return loss, grad_val

    def __str__(self):
        return "vjp_grad_scanned"    
    
def inject_dynamic_vars(base_state, dynamic_updates):
    """
    Pure functional update of VerosState using its native PyTree definitions.
    Replaces only the arrays specified in dynamic_updates.
    """
    # 1. Flatten the base variables
    var_leaves, var_aux = veros_variables_pytree_flatten(base_state.variables)
    var_keys = var_aux[0] # Tuple of all variable names
    
    # 2. Swap out the dynamic arrays; keep static/diagnostic arrays unchanged
    new_leaves = [
        dynamic_updates[k] if k in dynamic_updates else leaf
        for k, leaf in zip(var_keys, var_leaves)
    ]
    
    # 3. Rebuild the variables object
    new_vars = veros_variables_pytree_unflatten(var_aux, new_leaves)
    
    # 4. Rebuild the full state object
    _, state_aux = veros_state_pytree_flatten(base_state)
    new_state = veros_state_pytree_unflatten(state_aux, [new_vars])
    
    return new_state

class vjp_grad_scanned_Nlvls(autodiff):

    def g(self, full_state, var_value, prognostic_keys, chunk_sizes=(3, 3, 3, 3, 3, 3), **kwargs):
        
        # Verify the chunks multiply out to the total iterations exactly
        iterations = 1
        for c in chunk_sizes: 
            iterations *= c
            
        def forward_trajectory(var_val):
            is_prognostic = self.var_name in prognostic_keys
            init_carry = {k: getattr(full_state.variables, k) for k in prognostic_keys}
            
            if is_prognostic:
                init_carry[self.var_name] = var_val

            # --- LEVEL 1: The Innermost Step ---
            @jax.checkpoint
            def innermost_step(inner_tuple, _):
                dynamic_s, step_var_val = inner_tuple
                
                updates = dict(dynamic_s)
                if not is_prognostic:
                    updates[self.var_name] = step_var_val
                    
                curr_state = inject_dynamic_vars(full_state, updates)
                next_state = self.step_function(curr_state)
                next_carry = {k: getattr(next_state.variables, k) for k in prognostic_keys}
                
                return (next_carry, step_var_val), None

            # --- RECURSIVELY BUILD THE MIDDLE LEVELS ---
            # We use a builder function to prevent Python late-binding closure bugs
            def make_chunk_func(func_to_scan, length):
                def chunk_def(carry, _):
                    out, _ = jax.lax.scan(func_to_scan, carry, xs=None, length=length, unroll=1)
                    return out, None
                
                # Apply the checkpoint boundary to the dynamically generated loop
                return jax.checkpoint(chunk_def, policy=jax.checkpoint_policies.nothing_saveable)

            current_func = innermost_step
            
            # Build from the inside out (skipping the outermost layer for the final scan)
            for size in reversed(chunk_sizes[1:]):
                current_func = make_chunk_func(current_func, size)

            # --- GLOBAL: Outermost Scan ---
            final_tuple, _ = jax.lax.scan(
                current_func, 
                (init_carry, var_val), 
                xs=None, 
                length=chunk_sizes[0],
                unroll=1
            )
            
            final_dynamic, final_var_val = final_tuple
            
            # Rebuild final state for loss calculation
            final_updates = dict(final_dynamic)
            if not is_prognostic:
                final_updates[self.var_name] = final_var_val
                
            final_state = inject_dynamic_vars(full_state, final_updates)
            
            return self.agg_function(final_state)

        loss, grad_val = jax.value_and_grad(forward_trajectory)(var_value)
        
        return loss, grad_val

    def __str__(self):
        return "vjp_grad_chunked_Nlvls"
    
def get_prognostic_keys_w_restart(state):
    """
    Filters for variables that hold historical state in the rolling buffer,
    while explicitly retaining the scalar time-tracking indices.
    """
    # Explicitly include the pointers that drive the time-integrator!
    prognostic_keys = ['tau', 'taum1', 'taup1', 'itt', 'time']
    
    for key, meta in state.var_meta.items():
        if meta.active and meta.dims is not None:
            # Grab the physical rolling buffers
            if 'timesteps' in meta.dims or meta.write_to_restart:
                if key not in prognostic_keys:
                    prognostic_keys.append(key)
                
    return prognostic_keys
    
def get_prognostic_keys(state):
    """
    Filters for variables that hold historical state in the rolling buffer,
    while explicitly retaining the scalar time-tracking indices.
    """
    # Explicitly include the pointers that drive the time-integrator!
    prognostic_keys = ['tau', 'taum1', 'taup1', 'itt', 'time']
    
    for key, meta in state.var_meta.items():
        if meta.active and meta.dims is not None:
            # Grab the physical rolling buffers
            if 'timesteps' in meta.dims:
                if key not in prognostic_keys:
                    prognostic_keys.append(key)
                
    return prognostic_keys

def get_prognostic_keys_all(state):
    """
    Carries every single active variable in the Veros state.
    This guarantees 100% physical continuity across chunk boundaries.
    """
    prognostic_keys = []
    
    for key, meta in state.var_meta.items():
        if meta.active:
            if key not in prognostic_keys:
                prognostic_keys.append(key)
                
    # Ensure the core time scalars are present, even if Veros metadata misses them
    for k in ['tau', 'taum1', 'taup1', 'itt', 'time']:
        if k not in prognostic_keys:
            prognostic_keys.append(k)
            
    return prognostic_keys


class jvp_grad(autodiff) :

    def g(self, state, var_value, iterations=1, **kwargs) :
        n_state = autodiff.set_var(self.var_name, state, var_value)
        tangent_state = n_state.get_tangeant(self.var_name)

        for i in range(iterations) :
            print('j', i)
            n_state, tangent_state = jax.jvp(self.step_function, (n_state,), (tangent_state,))
        center, grad_jvp = jax.jvp(self.agg_function, (n_state,), (tangent_state, ))
        return center, grad_jvp

    def __str__(self) :
        return "jvp_diff"

class jvp_grad_scan(autodiff):

    def g(self, state, var_value, iterations=1, **kwargs):

        def forward_trajectory(var_val):
            init_state = autodiff.set_var(self.var_name, state, var_val)

            def scan_step(carry_state, _):
                next_state = self.step_function(carry_state)
                return next_state, None

            final_state, _ = jax.lax.scan(
                scan_step, init_state, xs=None, length=iterations
            )
            return self.agg_function(final_state)

        primal_out, tangent_out = jax.jvp(
            forward_trajectory, (var_value,), (jnp.ones_like(var_value),)
        )
        return primal_out, tangent_out

    def __str__(self):
        return "jvp_diff_scanned"
    

def main():
 
    output_folder = "output_folder_no_restart/"
    os.makedirs(output_folder, exist_ok=True)
 
    gpu_name = get_gpu_name()
    print(f"Running on device: {gpu_name}")

    gpu_tag = get_gpu_tag()
    print(f"GPU tag for output files: {gpu_tag}")
 
    # Initialize simulation and model
    simulation, n_timesteps, sampling_steps = init_simulation_ssh(nb_days_training=1, warmup=0)
    print("simulation initialized")
    prognostic_keys = get_prognostic_keys(simulation.state)
    print(prognostic_keys)
 
    step_function = simulation.step
    agg_function = agg_sum
    var_name = "A_h"
    var_value = jnp.array(5e4, dtype=jnp.float64)
 
    # Powers of 3, matching the 3^n chunk_sizes pattern used by vjp_grad_scanned_Nlvls
    steps_list = [3, 9, 27, 81, 243, 729, 2187, 6561]

    # --- GPU/VRAM-tagged, resumable output file ---
    pkl_path = os.path.join(output_folder, f"ad_comparison_results_{gpu_tag}.pkl")

    existing_results = None
    if os.path.exists(pkl_path):
        print(f"Found existing results file for this GPU/VRAM combo: {pkl_path}")
        with open(pkl_path, "rb") as f:
            existing_results = pickle.load(f)
        done_steps = existing_results.get("steps", [])
        print(f"  -> resuming; already have results for steps: {done_steps}")
    else:
        print(f"No existing results file for this GPU/VRAM combo -> starting fresh: {pkl_path}")
 
    results = run_comparison(
        simulation, prognostic_keys, step_function, agg_function,
        var_name, var_value, steps_list,
        results=existing_results,
        pkl_path=pkl_path,
    )
    results["gpu_name"] = gpu_name
    results["gpu_tag"] = gpu_tag
 
    # Final save (redundant with the incremental saves inside run_comparison,
    # but cheap and guarantees the gpu_name/gpu_tag metadata above is persisted too)
    save_results(results, pkl_path)
    print(f"Saved benchmark results to {pkl_path}")
 
    plot_path = os.path.join(output_folder, f"ad_comparison_{gpu_tag}.png")
    grad_err_pct, loss_err_pct = plot_results(results, gpu_name=gpu_name, save_path=plot_path)
 
    print("\n=== Summary ===")
    for i, n in enumerate(results["steps"]):
        print(
            f"steps={n:5d} | "
            f"grad_jvp={results['jvp_grad'][i]:.4e} grad_vjp={results['vjp_grad'][i]:.4e} "
            f"grad_err%={grad_err_pct[i]:.4f} | "
            f"loss_jvp={results['jvp_loss'][i]:.4e} loss_vjp={results['vjp_loss'][i]:.4e} "
            f"loss_err%={loss_err_pct[i]:.4f} | "
            f"t_jvp={results['jvp_time'][i]:.2f}s t_vjp={results['vjp_time'][i]:.2f}s"
        )
 
    
if __name__ == '__main__':
    main()