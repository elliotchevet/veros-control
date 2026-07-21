import sys
from pathlib import Path
import os
import time
from operator import attrgetter

# Configure environment
PRP = str(Path.cwd().resolve().parents[0]) + '/'
print(PRP)
sys.path.extend([PRP, PRP + 'utils/'])
sys.path.append("/scratch/so2495/code/OceanHybridModels/VerosLearning/veros/")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
import jax
import jax.numpy as jnp
from jax import vjp, vmap, jit
from functools import partial
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import optax

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

    def g(self, state, var_value, iterations=1, sample_steps=1, **kwargs):
        
        def forward_trajectory(var_val):
            init_state = autodiff.set_var(self.var_name, state, var_val)
            
            def scan_step(carry_state, step_idx):
                next_state = jax.checkpoint(
                    self.step_function, 
                    policy=jax.checkpoint_policies.nothing_saveable
                )(carry_state)
                
                # 1. Determine if this is a sampling step
                is_sample_step = (step_idx % sample_steps) == 0
                
                # 2. Conditionally compute the loss using jax.lax.cond
                # If True: runs self.agg_function(next_state)
                # If False: returns a scalar 0.0 without executing agg_function
                step_loss = jax.lax.cond(
                    is_sample_step,
                    lambda s: self.agg_function(s),
                    lambda s: jnp.zeros((), dtype=jnp.float64),
                    next_state
                )
                
                return next_state, step_loss
            
            # Pass indices 1 to `iterations` into the scan via `xs`
            step_indices = jnp.arange(1, iterations + 1)
            
            final_state, all_step_losses = jax.lax.scan(
                scan_step, 
                init_state, 
                xs=step_indices
            )
            
            # Non-sampled steps are already strictly 0.0, so we just sum
            total_loss = jnp.sum(all_step_losses)
            return total_loss

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
class vjp_grad_scanned_2lvls(autodiff):

    def g(self, full_state, var_value, prognostic_keys, iterations=49, chunk_size=7, sample_steps=1, **kwargs):
        
        assert iterations % chunk_size == 0, "iterations must be divisible by chunk_size"
        num_chunks = iterations // chunk_size

        def forward_trajectory(var_val):
            
            # Check if we are differentiating an initial state or a static parameter
            is_prognostic = self.var_name in prognostic_keys
            
            # 1. Base initialization from the pristine, untouched full_state
            init_carry = {k: getattr(full_state.variables, k) for k in prognostic_keys}
            
            # 2. If differentiating an initial condition, inject it into the carry
            if is_prognostic:
                init_carry[self.var_name] = var_val

            # Inner chunk now receives the 1D array of step indices for this chunk
            def inner_chunk_def(carry_tuple, chunk_indices):
                carry_dict, explicit_var_val = carry_tuple
                
                @jax.checkpoint
                def checkpointed_scan_step(inner_tuple, step_idx):
                    dynamic_s, step_var_val = inner_tuple
                    
                    # Prepare the functional updates for this step
                    updates = dict(dynamic_s)
                    
                    # 3. If differentiating a static parameter (like A_h), inject the 
                    # threaded tracer into the update dictionary at every step!
                    if not is_prognostic:
                        updates[self.var_name] = step_var_val
                        
                    # 4. Pure functional build! JAX sees the graph perfectly here.
                    curr_state = inject_dynamic_vars(full_state, updates)
                    
                    next_state = self.step_function(curr_state)
                    next_carry = {k: getattr(next_state.variables, k) for k in prognostic_keys}
                    
                    # 5. Conditionally evaluate the loss using JAX cond
                    is_sample_step = (step_idx % sample_steps) == 0
                    
                    step_loss = jax.lax.cond(
                        is_sample_step,
                        lambda state: self.agg_function(state),
                        lambda state: jnp.zeros((), dtype=jnp.float64),
                        next_state
                    )
                    
                    # Output the loss as the second tuple element (y) to bubble it up
                    return (next_carry, step_var_val), step_loss
                
                chunk_final_tuple, chunk_losses = jax.lax.scan(
                    checkpointed_scan_step, 
                    (carry_dict, explicit_var_val), 
                    xs=chunk_indices, 
                    unroll=1
                )
                return chunk_final_tuple, chunk_losses
            
            inner_chunk = jax.checkpoint(
                inner_chunk_def, 
                policy=jax.checkpoint_policies.nothing_saveable
            )

            # 6. Create the step indices and reshape them for the two-level scan
            # Shape: (num_chunks, chunk_size)
            step_indices = jnp.arange(1, iterations + 1).reshape(num_chunks, chunk_size)

            # Outer scan threads the tuple across all chunks AND passes the indices
            final_tuple, all_chunk_losses = jax.lax.scan(
                inner_chunk, 
                (init_carry, var_val), 
                xs=step_indices, 
                unroll=1
            )
            
            # 7. Aggregate all collected sampled losses
            # Non-sampled steps evaluated to exactly 0.0, so a straight sum is perfectly safe
            total_loss = jnp.sum(all_chunk_losses)
            
            return total_loss

        loss, grad_val = jax.value_and_grad(forward_trajectory)(var_value)
        
        return loss, grad_val

    def __str__(self):
        return "vjp_grad_chunked_filtered_sampled"
    
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

def main():
    # maximum working iters is 100
    # Initialize simulation and model
    simulation, n_timesteps, sampling_steps = init_simulation_ssh(nb_days_training = 1, warmup = 0)
    print("simulation initialized")
    prognostic_keys = get_prognostic_keys(simulation.state)
    print(prognostic_keys)
    step_function = simulation.step
    agg_function = agg_sum
    var_name = 'A_h'
    var_value = jnp.array(5e4, dtype=jnp.float64)
    
    ad = vjp_grad_scanned_2lvls(step_function, agg_function, var_name)
    
    iteration_grad = 25
    sample_steps = 5
    print("gradient computation ...")
    start_time = time.time()
    n_diff = ad.g(simulation.state, var_value=var_value, prognostic_keys = prognostic_keys, iterations=iteration_grad, chunk_size = 5, sample_steps = sample_steps)
    elapsed = time.time() - start_time
    grad = n_diff[1] if isinstance(n_diff, tuple) else n_diff
    grad_val = float(jnp.mean(grad))
    print(f'{ad} : {grad_val}')
    print(f'Time taken for {ad}: {elapsed:.4f} seconds - its={iteration_grad}\n')    
    
    
if __name__ == '__main__':
    main()