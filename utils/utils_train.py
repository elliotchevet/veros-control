import sys;
sys.path.append("/scratch/so2495/code/OceanHybridModels/VerosLearning/veros/")
import h5py
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from tqdm import tqdm
from functools import partial
from dataclasses import dataclass, field
from typing import Union, List
import glob
from utils_grad import run_jvp, run_vjp
from setups.global_1deg_learning import GlobalOneDegreeLearningSetup as GlobalOneDegreeLearningSetup
from veros.core.operators import update, at
from veros.io_tools import hdf5 as h5tools
from veros import veros_routine


def read_single_from_h5(var_meta, infile, groupname):
    """
    Read variables and their attributes from an HDF5 file group (from veros code).

    Args:
        var_meta (dict): Metadata for the variables to be read.
        infile (h5py.File): The HDF5 file object.
        groupname (str): The name of the group in the HDF5 file to read from.

    Returns:
        tuple: A tuple containing:
            - attributes (dict): Attributes of the variables present in var_meta.
            - variables (dict): Variables read from the HDF5 file as jax arrays.
    """
    variables = {}

    # Iterate over the variables in the group
    for key, var in infile[groupname].items():
        if key in var_meta.keys():
            # Directly store the variable from HDF5 as a jax array
            variables[key] = jnp.array(var)

    # Extract attributes only for variables present in var_meta
    attributes = {key: var.item() for key, var in infile[groupname].attrs.items() if key in var_meta}

    return attributes, variables

def loss_function_acc_transport_predefined(state, target_acc_transport, iter):
    """
    Calculate the loss between the current state and a predefined target ACC transport.
    
    Args:
        state: The current state of the simulation.
        target_acc_transport (float or jnp.ndarray): The predefined target ACC transport value, 
                                                     not computed from ground truth files.

    Returns:
        float: The calculated mean squared error loss.
    """        
    # Calculate the predicted ACC transport from the current state
    acc_transport_pred = jnp.mean(jnp.sum(
        state.variables.u[2:-2, 2:-2, :, state.variables.tau] * state.variables.dyt[np.newaxis, 2:-2, np.newaxis] * state.variables.dzt[np.newaxis, np.newaxis, :], 
        axis=(1, 2)
    ))/1E6
    transport_deviation = acc_transport_pred - target_acc_transport
    # Print values for verification (works even if the function is JIT-compiled)
    #jax.debug.print("Predicted Transport: {pred}", pred=acc_transport_pred)
    #jax.debug.print("Transport Deviation: {dev}", dev=transport_deviation)
    return transport_deviation**2

def loss_function_acc_transport(state, gt, iter) :
    """
    Calculate the loss between the current state and ground truth data.
    here, loss function is the ACC transport
    Args:
        state: The current state of the simulation.
        gt (dict): Ground truth data.
        iter (int): file idx to use as ground truth.

    Returns:
        float: The calculated loss.
    """        
    with h5tools.threaded_io(gt[iter], "r") as infile:
        gt_vars = {var: meta for var, meta in state.var_meta.items() if var == 'u' and meta.write_to_restart and meta.active}
        _, gt_data = read_single_from_h5(gt_vars, infile, "core")
    acc_transport_pred = jnp.sum(state.variables.u[2:-2,2:-2,:,state.variables.tau] * state.variables.dyt[np.newaxis, 2:-2, np.newaxis] * state.variables.dzt[np.newaxis, np.newaxis, :], axis = (1,2))    
    acc_transport_true = jnp.sum(gt_data['u'][2:-2,2:-2,:,state.variables.tau] * state.variables.dyt[np.newaxis, 2:-2, np.newaxis] * state.variables.dzt[np.newaxis, np.newaxis, :], axis = (1,2))
    return ((acc_transport_pred - acc_transport_true)**2).mean() 

def loss_function_ssh_gulf_stream(state, gt, iter) :
    """
    Calculate the loss between the current state and ground truth data.

    Args:
        state: The current state of the simulation.
        gt (dict): Ground truth data.
        iter (int): file idx to use as ground truth.

    Returns:
        float: The calculated loss.
    """        
    with h5tools.threaded_io(gt[iter], "r") as infile:
        gt_vars = {var: meta for var, meta in state.var_meta.items() if var == 'ssh' and meta.write_to_restart and meta.active}
        _, gt_data = read_single_from_h5(gt_vars, infile, "core")
    return ((state.variables.ssh[:,-150:] - gt_data['ssh'][:,-150:])**2).mean() 

def loss_function_ssh(state, gt, iter) :
    """
    Calculate the loss between the current state and ground truth data.

    Args:
        state: The current state of the simulation.
        gt (dict): Ground truth data.
        iter (int): file idx to use as ground truth.

    Returns:
        float: The calculated loss.
    """        
    with h5tools.threaded_io(gt[iter], "r") as infile:
        gt_vars = {var: meta for var, meta in state.var_meta.items() if var == 'ssh' and meta.write_to_restart and meta.active}
        _, gt_data = read_single_from_h5(gt_vars, infile, "core")
    return ((state.variables.ssh[:,:] - gt_data['ssh'][:,:])**2).mean() 


def loss_function_psi_gulf_stream(state, gt, iter) :
    """
    Calculate the loss between the current state and ground truth data.

    Args:
        state: The current state of the simulation.
        gt (dict): Ground truth data.
        iter (int): file idx to use as ground truth.

    Returns:
        float: The calculated loss.
    """        
    with h5tools.threaded_io(gt[iter], "r") as infile:
        gt_vars = {var: meta for var, meta in state.var_meta.items() if var == 'psi' and meta.write_to_restart and meta.active}
        _, gt_data = read_single_from_h5(gt_vars, infile, "core")
    return ((state.variables.psi[:,-150:,state.variables.tau] - gt_data['psi'][:,-150:,state.variables.tau])**2).mean() 

def loss_function_psi(state, gt, iter) :
    """
    Calculate the loss between the current state and ground truth data.

    Args:
        state: The current state of the simulation.
        gt (dict): Ground truth data.
        iter (int): file idx to use as ground truth.

    Returns:
        float: The calculated loss.
    """        
    with h5tools.threaded_io(gt[iter], "r") as infile:
        gt_vars = {var: meta for var, meta in state.var_meta.items() if var == 'psi' and meta.write_to_restart and meta.active}
        _, gt_data = read_single_from_h5(gt_vars, infile, "core")
    return ((state.variables.psi[:,:,state.variables.tau] - gt_data['psi'][:,:,state.variables.tau])**2).mean()


def initialize_variables(simulation, variables):
    with simulation.state.variables.unlock():
        for var_name, var_value in variables.items():
            setattr(simulation.state.variables, var_name,
                    update(getattr(simulation.state.variables, var_name), at[...], var_value))
            print(f'{var_name} value: {getattr(simulation.state.variables, var_name)}')
def init_simulation(nb_days_training = 2, warmup = 20):
    """
    Initialize the simulation.

    Args:
        A_h_init (float): Initial value for the A_h variable.

    Returns:
        tuple: A tuple containing:
            - simulation (ACCLearningSetup): The initialized simulation object.
            - n_timesteps (int): The number of timesteps for the simulation.
            - sampling_steps (int): The number of sampling steps.
            - nb_days_training (int): The number of training days.
    """
    simulation = GlobalOneDegreeLearningSetup()
    simulation.setup()

    nb_sec_per_day = 24 * 60 * 60
    runlen = nb_days_training * nb_sec_per_day
    with simulation.state.settings.unlock():
        simulation.state.settings.runlen = runlen

    n_timesteps = runlen // simulation.state.settings.dt_mom

    sampling_steps = n_timesteps // nb_days_training
    
    for i in tqdm(range(warmup)):
        simulation.step(simulation.state)

    return simulation, int(n_timesteps), int(sampling_steps)

class GlobalOneDegreeLearningValid(GlobalOneDegreeLearningSetup):
    def __init__(self, identifier):
        self.custom_identifier = identifier
        super().__init__()
        
    @veros_routine
    def set_parameter(self, state):
        super().set_parameter(state)
        state.settings.identifier = self.custom_identifier
    
    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        diagnostics = state.diagnostics

        diagnostics["energy"].sampling_frequency = settings.dt_tracer *10
        diagnostics["energy"].output_frequency   = 86400.0*30  # 30 day

        diagnostics["snapshot"].output_frequency = 86400*30 # 30 day
        diagnostics["snapshot"].output_variables = (
            "salt",
            "temp",
            "u",
            "v",
            "w",
            "psi",
            "ssh",
        )

        diagnostics["averages"].output_variables = (
            "salt",
            "temp",
            "u",
            "v",
            "w",
            "psi",
            "surface_taux",
            "surface_tauy",
            "ssh",
        )
        diagnostics["averages"].output_frequency = 365 * 86400.0 # 1 year
        diagnostics["averages"].sampling_frequency = settings.dt_tracer *10

    
def init_simulation_ssh(nb_days_training = 2, warmup = 20, evaluation = False, path_eval_simulation = ''):
    """
    Initialize the simulation.

    Args:
        A_h_init (float): Initial value for the A_h variable.

    Returns:
        tuple: A tuple containing:
            - simulation (ACCLearningSetup): The initialized simulation object.
            - n_timesteps (int): The number of timesteps for the simulation.
            - sampling_steps (int): The number of sampling steps.
            - nb_days_training (int): The number of training days.
    """
    #if evaluation:
    #    ACCLearningSetupSSH.set_diagnostics = custom_eval_diagnostics
    if evaluation:
        simulation = GlobalOneDegreeLearningValid(identifier = path_eval_simulation)
    else:
        simulation = GlobalOneDegreeLearningSetup()
    #if evaluation:
    #    print("modification path identifier")
    #    with simulation.state.settings.unlock():
    #        simulation.state.settings.identifier = path_eval_simulation  
    print("############################### : ", simulation.state.settings.identifier)
    #print("############################### : ", simulation.state.variables)
    simulation.setup()

    nb_sec_per_day = 24 * 60 * 60
    runlen = nb_days_training * nb_sec_per_day
    with simulation.state.settings.unlock():
        simulation.state.settings.runlen = runlen

    n_timesteps = runlen // simulation.state.settings.dt_mom

    sampling_steps = n_timesteps // nb_days_training
    
    for i in tqdm(range(warmup)):
        simulation.step(simulation.state)
        
    return simulation, int(n_timesteps), int(sampling_steps)

def init_simulation_A_h(A_h_init, nb_days_training = 2):
    """
    Initialize the simulation.

    Args:
        A_h_init (float): Initial value for the A_h variable.

    Returns:
        tuple: A tuple containing:
            - simulation (ACCLearningSetup): The initialized simulation object.
            - n_timesteps (int): The number of timesteps for the simulation.
            - sampling_steps (int): The number of sampling steps.
            - nb_days_training (int): The number of training days.
    """
    simulation = GlobalOneDegreeLearningSetup()
    simulation.setup()

    with simulation.state.variables.unlock():
        simulation.state.variables.A_h = update(simulation.state.variables.A_h, at[...], A_h_init)

    print(f'A_h value: {simulation.state.variables.A_h}')

    nb_sec_per_day = 24 * 60 * 60
    runlen = nb_days_training * nb_sec_per_day
    with simulation.state.settings.unlock():
        simulation.state.settings.runlen = runlen

    n_timesteps = runlen // simulation.state.settings.dt_mom

    sampling_steps = n_timesteps // nb_days_training

    return simulation, int(n_timesteps), int(sampling_steps)

def step_function_(function, state) :
    """
    Perform a single step of the simulation in a pure way (no inplace modification)

    Args:
        function: The step function to apply.
        state: The current state of the simulation.

    Returns:
        The next state of the simulation.
    """

    next_state = state.copy()
    function(next_state)
    return next_state

def step_function_vjp_(function, state, var):
    next_state = state.copy()
    with next_state.variables.unlock():
        for param, value in var.items():
            setattr(next_state.variables, param, value)
    function(next_state)
    return next_state 

from functools import partial
import numpy as np
from jax import jit

def training_step(simulation, GT_files, grad_var_name, n_timesteps, sampling_steps, loss_function_, predefined_transport=None):
    """
        Perform a single training step.

        Args:
            simulation (ACCLearningSetup): The simulation object.
            GT_files (list str): list of restart files to use as GT.
            grad_var_name (str): Name of the variable to calculate gradients for.
            n_timesteps (int): Number of timesteps to simulate.
            sampling_steps (int): Number of steps between samples.
            loss_function_ (callable): The base loss function to use.
            predefined_transport (float, optional): A predefined target transport value. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - float: The mean of errors over all sampled timesteps.
                - float: The mean of the calculated gradients.
    """

    # Check if predefined_transport is provided and is a number
    if predefined_transport is not None and isinstance(predefined_transport, (int, float, np.number)):
        # Bind the predefined transport target for the new loss function
        loss_function = partial(loss_function_, target_acc_transport=predefined_transport)
    else:
        # Default to the ground truth files for the original loss function
        loss_function = partial(loss_function_, gt=GT_files)

    step_function = jit(lambda state : step_function_(simulation.step, state))

    losses, grads = run_jvp(
        simulation.state, 
        grad_var_name, 
        step_function, 
        loss_function, 
        n_iterations=n_timesteps, 
        sampling_steps=sampling_steps
    )

    return np.mean(losses), np.mean(grads)

def training_step_backward(simulation, GT_files, grad_var, n_timesteps, sampling_steps, loss_function_) :
    """
        Perform a single training step.

        Args:
            simulation (ACCLearningSetup): The simulation object.
            GT_files (list str): list of restart files to use as GT.
            grad_var_name (str): Name of the variable to calculate gradients for.
            n_timesteps (int): Number of timesteps to simulate.

        Returns:
            tuple: A tuple containing:
                - float: The sum of errors over all timesteps.
                - float: The updated value of A_h.
    """


    loss_function = partial(loss_function_, gt=GT_files)
    step_function = jit(lambda state, var : step_function_vjp_(simulation.step, state, var))
    losses, grads = run_vjp(simulation.state, grad_var, step_function, loss_function, n_iterations=n_timesteps, sampling_steps = sampling_steps)

    return losses, grads


import numpy as np

def sgd(simulation, grads, lr):
    """
    Perform a single step of Stochastic Gradient Descent (SGD) optimization for multiple parameters.

    Args:
        simulation (ACCLearningSetup): The simulation object.
        grads (dict): Dictionary where keys are parameter names and values are scalar gradients.
        lr (float or dict): Learning rate. Can be:
            - A single scalar (same LR for all parameters).
            - A dictionary {param_name: lr} for per-parameter learning rates.

    Updates:
        simulation.state.variables: The updated values of the specified parameters.
    """
    if isinstance(lr, (float, int)):
        lr = {param: lr for param in grads}

    with simulation.state.variables.unlock():
        for param, grad in grads.items():
            # Get current parameter value
            param_value = getattr(simulation.state.variables, param)
            # Update using SGD
            new_value = param_value - lr[param] * grad
            setattr(simulation.state.variables, param, new_value)

@dataclass
class Config:
    epochs: int = 1
    batch_size: int = 16
    training_files: list = field(default_factory=lambda: sorted(
        glob.glob('/scratch/so2495/data/numerical_simulations/veros_output/acc_quarter/acc_simulation_quarter_postspinup_with_restart_restarts/*.h5')
    ))
    grad_var_name: Union[str, List[str]] = 'A_h'