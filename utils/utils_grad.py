from functools import partial
from jax import jvp, vjp
import jax.numpy as jnp
from veros.io_tools import hdf5 as h5tools
import matplotlib.pyplot as plt 
import numpy as np
import jax

def step_jvp(step_function, state, jac) :
    """
        Apply the step function and compute the Jacobian with respect to var_name.

        Args:
            step_function (callable): Step function (must be pure).
            state: Veros current state S_{t-1}.
            jac: Veros previous Jacobian dS_{t-1}/dvar_name.

        Returns:
            tuple: A tuple containing:
                - next_state: Veros state after step S_t.
                - jac: Veros Jacobian dS_t / dvar_name.
        """
    # Perform iterations to compute the JVP
    next_state, jac = jvp(step_function, (state,), (jac,))

    return next_state, jac

def loss_jvp(loss_function, state, jac) :
    """
     Compute the loss and its gradient.

     Args:
         loss_function (callable): Function of type loss(State) (must be pure).
         state: Veros state S_t.
         jac: Veros gradient dS_t/dvar_name.

     Returns:
         tuple: A tuple containing:
             - loss (float): loss(S_t).
             - grad (float): dL(S_t)/dvar_name.
     """
    loss, grad = jvp(loss_function, (state,), (jac,))
    return loss, grad

def run_jvp(state, grad_var_name, step_function, loss_function, n_iterations=100, sampling_steps = 1, GT_files = None) :
    """
    Run step function and loss function and return results and gradients.

    Args:
        state: Veros state.
        grad_var_name (str): Name of the variable to differentiate.
        step_function (callable): Function for stepping.
        loss_function (callable): Function for computing the loss.
        n_iterations (int, optional): Number of iterations to run. Defaults to 100.

    Returns:
        list: A list of dictionaries, each containing:
            - 'loss': loss(S_t) for each iteration.
            - 'grad': dL(S_t)/dgrad_var_name for each iteration.
    """
    jac = state.get_tangeant(grad_var_name)
    losses, grads = [], []
    itr_days = 0
    for iter in range(n_iterations):
        state, jac = step_jvp(step_function, state, jac)
        if (iter+1) % sampling_steps == 0:
            itr_days +=1
            l_function = partial(loss_function, iter=itr_days)
            loss, grad = loss_jvp(l_function, state, jac)
            losses.append(loss)
            grads.append(grad)
    return losses, grads


def run_vjp(x0, grad_var, step_function, loss_function, n_iterations=100, sampling_steps=1, gt=None):
    states = [x0]
    vjp_stack = []
    loss_vals = []
    sampled_indices = []  # store the steps where loss is computed
    itr_days = 0

    # Forward pass
    for i in range(n_iterations):
        print('itteration ',i, ' out of ',n_iterations)
        #x = step_function(states[-1], grad_var)
        x, vjp_fun = vjp(step_function, states[-1], grad_var)
        print(i,vjp_fun)
        states.append(x)
        vjp_stack.append(vjp_fun)
            
        if (i + 1) % sampling_steps == 0:
            loss_vals.append(loss_function(x, gt, itr_days))
            sampled_indices.append(i)
            itr_days += 1

    # Total loss
    loss_array = jnp.stack(loss_vals)
    total_loss = jnp.mean(loss_array)

    # Gradients only at sampled steps
    grads = {i: jax.grad(loss_function, argnums=0)(states[i + 1], gt, j)
             for j, i in enumerate(sampled_indices)}

    dl_dx = grads[sampled_indices[-1]]
    dl_dvar_total = 0.0

    # Reverse-mode backprop
    for i in reversed(range(n_iterations)):
        vjp_fun = vjp_stack[i]
        dl_dx, dl_dvar = vjp_fun(dl_dx)
        if i in grads and i != sampled_indices[-1]:
            dl_dx += grads[i]
        dl_dvar_total += dl_dvar
    dl_dvar_total /= len(sampled_indices)
    return total_loss, dl_dvar_total


def run_jvp_list(state, grad_var_name, step_function, loss_function, n_iterations=100, sampling_steps = 1, GT_files = None) :
    """
    Run step function and loss function and return results and gradients.

    Args:
        state: Veros state.
        grad_var_name (str): Name of the variable to differentiate.
        step_function (callable): Function for stepping.
        loss_function (callable): Function for computing the loss.
        n_iterations (int, optional): Number of iterations to run. Defaults to 100.

    Returns:
        list: A list of dictionaries, each containing:
            - 'loss': loss(S_t) for each iteration.
            - 'grad': dL(S_t)/dgrad_var_name for each iteration.
    """
    if isinstance(grad_var_name, str):
        grad_var_name = [grad_var_name] 
        
    grad_vars = []
    losses_vars = [] # should be the same
    initial_state = state.copy() # freez the initial state
    for i in range(len(grad_var_name)):
        jac = state.get_tangeant(grad_var_name[i])
        losses, grads = [], []
        itr_days = 0
        state = initial_state.copy()
        for iter in range(n_iterations):
            state, jac = step_jvp(step_function, state, jac)
            if (iter+1) % sampling_steps == 0:
                itr_days +=1
                l_function = partial(loss_function, iter=itr_days)
                loss, grad = loss_jvp(l_function, state, jac)
                losses.append(loss)
                grads.append(grad)
        grad_vars.append(np.mean(np.array(grads)))
        losses_vars.append(np.mean(np.array(losses)))
    return losses, grads