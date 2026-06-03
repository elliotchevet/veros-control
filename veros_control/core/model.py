from veros.core.operators import numpy as npx, update, at
from jax import vjp, vmap, jit
from functools import partial
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import optax

class VerticalCorrectionMLP(nn.Module):
    hidden_dim: int = 64
    depth: int = 2
    nz: int = 15
    nb_profiles: int = 4                  # always 4 outputs: T, S, U, V
    init_eps: float = 1e-6
    train_epsilon: bool = False
    use_lon_lat: bool = False
    conservative_mixing: bool = False
    use_one_hot: bool = False
    num_mlps: int = 1                     # allowed: 1, 2, or 4
    TS: bool = False                      # if True: use only T,S as input

    @nn.compact
    def __call__(self, x, lon_lat=None, dz=None):
        """
        x: array with shape (nz, 4) = vertical profiles [T,S,U,V]
        lon_lat: optional array with shape (2,4)
        dz: array of vertical layer thickness, shape (nz,)
        returns: array with shape (nz, 4)  # corrections for T,S,U,V
        """

        if x.shape[-1] != 4:
            raise ValueError(f"x must have shape (nz,4), got {x.shape}")

        # -------- Inputs vs Outputs --------
        if self.TS:
            # use only T,S as input
            x_in = x[:, :2]
            if lon_lat is not None:
                lon_lat = lon_lat[:, :2]
            nb_outputs = 4
            if self.num_mlps not in (1, 4):
                raise ValueError("When TS=True, num_mlps must be 1 or 4")
        else:
            x_in = x
            nb_outputs = self.nb_profiles
            if self.num_mlps not in (1, 2, 4):
                raise ValueError("num_mlps must be 1, 2, or 4 when TS=False")

        # -------- Grouping of outputs --------
        if self.num_mlps == 1:
            group_idxs = [npx.arange(nb_outputs)]
        elif self.num_mlps == 2:
            # only valid if TS=False
            group_idxs = [npx.array([0, 1]), npx.array([2, 3])]  # TS / UV
        elif self.num_mlps == 4:
            group_idxs = [npx.array([i]) for i in range(nb_outputs)]
        else:
            raise ValueError("Invalid num_mlps configuration")

        # -------- Forward pass --------
        eps_param_names = [f"raw_epsilon_group_{i}" for i in range(len(group_idxs))]
        outputs = []

        for group_idx, group in enumerate(group_idxs):
            # input: always from x_in (T,S if TS=True; else all 4)
            x_group = x_in.reshape(-1)

            # add lon/lat if requested
            if self.use_lon_lat:
                if lon_lat is None:
                    raise ValueError("lon_lat must be provided when use_lon_lat=True")
                coords_group = lon_lat.reshape(-1)
                x_group = npx.concatenate([x_group, coords_group], axis=0)

            # add one-hot encoding if requested (not for 4 MLPs)
            if self.use_one_hot and self.num_mlps < 4:
                one_hot_vectors = npx.eye(nb_outputs)[group]
                one_hot_flat = one_hot_vectors.flatten()
                x_group = npx.concatenate([x_group, one_hot_flat], axis=0)

            # hidden layers
            for _ in range(self.depth):
                x_group = nn.Dense(self.hidden_dim)(x_group)
                x_group = nn.relu(x_group)

            # output layer -> vertical corrections for this group
            out_dim = self.nz * len(group)
            x_group = nn.Dense(out_dim)(x_group)
            x_group = x_group.reshape(self.nz, len(group))

            # epsilon scaling
            if self.train_epsilon:
                init_val = raw_epsilon_init(self.init_eps)
                raw_eps = self.param(
                    eps_param_names[group_idx],
                    lambda key: npx.full((len(group),), init_val),
                )
                eps = nn.softplus(raw_eps)
            else:
                eps = npx.full((len(group),), self.init_eps)
            x_group = x_group * eps[None, :]



            if self.conservative_mixing:
                if dz is None:
                    raise ValueError("dz must be provided when conservative_mixing=True")
                weighted_mean = (x_group * dz[:, None]).sum(axis=0, keepdims=True) / dz.sum()
                x_group = x_group - weighted_mean

            outputs.append(x_group)

        # -------- Concatenate outputs --------
        x_out = npx.concatenate(outputs, axis=1)  # shape = (nz, nb_outputs)
        return x_out
