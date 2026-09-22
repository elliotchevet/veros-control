from veros.core.operators import numpy as npx, update, at
from flax import linen as nn


def raw_epsilon_init(target_eps: float) -> float:
    if target_eps <= 0:
        raise ValueError("target_eps must be positive")

    return npx.log(npx.expm1(target_eps))


def encode_lon_lat_sincos(lon_lat):
    lon = lon_lat[0]
    lat = lon_lat[1]
    radians_per_degree = 3.141592653589793 / 180.0
    lon_rad = lon * radians_per_degree
    lat_rad = lat * radians_per_degree
    return npx.stack(
        [
            npx.sin(lon_rad),
            npx.cos(lon_rad),
            npx.sin(lat_rad),
            npx.cos(lat_rad),
        ],
        axis=0,
    )


class VerticalCorrectionMLP(nn.Module):
    hidden_dim: int = 64
    depth: int = 2
    nz: int = 3
    nb_profiles: int = 4                  # always 4 outputs: T, S, U, V
    init_eps: float = 1e-6
    train_epsilon: bool = False
    use_lon_lat: bool = False
    conservative_mixing: bool = False
    use_one_hot: bool = False
    num_mlps: int = 1                     # allowed: 1, 2, or 4
    TS: bool = False                      # if True: use only T,S as input
    use_forcings: bool = False
    zero_output_init: bool = False

    @nn.compact
    def __call__(self, x, lon_lat=None, dz=None):
        """
        x: array with shape (nz, n_features), first channels are [T,S,U,V]
        lon_lat: optional array with shape (2,4), encoded as sin/cos features when used
        dz: array of vertical layer thickness, shape (nz,)
        returns: array with shape (nz, 4)  # corrections for T,S,U,V
        """

        if x.shape[-1] < 4:
            raise ValueError(f"x must have at least 4 channels [T,S,U,V], got {x.shape}")
        if self.use_forcings and x.shape[-1] == 4:
            raise ValueError("use_forcings=True requires forcing channels after [T,S,U,V]")

        state_features = x[:, :4]
        forcing_features = x[:, 4:] if self.use_forcings else x[:, 4:4]

        # -------- Inputs vs Outputs --------
        if self.TS:
            # use only T,S from the state, plus any requested extra inputs
            x_in = npx.concatenate([state_features[:, :2], forcing_features], axis=-1)
            if lon_lat is not None:
                lon_lat = lon_lat[:, :2]
            nb_outputs = 4
            if self.num_mlps not in (1, 4):
                raise ValueError("When TS=True, num_mlps must be 1 or 4")
        else:
            x_in = npx.concatenate([state_features, forcing_features], axis=-1)
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

            # add periodic lon/lat features if requested
            if self.use_lon_lat:
                if lon_lat is None:
                    raise ValueError("lon_lat must be provided when use_lon_lat=True")
                coords_group = encode_lon_lat_sincos(lon_lat).reshape(-1)
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
            output_init = {}
            if self.zero_output_init:
                output_init = {
                    "kernel_init": nn.initializers.zeros,
                    "bias_init": nn.initializers.zeros,
                }
            x_group = nn.Dense(out_dim, **output_init)(x_group)
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

