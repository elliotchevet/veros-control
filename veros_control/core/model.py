"""Column-wise MLP; output units and constraints are applied in correction.py."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    depth: int = 2
    use_lon_lat: bool = True
    use_forcings: bool = True
    residual_blocks: int = 0
    surface_only: bool = False
    zero_output_init: bool = True


class VerticalCorrectionMLP(nn.Module):
    nz: int
    hidden_dim: int = 64
    depth: int = 2
    use_lon_lat: bool = True
    use_forcings: bool = True
    residual_blocks: int = 0
    surface_only: bool = False
    zero_output_init: bool = True

    @nn.compact
    def __call__(self, features, coordinates=None, dzt=None):
        x = features.reshape(-1)
        if self.use_lon_lat:
            if coordinates is None:
                raise ValueError('Coordinate inputs are enabled but missing')
            lon, lat = jnp.deg2rad(coordinates)
            encoded = jnp.stack((jnp.sin(lon), jnp.cos(lon), jnp.sin(lat), jnp.cos(lat)))
            x = jnp.concatenate((x, encoded.reshape(-1)))
        if self.residual_blocks < 0 or self.depth < 1 or self.hidden_dim < 1:
            raise ValueError('Invalid residual block count, depth, or width')
        if self.residual_blocks:
            x = nn.Dense(self.hidden_dim, name='input_projection_0')(x)
            for block in range(self.residual_blocks):
                residual = x
                for layer in range(self.depth):
                    residual = nn.Dense(self.hidden_dim, name=f'block_0_{block}_{layer}')(residual)
                    if layer < self.depth - 1:
                        residual = nn.relu(residual)
                x = x + residual
        else:
            for _ in range(self.depth):
                x = nn.relu(nn.Dense(self.hidden_dim)(x))
        options = dict(kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros) if self.zero_output_init else {}
        levels = 1 if self.surface_only else self.nz
        output = nn.Dense(levels * 4, **options)(x).reshape(levels, 4)
        return jnp.pad(output, ((self.nz - levels, 0), (0, 0)))


def initialize(config, nz, seed):
    from dataclasses import asdict
    model = VerticalCorrectionMLP(nz=nz, **asdict(config))
    features = jnp.zeros((nz, 9 if config.use_forcings else 4))
    params = model.init(jax.random.PRNGKey(seed), features, jnp.zeros((2, 4)), jnp.ones(nz))
    return model, params
