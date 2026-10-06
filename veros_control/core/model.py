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
    zero_output_init: bool = True


class VerticalCorrectionMLP(nn.Module):
    nz: int
    hidden_dim: int = 64
    depth: int = 2
    use_lon_lat: bool = True
    use_forcings: bool = True
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
        for _ in range(self.depth):
            x = nn.relu(nn.Dense(self.hidden_dim)(x))
        options = dict(kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros) if self.zero_output_init else {}
        return nn.Dense(self.nz * 4, **options)(x).reshape(self.nz, 4)


def initialize(config, nz, seed):
    from dataclasses import asdict
    model = VerticalCorrectionMLP(nz=nz, **asdict(config))
    features = jnp.zeros((nz, 9 if config.use_forcings else 4))
    params = model.init(jax.random.PRNGKey(seed), features, jnp.zeros((2, 4)), jnp.ones(nz))
    return model, params
