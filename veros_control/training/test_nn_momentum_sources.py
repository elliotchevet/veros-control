"""CPU checks for NN source application and the Veros momentum-source kernel."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
os.environ['TMPDIR'] = str(ROOT / 'training')
import jax
import jax.numpy as jnp
import numpy as np

class Indexer:
    def __getitem__(self, index):
        return index

NS = dict(jnp=jnp, npx=jnp, at=Indexer(),
          update=lambda a, i, v: a.at[i].set(v),
          update_add=lambda a, i, v: a.at[i].add(v),
          KernelOutput=lambda **kw: kw)

def extract(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), NS)

extract(ROOT / 'core/forcing.py', {'write_nn_tendency_diagnostics', 'add_nn_forcing_to_state', 'reset_nn_momentum_sources'})
extract(ROOT / 'training/train.py', {'zero_tendency_sources'})
extract(Path('/Odyssey/private/e25cheve/veros/veros/core/friction.py'), {'momentum_sources'})

class SourceTests(unittest.TestCase):
    def state(self, enabled=True):
        shape = (2, 2, 3)
        variables = SimpleNamespace(tau=1, unlock=nullcontext)
        for key in ('temp_source', 'salt_source', 'u_source', 'v_source', 'du_mix', 'dv_mix', 'K_diss_bot'):
            setattr(variables, key, jnp.zeros(shape))
        for key in ('dtemp_nn', 'dsalt_nn', 'du_nn', 'dv_nn'):
            setattr(variables, key, jnp.zeros(shape + (3,)))
        variables.maskU = jnp.ones(shape).at[0, 0, 0].set(0)
        variables.maskV = jnp.ones(shape).at[1, 1, 1].set(0)
        return SimpleNamespace(variables=variables,
                               settings=SimpleNamespace(enable_tempsalt_sources=True,
                                                        enable_momentum_sources=enabled,
                                                        enable_conserve_energy=False),
                               var_meta={key: None for key in ('temp_source', 'salt_source', 'u_source', 'v_source')})

    def test_full_depth_sources_reach_momentum(self):
        state = self.state()
        vs = state.variables
        du = jnp.broadcast_to(jnp.array([1., 2., 3.]), vs.u_source.shape) * vs.maskU
        dv = -2 * du * vs.maskV
        vs.u_source = jnp.ones_like(du) * .25
        vs.v_source = jnp.ones_like(dv) * .5
        NS['add_nn_forcing_to_state'](state, du * 3, du * 4, du, dv)
        np.testing.assert_allclose(vs.u_source, du + .25)
        np.testing.assert_allclose(vs.v_source, dv + .5)
        np.testing.assert_allclose(vs.temp_source, du * 3)
        np.testing.assert_allclose(vs.salt_source, du * 4)
        np.testing.assert_allclose(vs.du_nn[..., 1], du)
        np.testing.assert_allclose(vs.dv_nn[..., 1], dv)
        output = NS['momentum_sources'](state)
        np.testing.assert_allclose(output['du_mix'], vs.maskU * (du + .25))
        np.testing.assert_allclose(output['dv_mix'], vs.maskV * (dv + .5))
        self.assertNotEqual(float(output['du_mix'][1, 0, 0]), 0.)

    def test_reset_prevents_accumulation_and_disabled_sources(self):
        state = self.state()
        field = jnp.ones_like(state.variables.u_source)
        for _ in range(3):
            NS['reset_nn_momentum_sources'](state)
            NS['add_nn_forcing_to_state'](state, field * 0, field * 0, field, -field)
            np.testing.assert_allclose(state.variables.u_source, field)
            np.testing.assert_allclose(state.variables.v_source, -field)
        NS['zero_tendency_sources'](state)
        for key in state.var_meta:
            np.testing.assert_allclose(getattr(state.variables, key), 0.)
        state = self.state(enabled=False)
        del state.variables.u_source, state.variables.v_source
        NS['reset_nn_momentum_sources'](state)
        NS['add_nn_forcing_to_state'](state, field, field, field, field)
        np.testing.assert_allclose(state.variables.du_nn[..., 1], field)

    def test_velocity_outputs_receive_gradients(self):
        def objective(amplitudes):
            state = self.state()
            field = jnp.ones_like(state.variables.u_source)
            NS['add_nn_forcing_to_state'](state, field * 0, field * 0,
                                           field * amplitudes[0], field * amplitudes[1])
            output = NS['momentum_sources'](state)
            return jnp.sum(output['du_mix']**2 + output['dv_mix']**2)
        grad = jax.grad(objective)(jnp.array([2., 3.]))
        # Each velocity grid has 11 wet cells; d(sum((a*mask)^2))/da = 22*a.
        np.testing.assert_allclose(grad, [44., 66.])

if __name__ == '__main__':
    unittest.main()
