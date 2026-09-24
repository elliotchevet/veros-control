"""Small CPU regression tests for persistent Python references in AD steps."""
import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
os.environ['TMPDIR'] = str(ROOT)
import jax
import jax.numpy as jnp
import numpy as np

NS = {'zero_tendency_sources': lambda state: None}
for filename, name in [('ad_benchmark.py', 'nn_epsilon_step'), ('train.py', 'nn_set_forcing_step')]:
    node = next(n for n in ast.parse((ROOT / filename).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), NS)

class State:
    def __init__(self, x):
        self.x = x
        self.settings = SimpleNamespace(unlock=nullcontext, use_forcings=False, baseline_model=True)
    def copy(self):
        return State(self.x)

class LifetimeTests(unittest.TestCase):
    def exercise(self, benchmark, fail=False):
        saved = {'_nn_correction_initialized': False, '_nn_params': object(),
                 '_nn_model_batched': object(), '_nn_stats': object()}
        sim = SimpleNamespace(**saved)
        def step(state):
            if fail:
                raise RuntimeError('intentional failure')
            state.x = state.x + sim._nn_model_batched(sim._nn_params, None, None, None)
        sim.step = step
        def objective(params):
            model = lambda p, *args: p
            state = State(jnp.array(0.))
            if benchmark:
                return NS['nn_epsilon_step'](state, sim, params, model, None, jnp.array(3.)).x
            return NS['nn_set_forcing_step'](state, params, sim, model).x
        if fail:
            with self.assertRaisesRegex(RuntimeError, 'intentional failure'):
                objective(jnp.array(2.))
        else:
            with jax.checking_leaks():
                value, grad = jax.jit(jax.value_and_grad(objective))(jnp.array(2.))
                jax.block_until_ready((value, grad))
            np.testing.assert_allclose(value, 6. if benchmark else 2.)
            np.testing.assert_allclose(grad, 3. if benchmark else 1.)
        for name, value in saved.items():
            self.assertIs(getattr(sim, name), value)

    def test_benchmark_no_leak(self): self.exercise(True)
    def test_training_no_leak(self): self.exercise(False)
    def test_benchmark_exception_restores(self): self.exercise(True, True)
    def test_training_exception_restores(self): self.exercise(False, True)

if __name__ == '__main__':
    unittest.main()
