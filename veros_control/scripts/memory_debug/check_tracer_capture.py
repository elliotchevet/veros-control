import ast
from contextlib import nullcontext
import os
from pathlib import Path
from types import SimpleNamespace
root = Path(__file__).resolve().parents[2]
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
os.environ['TMPDIR'] = str(root / 'scripts/memory_debug')
import jax
import jax.numpy as jnp

class State:
    def __init__(self, x):
        self.x = x
        self.settings = SimpleNamespace(unlock=nullcontext, use_forcings=False, baseline_model=True)
    def copy(self):
        return State(self.x)

sim = SimpleNamespace(_nn_correction_initialized=False, _nn_params=None,
                      _nn_model_batched=None, _nn_stats=None)
def step(state):
    state.x = state.x + sim._nn_model_batched(sim._nn_params, None, None, None)
sim.step = step
namespace = {'zero_tendency_sources': lambda state: None}
node = next(n for n in ast.parse((root / 'training/ad_benchmark.py').read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'nn_epsilon_step')
exec(compile(ast.Module(body=[node], type_ignores=[]), 'nn_epsilon_step', 'exec'), namespace)
def objective(params):
    return namespace['nn_epsilon_step'](State(jnp.array(0.)), sim, params,
                                        lambda p, *args: p, None, jnp.array(1.)).x
try:
    with jax.checking_leaks():
        print(jax.jit(objective)(jnp.array(2.)))
except Exception as exc:
    print('TRACER_LEAK_DETECTED:', str(exc))
    raise SystemExit(2)
print('PASS: no escaped tracer')
