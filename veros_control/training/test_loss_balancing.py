"""CPU regression checks without importing the GPU-configuring benchmark module."""
import argparse
import ast
from contextlib import contextmanager
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = '1'
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
os.environ['TMPDIR'] = str(ROOT)
import jax
import jax.numpy as jnp
import numpy as np

@contextmanager
def no_solver_cache():
    # These toy rollouts never call the Veros pressure solver; its cache scope
    # has separate lifetime tests in test_solver_cache_scope.py.
    yield

NS = dict(np=np, jnp=jnp, jax=jax, os=os, argparse=argparse, pickle=pickle,
          _temporary_solver_cache=no_solver_cache)

def extract(filename, names):
    # Execute the real definitions while avoiding module-level Veros/GPU setup.
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), NS)

extract('ad_benchmark.py', {'reduce_sampled_losses', 'make_steps_list'})
extract('ad_benchmark_root.py', {
    'parse_args', '_reduce_sampled_loss_terms', 'NNParamsTrainChunkedVJP',
    '_initial_loss_denominators', '_validate_loss_denominators',
    '_load_or_init_train_state', '_resolve_train_n_steps', '_train_checkpoint_payload',
})
extract('plot_train_checkpoint.py', {'_as_float', '_flatten_numeric_dict', 'load_train_history'})

@jax.tree_util.register_pytree_node_class
class State:
    def __init__(self, x):
        self.x = x
    @property
    def variables(self):
        return self
    def copy(self):
        return State(self.x)
    def tree_flatten(self):
        return (self.x,), None
    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

NS['inject_dynamic_vars'] = lambda state, values: State(values['x'])

class LossBalanceTests(unittest.TestCase):
    def args(self):
        with patch.object(sys, 'argv', ['benchmark']):
            args = NS['parse_args']()
        args.train_n_steps = 4
        args.sample_steps = 2
        args.loss_reduction = 'mean'
        args.glorys_interpolation_alpha = None
        return args

    def test_actual_rollout_and_gradients(self):
        for freeze in (False, True):
            for reduction in ('mean', 'sum'):
                ad = NS['NNParamsTrainChunkedVJP'](
                    None, None, None, lambda s: jnp.mean(s.x**2 * jnp.array([1., 100.])),
                    freeze_vars=freeze, loss_reduction=reduction,
                    loss_terms_function=lambda s: s.x**2 * jnp.array([1., 100.]), loss_term_count=2,
                )
                ad.step_function = lambda state, params, epsilon: State(state.x + params)
                kw = dict(prognostic_keys=['x'], chunk_sizes=(2, 2), sample_steps=2, return_aux=True)
                state, params = State(jnp.zeros(2)), jnp.array([1., 2.])
                baseline, terms = ad.g(state, params, forward_only=True, **kw)
                factor = 1 if reduction == 'mean' else 2
                np.testing.assert_allclose(terms, np.array([10., 4000.]) * factor)
                raw, raw_grad, raw_terms = ad.g(state, params, **kw)
                np.testing.assert_allclose(raw, 2005 * factor)
                np.testing.assert_allclose(raw_grad, np.array([10., 2000.]) * factor)
                denominators = NS['_initial_loss_denominators'](terms, ('temp', 'salt'))
                ad.loss_term_denominators = jnp.array(list(denominators.values()))
                balanced, grad, raw_terms = ad.g(state, params, **kw)
                np.testing.assert_allclose(balanced, 1.)
                np.testing.assert_allclose(grad, [1., .5])
                np.testing.assert_allclose(raw_terms, terms)
                # Denominators stay fixed as parameters change, rather than renormalizing every update.
                value, grad, _ = ad.g(state, params * 2, **kw)
                np.testing.assert_allclose(value, 4.)
                np.testing.assert_allclose(grad, [2., 1.])
                kw['return_aux'] = False
                value_no_aux, grad_no_aux = ad.g(state, params * 2, **kw)
                np.testing.assert_allclose(value_no_aux, value)
                np.testing.assert_allclose(grad_no_aux, grad)
                # Legacy no-aux path remains raw when balancing is disabled.
                ad.loss_term_denominators = None
                old_value, old_grad = ad.g(state, params, **kw)
                np.testing.assert_allclose(old_value, raw)
                np.testing.assert_allclose(old_grad, raw_grad)

    def test_invalid_and_zero_baseline(self):
        self.assertEqual(NS['_initial_loss_denominators']([0.], ['temp']), {'temp': 1e-12})
        for values in ([np.nan], [np.inf], [-1.], [1., 2.]):
            with self.assertRaises(ValueError):
                NS['_initial_loss_denominators'](values, ['temp'])
        for values in (None, {}, {'salt': 1.}, {'temp': 0.}, {'temp': np.inf}):
            with self.assertRaises(ValueError):
                NS['_validate_loss_denominators'](values, ['temp'])

    def test_checkpoint_roundtrip_and_legacy(self):
        args = self.args()
        self.assertFalse(args.balance_loss_terms)
        args.balance_loss_terms = True
        args.loss_term_denominators = {'temp': 10., 'salt': 4000.}
        checkpoint = NS['_train_checkpoint_payload'](
            args, 'params', 'optimizer', [], 2, 'cpu', 'cpu', None, None, None, {}, None, 4,
        )
        optimizer = SimpleNamespace(init=lambda params: 'new optimizer')
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / 'checkpoint.pkl'
            def load(path):
                with open(path, 'rb') as stream:
                    return pickle.load(stream)
            NS['load_checkpoint'] = load
            def save(value):
                with path.open('wb') as stream:
                    pickle.dump(value, stream)
            save(checkpoint)
            args.loss_term_denominators = None
            result = NS['_load_or_init_train_state'](args, path, 'initial', optimizer)
            self.assertEqual(result[-1], 2)
            self.assertEqual(args.loss_term_denominators, {'temp': 10., 'salt': 4000.})
            args.balance_loss_terms = False
            with self.assertRaises(ValueError):
                NS['_load_or_init_train_state'](args, path, 'initial', optimizer)
            args.balance_loss_terms = True
            args.sample_steps = 1
            with self.assertRaises(ValueError):
                NS['_load_or_init_train_state'](args, path, 'initial', optimizer)
            args.sample_steps = 2
            damaged = dict(checkpoint, loss_term_denominators=None)
            save(damaged)
            with self.assertRaises(ValueError):
                NS['_load_or_init_train_state'](args, path, 'initial', optimizer)
            legacy = {key: value for key, value in checkpoint.items()
                      if key not in ('balance_loss_terms', 'loss_term_denominators')}
            save(legacy)
            with self.assertRaises(ValueError):
                NS['_load_or_init_train_state'](args, path, 'initial', optimizer)
            args.balance_loss_terms = False
            NS['_load_or_init_train_state'](args, path, 'initial', optimizer)
            self.assertIsNone(args.loss_term_denominators)
            args.balance_loss_terms = True
            args.train_overwrite = True
            result = NS['_load_or_init_train_state'](args, path, 'initial', optimizer)
            self.assertEqual(result, ('initial', 'new optimizer', [], 0))
            self.assertIsNone(args.loss_term_denominators)
            legacy['train_history'] = [{'loss': 1., 'loss_terms': {'temp': 10., 'salt': 4000.},
                                        'normalized_loss_terms': {'temp': 1., 'salt': 1.}}]
            save(legacy)
            rows, _ = NS['load_train_history'](path)
            self.assertEqual(rows[0]['loss_term_salt'], 4000.)
            self.assertEqual(rows[0]['normalized_loss_term_salt'], 1.)

if __name__ == '__main__':
    unittest.main()
