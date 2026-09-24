"""Lifetime tests for the contained training-only solver-cache cleanup."""
import ast
from contextlib import contextmanager
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

path = Path(__file__).with_name('ad_benchmark_root.py')
node = next(n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == '_temporary_solver_cache')
namespace = {'contextmanager': contextmanager}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
scope = namespace['_temporary_solver_cache']

class State:
    pass

class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.solver = lambda state: None
        self.solver.cache = {}
        self.modules = patch.dict(sys.modules, {
            'veros.core.external.solvers': SimpleNamespace(get_linear_solver=self.solver),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_preserves_existing_and_releases_temporary_state(self):
        existing = State()
        solver = object()
        self.solver.cache[(existing,)] = solver
        with scope():
            temporary = State()
            ref = weakref.ref(temporary)
            self.solver.cache[(temporary,)] = object()
            del temporary
            gc.collect()
            self.assertIsNotNone(ref())
        gc.collect()
        self.assertIsNone(ref())
        self.assertEqual(self.solver.cache, {(existing,): solver})

    def test_exception_cleanup(self):
        existing = State()
        self.solver.cache[(existing,)] = 'persistent solver'
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with scope():
                self.solver.cache[(State(),)] = 'temporary solver'
                raise RuntimeError('failure')
        self.assertEqual(self.solver.cache, {(existing,): 'persistent solver'})

    def test_nested_scopes(self):
        outer = State()
        with scope():
            self.solver.cache[(outer,)] = 'outer solver'
            with scope():
                self.solver.cache[(State(),)] = 'inner solver'
            self.assertEqual(self.solver.cache, {(outer,): 'outer solver'})
        self.assertFalse(self.solver.cache)

    def test_decorator_preserves_result(self):
        @scope()
        def calculate():
            self.solver.cache[(State(),)] = 'temporary solver'
            return 42
        self.assertEqual(calculate(), 42)
        self.assertFalse(self.solver.cache)

    def test_without_legacy_cache(self):
        del self.solver.cache
        with scope():
            pass

if __name__ == '__main__':
    unittest.main()
