"""Batch gradients, deterministic resume and timestamp/forcing alignment."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import unittest
import tempfile
import json
from unittest.mock import patch
from datetime import datetime
import numpy as np

spec = importlib.util.spec_from_file_location('monthly', Path(__file__).with_name('monthly_batches.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MonthlyTests(unittest.TestCase):
    def test_each_pass_covers_months_once_and_resume_is_exact(self):
        months = tuple(range(1, 12))
        for epoch in range(4):
            batches = [m.batch_months(months, 3, 42, epoch*4+i) for i in range(4)]
            self.assertEqual(sorted(sum(batches, ())), list(months))
            self.assertEqual(batches[2], m.batch_months(months, 3, 42, epoch*4+2))

    def test_cyclic_batches_cover_all_twelve_months(self):
        months = tuple(range(1, 13))
        batches = [m.batch_months(months, 3, 0, i) for i in range(4)]
        self.assertTrue(all(len(b) == 3 for b in batches))
        self.assertEqual(sorted(sum(batches, ())), list(months))

    def test_midmonth_alignment_and_year_wrap(self):
        year = 360*86400.
        with patch.object(m, 'snapshot_date', side_effect=[datetime(1993,12,16,12), datetime(1993,1,16,12)]):
            shift, interval = m.monthly_clock('dec', 'jan', year, cyclic=True)
        self.assertAlmostEqual(shift/year, 349.5/365)
        self.assertAlmostEqual(interval/year, 31/365)
        with patch.object(m, 'snapshot_date', side_effect=[datetime(1993,12,16), datetime(1993,1,16)]):
            with self.assertRaises(ValueError):
                m.monthly_clock('dec', 'jan', year)

    def test_no_nn_calibration_uses_all_months_and_preserves_weights(self):
        import jax
        import jax.numpy as jnp
        with tempfile.TemporaryDirectory() as directory:
            obj = object.__new__(m.MonthlyRollouts)
            obj.m = SimpleNamespace(jax=jax, jnp=jnp)
            obj.files = {month: None for month in range(1, 13)}
            obj.steps = 81
            obj.months = (2, 8, 6)
            obj.single = SimpleNamespace(loss_term_denominators=jnp.ones(2))
            obj.cfg = SimpleNamespace(train_output_file=str(Path(directory)/'train.pkl'), sample_steps=27,
                                      monthly_fused_validation=True)
            obj.rollout_counts = {}
            class Workers:
                def map(self, op, months, params, options, denominators):
                    if op == 'g':
                        self.months = months
                        np.testing.assert_array_equal(params['w'], [0., 0.])
                        assert options['forward_only'] and options['return_aux']
                        return [(0., np.array([month, 4*month])) for month in months]
                    # The fused candidate scorer must receive the new fixed scales.
                    np.testing.assert_allclose(denominators, [6.5, 26.])
                    return [(True, {'loss_terms':[6.5, 26.]}) for month in months]
            obj.workers = Workers()
            params = {'w':jnp.array([2., -3.])}
            result = obj.calibrate_no_nn(params)
            np.testing.assert_array_equal(params['w'], [2., -3.])
            self.assertEqual(obj.workers.months, tuple(range(1,13)))
            self.assertEqual(obj.months, (2,8,6))
            self.assertEqual(result['denominators'], {'temp':6.5, 'salt':26.})
            self.assertEqual(obj.cfg.monthly_baseline_calibration, result)
            self.assertEqual(json.loads((Path(directory)/'no_nn_baseline.json').read_text()), result)
            obj.rollout_counts['candidate'] = 0
            # Avoid importing Veros for the aggregation-only test.
            with patch.dict('sys.modules', {'veros_control.training.stability':SimpleNamespace(make_rollout_validator=None)}):
                valid, _ = obj.validate(params)
            self.assertTrue(valid)
            self.assertAlmostEqual(float(obj.cached_candidate[1][0]), 1.)

    def test_sequential_mean_gradient_matches_analytic_batch(self):
        import jax
        import jax.numpy as jnp
        obj = object.__new__(m.MonthlyRollouts)
        obj.m = SimpleNamespace(jax=jax)
        obj.simulation = SimpleNamespace(state=None)
        obj.months = (1, 5, 9)
        obj.score_months = (2, 6)
        @contextmanager
        def member(month):
            def g(state, params, forward_only=False, **kwargs):
                fn = lambda p: (p['w']-month)**2
                loss, grad = jax.value_and_grad(fn)(params)
                terms = jnp.array([loss, 2*loss])
                return (loss, terms) if forward_only else (loss, grad, terms)
            yield SimpleNamespace(g=g)
        obj.member = member
        params = {'w': jnp.asarray(3.)}
        loss, grad, terms = obj.g(None, params, return_aux=True)
        self.assertAlmostEqual(float(loss), np.mean((3-np.array(obj.months))**2), places=5)
        self.assertAlmostEqual(float(grad['w']), -4.)
        score, _ = obj.checkpoint_score(params)
        self.assertAlmostEqual(float(score), 5.)
        self.assertEqual(obj.months, (1,5,9))  # scoring never changes the training batch
        obj.cached_candidate = (params, (score, terms))
        # Accepted candidate scores must be reused without another trajectory.
        with patch.object(obj, '_mean', side_effect=AssertionError('duplicate rollout')):
            cached, _ = obj.g(None, params, forward_only=True)
        self.assertEqual(float(cached), float(score))
        class FakeWorkers:
            def map(self, operation, months, parameters, options, denominators):
                result=[]
                for month in months:
                    with member(month) as ad: result.append(ad.g(None, parameters, **options))
                return result
        obj.workers = FakeWorkers()
        obj.single = SimpleNamespace(loss_term_denominators=jnp.ones(2))
        obj.m.jnp = jnp
        parallel_loss, parallel_grad, _ = obj._mean(obj.months, params, {'return_aux': True})
        self.assertAlmostEqual(float(parallel_loss), float(loss))
        self.assertAlmostEqual(float(parallel_grad['w']), float(grad['w']))


if __name__ == '__main__':
    unittest.main()
