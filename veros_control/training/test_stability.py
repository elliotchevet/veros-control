"""CPU checks of correction limits, rejection, and intermediate-step validation."""
import importlib.util
from pathlib import Path
from collections import namedtuple
from contextlib import nullcontext
import unittest
import jax
import jax.numpy as jnp
import numpy as np

spec=importlib.util.spec_from_file_location('stability',Path(__file__).with_name('stability.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

class StabilityTests(unittest.TestCase):
    def test_depth_taper_physical_order(self):
        depths = -np.array([140., 110., 95., 80., 60., 45., 30., 20., 10., 5.])
        taper = s.correction_depth_taper(depths, 5, 5)
        np.testing.assert_array_equal(taper[-5:], np.ones(5))
        self.assertEqual(taper[0], 0.)
        self.assertTrue(np.all(np.diff(taper) >= 0.))
        np.testing.assert_allclose(s.correction_depth_taper(depths[::-1], 5, 5), taper[::-1])
        expected = .5 * (1 + np.cos(np.pi * (80.-45.) / (140.-45.)))
        self.assertAlmostEqual(taper[3], expected)
        with self.assertRaises(ValueError):
            s.correction_depth_taper(depths, 5, 6)

    def test_bounds_and_small_signal_gradient(self):
        eps=jnp.array([1e-6,1e-6,1e-7,1e-7]);limits=jnp.array([.05,.05,.005,.005])
        for sign in (-1,1):
            actual=s.bound_corrections(jnp.ones(4)*sign*1e9,eps,limits)*3600
            np.testing.assert_allclose(actual,sign*limits,rtol=1e-6)
        jac=jax.jacfwd(lambda x:s.bound_corrections(x,eps,limits))(jnp.zeros(4))
        np.testing.assert_allclose(jac,np.diag(eps),rtol=1e-6)

    def test_backoff_keeps_one_direction(self):
        p={'x':jnp.array(1.)};direction={'x':jnp.array(-4.)};seen=[]
        def validate(c):
            seen.append(float(c['x']));return bool(c['x']>0),{'x':float(c['x'])}
        candidate,updates,lr,retries,_=s.backtrack_candidate(p,direction,1.,validate)
        self.assertEqual(seen,[-3.,-1.,0.,.5]);self.assertEqual(lr,.125);self.assertEqual(retries,3)
        self.assertEqual(float(p['x']),1.);self.assertEqual(float(direction['x']),-4.)
        self.assertEqual(float(candidate['x']),.5)
        with self.assertRaises(FloatingPointError):
            s.backtrack_candidate(p,direction,1.,lambda c:(False,{}),max_retries=1)
        self.assertEqual(float(p['x']),1.)

    def test_learning_rate_floor_and_warmup(self):
        p = {'x': jnp.array(1.)}
        direction = {'x': jnp.array(1.)}
        score = lambda c: (c['x'] ** 2, jnp.asarray([c['x'] ** 2]))
        for epoch in (1, 19, 20):
            seen = []
            def validate(c):
                seen.append(float(c['x']))
                return True, {}
            _, _, rate, retries, report = s.backtrack_candidate(
                p, direction, 1e-3, validate, score=score, reference_loss=1.,
                min_learning_rate=1e-5, enforce_loss_decrease=epoch >= 20)
            self.assertEqual(rate, 1e-5 if epoch == 20 else 1e-3)
            self.assertEqual(retries, 7 if epoch == 20 else 0)
            self.assertGreater(report['loss'], 1.)
        _, _, rate, _, _ = s.backtrack_candidate(p, direction, 1e-15,
            lambda c: (True, {}), min_learning_rate=1e-5)
        self.assertEqual(rate, 1e-5)
        for validate, candidate_score in (
            (lambda c: (False, {}), score),
            (lambda c: (True, {}), lambda c: (np.nan, [np.nan])),
        ):
            with self.assertRaises(FloatingPointError):
                s.backtrack_candidate(p, direction, 1e-5, validate,
                    score=candidate_score, reference_loss=1., min_learning_rate=1e-5)
        self.assertEqual(float(p['x']), 1.)

    def test_validator_detects_intermediate_invalid_state(self):
        fields=['temp','salt','u','v','w','rho','prho','Nsqr','tke','eke','Hd','kappaM','kappaH']
        names=fields+['maskT','maskU','maskV','maskW','tau','dxt','dyt','dzt','cost','itt']
        Vars=namedtuple('Vars',names); Settings=namedtuple('Settings',['dt_tracer'])
        class State(namedtuple('StateBase',['variables','settings'])):
            def copy(self): return self
        # Register the subclass as a pytree since JAX only recognizes exact namedtuple types.
        shape=(6,6,1)
        arrays={n:jnp.ones(shape+(3,)) for n in fields}
        for n in ['u','v','w']:arrays[n]=jnp.zeros(shape+(3,))
        arrays.update({n:jnp.ones(shape) for n in ['maskT','maskU','maskV','maskW']})
        arrays.update(tau=0,dxt=jnp.ones(6)*1e5,dyt=jnp.ones(6)*1e5,dzt=jnp.ones(1)*100,cost=jnp.ones(6),itt=0)
        state=State(Vars(**arrays),Settings(3600.))
        class Sim: pass
        sim=Sim();sim.state=state
        def step(state,simulation,params,model,stats,epsilon):
            n=state.variables.itt+1
            salt=jnp.where((n==2)&(params>0),-1.,1.)*jnp.ones(shape+(3,))
            return state._replace(variables=state.variables._replace(salt=salt,itt=n))
        validate=s.make_rollout_validator(sim,None,None,1e-6,(1,3),step,nullcontext)
        ok,report=validate(jnp.array(0.));self.assertTrue(ok)
        ok,report=validate(jnp.array(1.));self.assertFalse(ok);self.assertEqual(report['min_salinity'],-1.)

        fused=s.make_rollout_validator(sim,None,None,1e-6,(1,3),step,nullcontext,
            score_terms=lambda st:jnp.asarray([st.variables.itt*1., st.variables.itt*2.]),
            sample_steps=1,score_denominators=jnp.asarray([2.,2.]))
        ok,report=fused(jnp.array(0.))
        self.assertTrue(ok)
        np.testing.assert_allclose(report['loss_terms'],[2.,4.])
        self.assertAlmostEqual(report['loss'],1.5)

    def test_loss_backtracking_transaction_and_recovery(self):
        import optax
        p={'x':jnp.array(1.)};opt=optax.adam(1.)
        state=opt.init(p)
        direction,candidate_state=opt.update({'x':jnp.array(2.)},state,p)
        seen=[]
        def score(c):
            seen.append(float(c['x']))
            value=c['x']**2
            return value,jnp.asarray([value])
        candidate,_,rate,retries,report=s.backtrack_candidate(p,direction,4.,lambda c:(True,{}),
                score=score,reference_loss=1.,loss_rtol=0.,loss_atol=0.)
        self.assertGreater(retries,0)
        self.assertLessEqual(report['loss'],1.)
        self.assertEqual(int(state[0].count),0)
        self.assertEqual(int(candidate_state[0].count),1)
        self.assertEqual(float(p['x']),1.)
        with self.assertRaises(FloatingPointError):
            s.backtrack_candidate(p,{'x':jnp.array(1.)},1.,lambda c:(True,{}),
                max_retries=2,score=score,reference_loss=1.)
        self.assertEqual(int(state[0].count),0)
        with self.assertRaises(FloatingPointError):
            s.backtrack_candidate(p,direction,1.,lambda c:(True,{}),max_retries=0,
                score=lambda c:(np.nan,[np.nan]),reference_loss=1.)
        self.assertEqual(s.recover_learning_rate(.5,2,4),(.125,0))
        self.assertEqual(s.recover_learning_rate(.5,0,4),(.75,0))
        self.assertEqual(s.recover_learning_rate(1.,0,4),(1.,0))

if __name__=='__main__':unittest.main()
