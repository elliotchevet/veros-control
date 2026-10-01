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

if __name__=='__main__':unittest.main()
