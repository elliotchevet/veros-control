import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import numpy as np
import jax
import jax.numpy as jnp
spec=importlib.util.spec_from_file_location('basin_loss',Path(__file__).with_name('basin_loss.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class BasinLossTests(unittest.TestCase):
    def fixture(self):
        x=np.array([0,0,-150,-30,50,150,330,0,0.])
        y=np.array([0,0,-75,-30,10,45,80,0,0.])
        z=-np.arange(12,0,-1)*10.
        shape=(len(x),len(y),len(z))
        vs=NS(xt=x,yt=y,zt=z,maskT=np.ones(shape),area_t=np.ones(shape[:2]),tau=0)
        target={k:np.zeros(shape) for k in ('temp','salt')}
        return vs,target

    def test_domains_are_disjoint_and_wrap(self):
        mask=m.basin_masks(np.array([-150,210,-70,20,120]),np.array([-61,-60,0,30,90]))
        self.assertTrue((mask.sum(axis=0)<=1).all())
        np.testing.assert_array_equal(mask[:,0],mask[:,1])
        self.assertTrue(mask[0,2,2]);self.assertTrue(mask[5,0,0])
        self.assertEqual(mask[:,3,4].sum(),0)

    def test_shallow_levels_and_halos(self):
        vs,t=self.fixture()
        scalar,_,terms,meta=m.make_basin_loss(vs,t,levels=5)
        self.assertEqual(meta['level_indices'],[11,10,9,8,7])
        field=np.zeros_like(t['temp']);field[:,:,:7]=100.;field[:2]=np.nan
        state=NS(variables=NS(temp=jnp.array(field),salt=jnp.array(field),tau=0))
        np.testing.assert_allclose(terms(state),0.)
        _,_,terms10,_=m.make_basin_loss(vs,t,levels=10)
        self.assertGreater(float(terms10(state)[0]),0.)

    def test_area_weighted_mean_before_square_and_gradient(self):
        vs,t=self.fixture();vs.maskT[:]=0
        # Two north-Atlantic cells with area ratio 1:3.
        vs.maskT[3,4,:]=1;vs.maskT[6,4,:]=1;vs.area_t[6,4]=3
        _,_,terms,_=m.make_basin_loss(vs,t,levels=5)
        field=jnp.zeros_like(jnp.array(t['temp'])).at[3,4,:].set(3.).at[6,4,:].set(-1.)
        def loss(f):return terms(NS(variables=NS(temp=f,salt=jnp.zeros_like(f),tau=0)))[0]
        self.assertAlmostEqual(float(loss(field)),0.)
        shifted=field.at[3,4,11].add(4.)
        self.assertAlmostEqual(float(loss(shifted)),.2,places=6)
        grad=jax.grad(loss)(shifted)
        self.assertAlmostEqual(float(grad[6,4,11]/grad[3,4,11]),3.,places=6)
        self.assertEqual(float(grad[3,4,0]),0.)

    def test_all_levels_include_deep_errors_and_gradients(self):
        vs, target = self.fixture()
        field = jnp.ones_like(jnp.asarray(target['temp']))
        _, _, terms, meta = m.make_basin_loss(vs, target, levels=0, pattern_lambda=1.)
        self.assertEqual(meta['level_indices'], list(range(11, -1, -1)))
        def loss(f):
            return terms(NS(variables=NS(temp=f, salt=jnp.zeros_like(f), tau=0)))[0]
        self.assertAlmostEqual(float(loss(field)), 1., places=6)
        self.assertGreater(float(jnp.sum(jnp.abs(jax.grad(loss)(field)[:,:,0]))), 0.)
        for bad in (-1, 13):
            with self.assertRaises(ValueError):
                m.make_basin_loss(vs, target, levels=bad)

    def test_sampled_targets_use_each_state_time(self):
        vs, initial = self.fixture()
        target = {k:np.ones_like(v)*8 for k,v in initial.items()}
        _, _, terms, _ = m.make_basin_loss(vs, target, levels=0, pattern_lambda=1.,
            initial_target=initial, target_alpha=lambda state:state.variables.time/720.)
        for step in (27,54,243):
            field = jnp.ones_like(jnp.asarray(initial['temp'])) * (8*step/720.)
            def loss(f):
                return terms(NS(variables=NS(temp=f,salt=field,tau=0,time=jnp.asarray(step))))[0]
            self.assertAlmostEqual(float(loss(field)),0.,places=6)
            self.assertAlmostEqual(float(loss(field+1)),1.,places=6)
            self.assertGreater(float(jnp.sum(jnp.abs(jax.grad(loss)(field+1)))),0.)

    def test_pattern_penalty_and_gradient(self):
        vs, target = self.fixture()
        vs.maskT[:] = 0
        vs.maskT[3,4,:] = 1; vs.maskT[6,4,:] = 1
        vs.area_t[6,4] = 3
        field = jnp.zeros_like(jnp.asarray(target['temp'])).at[3,4,:].set(3.).at[6,4,:].set(-1.)
        for weight in (0., .5, 1., 2.):
            _, _, terms, _ = m.make_basin_loss(vs, target, pattern_lambda=weight)
            def loss(f):
                return terms(NS(variables=NS(temp=f, salt=jnp.zeros_like(f), tau=0)))[0]
            self.assertAlmostEqual(float(loss(field)), 3. * weight, places=6)
            shifted = field.at[3,4,:].add(2.).at[6,4,:].add(2.)
            self.assertAlmostEqual(float(loss(shifted)), 4. + 3. * weight, places=5)
            if weight == 1.:
                grad = jax.grad(loss)(field)
                self.assertAlmostEqual(float(grad[3,4,11]), .3, places=6)
                self.assertAlmostEqual(float(grad[6,4,11]), -.3, places=6)
                self.assertEqual(float(grad[3,4,0]), 0.)

    def test_depth_normalization_thickness_and_gradient(self):
        vs, target = self.fixture()
        vs.dzt = np.arange(1., 13.)
        vs.maskT[:] = 0
        vs.maskT[3,4,:] = 1
        vs.maskT[6,4,:] = 1
        vs.area_t[6,4] = 3
        # A second basin exists only in the shallow half of the column.
        vs.maskT[2,4,6:] = 1
        scales = np.arange(2., 14.)
        stats = dict(thickness_weighting=True, depths_m=np.abs(vs.zt).tolist(),
                     thickness_m=vs.dzt.tolist(), denominators=dict(temp=scales.tolist(), salt=(2*scales).tolist()))
        _, _, terms, _ = m.make_basin_loss(vs, target, levels=0, pattern_lambda=1., depth_loss=stats)
        field = jnp.zeros_like(jnp.asarray(target['temp'])).at[3,4,:].set(3.).at[6,4,:].set(-1.).at[2,4,6:].set(2.)
        def loss(f):
            return terms(NS(variables=NS(temp=f, salt=f, tau=0)))
        expected = .5 * (np.sum(vs.dzt*3/scales)/vs.dzt.sum()
                         + np.sum(vs.dzt[6:]*4/scales[6:])/vs.dzt[6:].sum())
        np.testing.assert_allclose(loss(field), [expected, expected/2], rtol=1e-6)
        grad = jax.grad(lambda f: loss(f)[0])(field)
        self.assertAlmostEqual(float(grad[3,4,0]), .5*vs.dzt[0]/vs.dzt.sum()*2*3*.25/scales[0], places=7)
        self.assertEqual(float(grad[2,4,0]), 0.)
        stats['denominators']['temp'][0] = 0.
        with self.assertRaises(ValueError):
            m.make_basin_loss(vs, target, levels=0, depth_loss=stats)

    def test_equal_bands_with_unequal_thickness(self):
        vs, target = self.fixture()
        vs.dzt = np.arange(1., 13.)
        vs.maskT[:] = 0; vs.maskT[3,4,:] = 1
        boundaries = [40., 80.]
        band = np.digitize(np.abs(vs.zt), boundaries)
        field = jnp.broadcast_to(jnp.asarray(band+1.), target['temp'].shape)
        scale = np.arange(2., 14.)
        cfg = dict(thickness_weighting=True, depths_m=np.abs(vs.zt).tolist(),
                   thickness_m=vs.dzt.tolist(), band_boundaries_m=boundaries,
                   denominators=dict(temp=scale.tolist(),salt=scale.tolist()))
        _, _, terms, _ = m.make_basin_loss(vs,target,levels=0,pattern_lambda=1.,depth_loss=cfg)
        def loss(f):return terms(NS(variables=NS(temp=f,salt=f,tau=0)))[0]
        expected=np.mean([np.average((band[band==b]+1.)**2/scale[band==b],weights=vs.dzt[band==b]) for b in range(3)])
        self.assertAlmostEqual(float(loss(field)),expected,places=6)
        grad=jax.grad(loss)(field)
        for k in range(12):
            want=2*(band[k]+1)*vs.dzt[k]/vs.dzt[band==band[k]].sum()/scale[k]/3
            self.assertAlmostEqual(float(grad[3,4,k]),want,places=6)
        # A basin with no deep band averages its remaining active bands only.
        vs.maskT[:,:,band==2]=0
        _, _, terms, _ = m.make_basin_loss(vs,target,levels=0,pattern_lambda=1.,depth_loss=cfg)
        expected=np.mean([np.average((band[band==b]+1.)**2/scale[band==b],weights=vs.dzt[band==b]) for b in range(2)])
        self.assertAlmostEqual(float(terms(NS(variables=NS(temp=field,salt=field,tau=0)))[0]),expected,places=6)

if __name__=='__main__':unittest.main()
