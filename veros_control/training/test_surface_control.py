"""Surface-head and filtered surface-objective regression checks."""
import unittest
from types import SimpleNamespace as NS
import numpy as np
import jax
import jax.numpy as jnp
from veros import runtime_settings
runtime_settings.backend = 'jax'
from veros_control.core.model import VerticalCorrectionMLP
from veros_control.core.checkpointing import checkpoint_model_kwargs
from veros_control.training.basin_loss import make_basin_loss
from veros_control.training.spatial_filter import DEFAULT_FILTER


class SurfaceControlTests(unittest.TestCase):
    def test_full_inputs_surface_outputs_and_gradients(self):
        x = jnp.ones((4, 6))
        coords = jnp.zeros((2, 4))
        for groups in (1, 2, 4):
            model = VerticalCorrectionMLP(nz=4, hidden_dim=8, residual_blocks=1,
                surface_only=True, use_forcings=True, use_lon_lat=True,
                num_mlps=groups, init_eps=1.)
            params = model.init(jax.random.PRNGKey(3), x, coords)
            out = model.apply(params, x, coords)
            self.assertEqual(out.shape, (4, 4))
            np.testing.assert_array_equal(out[:-1], 0.)
            self.assertGreater(float(jnp.linalg.norm(out[-1])), 0.)
            grad = jax.grad(lambda f: jnp.sum(model.apply(params, f, coords)[-1] ** 2))(x)
            self.assertGreater(float(jnp.linalg.norm(grad[:-1])), 0.)
            projection = params['params']['input_projection_0']['kernel']
            self.assertEqual(projection.shape[0], 4 * 6 + 16)
            restored = VerticalCorrectionMLP(**checkpoint_model_kwargs({'surface_only': True, 'residual_blocks': 1},
                dict(nz=4, hidden_dim=8, use_forcings=True, use_lon_lat=True, num_mlps=groups, init_eps=1.)))
            np.testing.assert_array_equal(restored.apply(params, x, coords), out)

    def test_old_checkpoints_and_incompatible_mixing(self):
        self.assertFalse(checkpoint_model_kwargs({}, {'surface_only': True})['surface_only'])
        model = VerticalCorrectionMLP(nz=4, surface_only=True, conservative_mixing=True)
        with self.assertRaises(ValueError):
            model.init(jax.random.PRNGKey(0), jnp.ones((4, 4)), dz=jnp.ones(4))

    def test_filtered_surface_loss_ignores_deep_layers(self):
        lon = np.arange(8) * 45.
        lat = np.array([-45., -15., 15., 45.])
        shape = (12, 8, 3)
        vs = NS(xt=np.r_[lon[-2:]-360, lon, lon[:2]+360],
                yt=np.r_[[-75.,-60.],lat,[60.,75.]], zt=np.array([-200.,-50.,-5.]),
                dzt=np.array([100.,40.,10.]), maskT=np.ones(shape), area_t=np.ones(shape[:2])*1e10)
        target = {k:np.zeros(shape) for k in ('temp','salt')}
        calibration = dict(depths_m=np.abs(vs.zt).tolist(), thickness_m=vs.dzt.tolist(),
            thickness_weighting=True, band_boundaries_m=[100.,1000.],
            denominators={k:[1.,2.,4.] for k in target})
        _,_,terms,meta = make_basin_loss(vs,target,levels=1,pattern_lambda=1.,
            depth_loss=calibration,spatial_filter=dict(DEFAULT_FILTER),lead_hours=lambda s:243.)
        self.assertEqual(meta['level_indices'],[2])
        field = jnp.ones(shape).at[:,:,:2].set(1000.)
        loss = lambda f: terms(NS(variables=NS(temp=f,salt=f,tau=0)))[0]
        self.assertAlmostEqual(float(loss(field)),.25,places=6)
        grad = jax.grad(loss)(field)
        np.testing.assert_array_equal(grad[:,:,:2],0.)
        self.assertGreater(float(jnp.linalg.norm(grad[:,:,2])),0.)
        np.testing.assert_allclose(loss(field.at[:,:,:2].set(jnp.nan)),loss(field))


if __name__ == '__main__':
    unittest.main()
