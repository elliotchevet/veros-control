import unittest
from dataclasses import asdict
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from veros_control.core.model import ModelConfig, initialize
from veros_control.core.correction import CorrectionConfig, physical_model, bound_corrections, correction_depth_taper


class CorrectionTests(unittest.TestCase):
    def test_bounds_gradient_and_taper(self):
        eps=jnp.array([1e-6,1e-6,1e-7,1e-7]);bounds=jnp.array([.05,.05,.005,.005])
        np.testing.assert_allclose(bound_corrections(jnp.ones(4)*1e9,eps,bounds)*3600,bounds,rtol=1e-6)
        jac=jax.jacfwd(lambda x:bound_corrections(x,eps,bounds))(jnp.zeros(4))
        np.testing.assert_allclose(jac,np.diag(eps),rtol=1e-6)
        depths=-np.array([140.,110.,95.,80.,60.,45.,30.,20.,10.,5.])
        taper=correction_depth_taper(depths,5,5)
        np.testing.assert_array_equal(taper[-5:],np.ones(5));self.assertEqual(taper[0],0.)
    def test_zero_initialization_and_momentum_switch(self):
        model,params=initialize(ModelConfig(hidden_dim=4,depth=1),3,0)
        features=jnp.ones((2,3,9));coordinates=jnp.ones((2,2,4));dzt=jnp.ones(3)
        apply=physical_model(model,CorrectionConfig(),-jnp.arange(3.))
        np.testing.assert_array_equal(apply(params,features,coordinates,dzt),0.)
        # Enable a nonzero output bias without changing the architecture.
        params['params']['Dense_1']['bias']=jnp.ones(12)
        apply=physical_model(model,CorrectionConfig(momentum=False),-jnp.arange(3.))
        out=apply(params,features,coordinates,dzt)
        self.assertTrue(np.all(np.asarray(out[...,:2])>0))
        np.testing.assert_array_equal(out[...,2:],0.)
    def test_dense_layout_matches_saved_single_mlp(self):
        config=ModelConfig(hidden_dim=4,depth=2,zero_output_init=False)
        model,params=initialize(config,3,12)
        x=jnp.arange(27.).reshape(3,9)*.01;coords=jnp.arange(8.).reshape(2,4)
        lon,lat=jnp.deg2rad(coords)
        value=jnp.concatenate((x.reshape(-1),jnp.stack((jnp.sin(lon),jnp.cos(lon),jnp.sin(lat),jnp.cos(lat))).reshape(-1)))
        for i in range(3):
            layer=params['params'][f'Dense_{i}'];value=value@layer['kernel']+layer['bias']
            if i<2:value=jnp.maximum(value,0.)
        np.testing.assert_allclose(model.apply(params,x,coords,jnp.ones(3)),value.reshape(3,4),rtol=1e-6)

if __name__=='__main__':unittest.main()
