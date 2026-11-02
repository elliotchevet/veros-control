"""Optional direct numerical comparison with a main checkout, including gradients.

Run with VEROS_CONTROL_MAIN=/path/to/main/veros_control. The plugin itself never
imports main; this path is only a regression oracle during development.
"""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from veros_control.core.model import ModelConfig, initialize
from veros_control.training.loss import make_basin_loss
from veros_control.training.scheduler import loss_test_next_learning_rate


def oracle(file):
    path = Path(os.environ['VEROS_CONTROL_MAIN'])/file
    spec = importlib.util.spec_from_file_location('main_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get('VEROS_CONTROL_MAIN'), 'Set VEROS_CONTROL_MAIN for direct main parity')
class MainParity(unittest.TestCase):
    def test_models_values_and_gradients(self):
        main = oracle('core/model.py')
        for surface in (False, True):
            for blocks in (0, 2):
                config = ModelConfig(hidden_dim=8, depth=2, residual_blocks=blocks,
                    surface_only=surface, zero_output_init=False)
                model, params = initialize(config, 4, 13)
                original = main.VerticalCorrectionMLP(nz=4, hidden_dim=8, depth=2,
                    residual_blocks=blocks, surface_only=surface, zero_output_init=False,
                    use_lon_lat=True, use_forcings=True, init_eps=1.)
                x = jnp.arange(36.).reshape(4,9)*.01
                coords = jnp.arange(8.).reshape(2,4)
                args = (x, coords, jnp.ones(4))
                np.testing.assert_allclose(model.apply(params,*args), original.apply(params,*args), rtol=1e-12, atol=1e-12)
                a = jax.grad(lambda p:jnp.sum(model.apply(p,*args)**2))(params)
                b = jax.grad(lambda p:jnp.sum(original.apply(p,*args)**2))(params)
                for left,right in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
                    np.testing.assert_allclose(left,right,rtol=1e-11,atol=1e-11)
                if surface:
                    np.testing.assert_array_equal(model.apply(params,*args)[:-1],0.)

    def test_loss_values_and_gradients(self):
        main = oracle('training/basin_loss.py')
        x,y,z = 16,12,4
        v = SimpleNamespace(xt=jnp.linspace(-180,180,x),yt=jnp.linspace(-80,80,y),
            zt=jnp.array([-1000.,-300.,-50.,-5.]),dzt=jnp.array([700.,400.,90.,10.]),
            area_t=jnp.ones((x,y)),maskT=jnp.ones((x,y,z)),tau=0)
        target = dict(temp=jnp.zeros((x,y,z)),salt=jnp.ones((x,y,z)))
        depth = dict(thickness_weighting=True,depths_m=np.abs(v.zt).tolist(),thickness_m=v.dzt.tolist(),
            denominators=dict(temp=[1.,2.,3.,4.],salt=[4.,3.,2.,1.]),band_boundaries_m=[100.,700.])
        for levels in (1,0):
            for dl in (None,depth):
                options=dict(levels=levels,pattern_lambda=1.,depth_loss=dl)
                new=make_basin_loss(v,target,**options)[2]
                old=main.make_basin_loss(v,target,**options)[2]
                def score(fn,field):
                    vs=SimpleNamespace(**vars(v),temp=field,salt=field+2)
                    return fn(SimpleNamespace(variables=vs)).sum()
                field=jnp.arange(x*y*z,dtype=float).reshape(x,y,z)*.001
                np.testing.assert_allclose(score(new,field),score(old,field),rtol=1e-13)
                np.testing.assert_allclose(jax.grad(lambda f:score(new,f))(field),jax.grad(lambda f:score(old,f))(field),rtol=1e-13)

    def test_scheduler_sequence(self):
        old=oracle('training/stability.py').loss_test_next_learning_rate
        rates=[1e-3,1e-3];streaks=[0,0]
        for epoch in range(1,61):
            after=1.1 if epoch in (20,21,40) else .9
            results=[]
            for i,fn in enumerate((loss_test_next_learning_rate,old)):
                rate,report=fn(rates[i],1.,after,epoch,clean_updates=streaks[i])
                rates[i],streaks[i]=rate,report['lr_clean_updates'];results.append((rate,report))
            self.assertEqual(*results)
