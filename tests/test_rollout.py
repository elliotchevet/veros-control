import unittest
import jax
import jax.numpy as jnp
import numpy as np
from veros_control.core.rollout import nested_scan


class RolloutTests(unittest.TestCase):
    def test_nested_scan_matches_forward_and_gradient(self):
        for length in (1,9,27,30):
            def run(parameter,nested):
                def step(state,index):
                    state=.97*state+parameter*jnp.sin(index*.1)
                    return state,jnp.where(index%3==0,state**2,0.)
                if nested:
                    final,losses=nested_scan(step,jnp.asarray(.2),length)
                else:
                    final,losses=jax.lax.scan(step,jnp.asarray(.2),jnp.arange(1,length+1))
                return final+losses.sum()
            x=jnp.asarray(.03)
            np.testing.assert_allclose(run(x,True),run(x,False),rtol=1e-6)
            actual=jax.jit(jax.grad(lambda p:run(p,True)))(x)
            expected=jax.grad(lambda p:run(p,False))(x)
            np.testing.assert_allclose(actual,expected,rtol=1e-6)

if __name__=='__main__':unittest.main()
