from veros import veros_routine
from veros.core.operators import numpy as npx, update, at

@veros_routine
def set_inits(state):
    vs = state.variables
    vs.dtemp_nn = update(vs.dtemp_nn, at[2:-2, 2:-2, :, :], 0 * vs.maskT[2:-2, 2:-2, :,npx.newaxis]) 
    vs.dsalt_nn = update(vs.dsalt_nn, at[2:-2, 2:-2, :, :], 0 * vs.maskT[2:-2, 2:-2, :,npx.newaxis])  
    vs.du_nn = update(vs.du_nn, at[2:-2, 2:-2, :, :], 0 * vs.maskU[2:-2, 2:-2, :,npx.newaxis])  
    vs.dv_nn = update(vs.dv_nn, at[2:-2, 2:-2, :, :], 0 * vs.maskV[2:-2, 2:-2, :,npx.newaxis])  

