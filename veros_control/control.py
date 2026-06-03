from veros import veros_routine 
from veros import time
from veros.core.operators import numpy as npx, update, update_multiply, at
from veros_control.core.preprocessing import extract_features

@veros_routine
def control(state):
    vs = state.variables
    features = extract_features(state, vs.tau)
    print(features.shape)
    vs.temp = update(vs.temp, at[2:-2,2:-2,:,vs.tau], vs.temp[2:-2,2:-2,:,vs.tau])#+0.01)
    print("correction temperature")

