from veros import veros_method 
from veros import time
from veros.core.operators import numpy as npx, update, update_multiply, at

@veros_method
def control(vs):
    vs.temp = update(vs.temp, at[2:-2,2:-2,:,vs.tau], vs.temp[2:-2,2:-2,:,vs.tau]+0.01)
    print(vs.tau)
    print("correction temperaturé")

