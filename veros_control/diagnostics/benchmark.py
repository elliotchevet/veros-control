"""Time a single caller-supplied operation; no training-length experiments."""
import time
import jax


def timed_call(function,*args,**kwargs):
    start=time.perf_counter()
    result=function(*args,**kwargs)
    jax.block_until_ready(result)
    return result,time.perf_counter()-start
