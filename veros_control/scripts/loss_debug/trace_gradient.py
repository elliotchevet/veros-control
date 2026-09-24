"""Replay a copied checkpoint with NaN checks on the differentiated program."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import pickle
source = Path(__file__).resolve().parents[2] / 'training/ad_benchmark_root.py'
sys.argv[0] = str(source)
spec = importlib.util.spec_from_file_location('gradient_benchmark', source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
from jax.experimental import checkify
out = Path(os.environ['LOSS_DEBUG_OUT'])
checkpoint = Path(os.environ['SOURCE_CHECKPOINT'])
shutil.copy2(checkpoint, out / 'train.pkl')
with (out / 'train.pkl').open('rb') as f:
    saved = pickle.load(f)
print('REPLAY checkpoint update:', saved.get('update'), flush=True)
if saved.get('update') != 24:
    raise RuntimeError('Expected checkpoint 24; refusing a different replay')
original = m.jax.value_and_grad
def checked_value_and_grad(*args, **kwargs):
    differentiated = original(*args, **kwargs)
    checked = checkify.checkify(differentiated, errors=checkify.nan_checks)
    def run(*a, **kw):
        print('Tracing gradient with NaN checks', flush=True)
        error, result = m.jax.jit(checked)(*a, **kw)
        m.jax.block_until_ready(result)
        message = str(error.get())
        print('FIRST NAN:', message, flush=True)
        (out / 'first_nan.txt').write_text(message + '\n')
        error.throw()
        return result
    return run
m.jax.value_and_grad = checked_value_and_grad
m.main()
