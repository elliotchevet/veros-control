"""Compare training results, excluding elapsed times, for a memory-only fix."""
import argparse
import json
import os
from pathlib import Path
import pickle

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = 'false'
os.environ['TMPDIR'] = str(Path(__file__).resolve().parent)
import jax
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('before')
parser.add_argument('after')
parser.add_argument('--output', required=True)
args = parser.parse_args()
with open(args.before, 'rb') as stream:
    before = pickle.load(stream)
with open(args.after, 'rb') as stream:
    after = pickle.load(stream)
assert before['update'] == after['update']
report = {'update': before['update'], 'groups': {}, 'all_close': True}

def compare(name, left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree, name
    exact, close, max_error = True, True, 0.
    for a, b in zip(left_leaves, right_leaves):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, name
        exact = exact and np.array_equal(a, b, equal_nan=True)
        close = close and np.allclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=True)
        finite = np.isfinite(a) & np.isfinite(b)
        if np.any(finite):
            max_error = max(max_error, float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))))
    report['groups'][name] = dict(leaves=len(left_leaves), bitwise_equal=bool(exact),
                                  all_close=bool(close), max_absolute_error=max_error)
    report['all_close'] = report['all_close'] and bool(close)

compare('params', before['params'], after['params'])
compare('optimizer', before['opt_state'], after['opt_state'])
compare('loss_denominators', before['loss_term_denominators'], after['loss_term_denominators'])
left_history = [{k: v for k,v in row.items() if k != 'time'} for row in before['train_history']]
right_history = [{k: v for k,v in row.items() if k != 'time'} for row in after['train_history']]
compare('history_including_losses_and_gradient_statistics', left_history, right_history)
Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['all_close'] else 1)
