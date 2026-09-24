"""Run the real training entrypoint with non-retaining GPU memory snapshots."""
import builtins
from collections import Counter
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import hashlib
import subprocess

source = Path(__file__).resolve().parents[2] / 'training/ad_benchmark_root.py'
sys.argv[0] = str(source)
spec = importlib.util.spec_from_file_location('memory_benchmark', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
jax = module.jax
out = Path(os.environ['MEMORY_DEBUG_OUT'])
out.mkdir(parents=True, exist_ok=True)
# Preserve exact source fingerprints for every submitted comparison.
source_paths = [source, source.with_name('ad_benchmark.py'), source.with_name('train.py'),
                Path('/Odyssey/private/e25cheve/veros/veros/core/external/solvers/__init__.py')]
(out / 'source_hashes.json').write_text(json.dumps(
    {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}, indent=2))
original_print = builtins.print
previous_ids = set()
start = time.monotonic()

def snapshot(label):
    global previous_ids
    gc.collect()
    arrays = jax.live_arrays()
    shapes = Counter()
    origins = Counter()
    ids = set()
    total = 0
    for array in arrays:
        ids.add(id(array))
        nbytes = array.nbytes
        total += nbytes
        if id(array) not in previous_ids:
            shapes[str((array.shape, str(array.dtype)))] += nbytes
            if nbytes >= 100000:
                origin = str(getattr(array, 'traceback', 'unknown'))
                origins[origin] += nbytes
    previous_ids = ids
    del arrays
    stats = jax.devices()[0].memory_stats() or {}
    from veros.core.external.solvers import get_linear_solver
    cache = getattr(get_linear_solver, 'cache', {})
    stats['solver_cache_entries'] = len(cache)
    try:
        stats['nvidia_process_memory'] = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    result = dict(label=label, seconds=time.monotonic()-start, arrays=len(ids),
                  array_bytes=total, device=stats, new_shapes=shapes.most_common(10),
                  new_origins=origins.most_common(5))
    with (out / 'memory.jsonl').open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    original_print('MEMORY', json.dumps({key: result[key] for key in ('label', 'arrays', 'array_bytes', 'device')}), flush=True)

old_g = module.NNParamsTrainChunkedVJP.g
calls = 0
def profiled_g(self, *args, **kwargs):
    global calls
    calls += 1
    tag = f'g{calls}' + ('_calibration' if kwargs.get('forward_only') else '')
    snapshot(tag + '_before')
    result = old_g(self, *args, **kwargs)
    jax.block_until_ready(result)
    if not kwargs.get('forward_only'):
        # Host-only comparison data: no device references survive this call.
        import pickle
        with (out / f'{tag}_loss_grad.pkl').open('wb') as stream:
            pickle.dump(jax.device_get(result), stream)
    snapshot(tag + '_after')
    return result
module.NNParamsTrainChunkedVJP.g = profiled_g
old_diagnostics = module.tendency_ratio_diagnostics
def diagnostics(*args, **kwargs):
    snapshot('diagnostics_before')
    result = old_diagnostics(*args, **kwargs)
    if os.environ.get('MEMORY_CLEAR_SOLVE_KERNEL') == '1':
        from veros.core.external.solvers.scipy_jax import solve_kernel
        compiled = solve_kernel.function
        if hasattr(compiled, 'clear_cache'):
            compiled.clear_cache()
    snapshot('diagnostics_after')
    return result
module.tendency_ratio_diagnostics = diagnostics
updates = 0
def debug_print(*args, **kwargs):
    global updates
    original_print(*args, **kwargs)
    if args and isinstance(args[0], str) and args[0].startswith('  memory after update:'):
        updates += 1
        snapshot(f'update{updates}_end')
        if updates == int(os.environ.get('MEMORY_CLEAR_AT', '-1')):
            jax.clear_caches()
            snapshot(f'update{updates}_after_clear_caches')
builtins.print = debug_print
if os.environ.get('MEMORY_DIAGNOSTICS_ONLY') == '1':
    import inspect
    old_train = module.run_train_mode
    def diagnostics_only(*args, **kwargs):
        values = inspect.signature(old_train).bind(*args, **kwargs).arguments
        sim, cfg = values['simulation'], values['args']
        sim.prepare_ERA5_forcing_window(sim.state, module._resolve_train_n_steps(cfg, values['steps_list']))
        for index in range(int(os.environ.get('MEMORY_DIAGNOSTIC_REPEATS', '3'))):
            module.tendency_ratio_diagnostics(sim, values['nn_params'], values['model_batched_unscaled'],
                                             values['nn_stats'], cfg.epsilon)
            if index == 0 and os.environ.get('MEMORY_REFERRERS') == '1':
                arrays = jax.live_arrays()
                candidates = [a for a in arrays if a.nbytes > 10000000 and 'tendency_ratio_diagnostics' in str(a.traceback)]
                if candidates:
                    target = candidates[0]
                    del arrays, candidates
                    seen = set()
                    def describe_refs(obj, depth=0):
                        if depth >= 5 or id(obj) in seen:
                            return
                        seen.add(id(obj))
                        refs = gc.get_referrers(obj)
                        for ref in refs[:12]:
                            description = type(ref).__name__
                            if isinstance(ref, dict):
                                description += ' keys=' + str([str(k) for k,v in ref.items() if v is obj][:8])
                            elif isinstance(ref, (tuple, list)):
                                description += ' length=' + str(len(ref))
                            original_print('REF', depth, description, flush=True)
                            if not isinstance(ref, list):
                                describe_refs(ref, depth+1)
                    describe_refs(target)
                    del target
            if index + 1 == int(os.environ.get('MEMORY_CLEAR_AT', '-1')):
                jax.clear_caches()
                snapshot('diagnostics_after_clear_caches')
    module.run_train_mode = diagnostics_only
try:
    module.main()
finally:
    builtins.print = original_print
    snapshot('finished')
