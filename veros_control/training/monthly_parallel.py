"""Persistent isolated monthly workers sharing the job's visible GPU(s).

Only host arrays cross the pipes. Each worker owns its Veros simulation and
solver cache; the coordinator alone performs the averaged Adam update.
"""
import atexit
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
import re
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
import traceback
import time


def atomic_pickle(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('wb') as f: pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def await_pickle(path, process=None, timeout=3600):
    started = time.monotonic()
    while not path.exists():
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'Worker exited with code {process.returncode}; see its log')
        if time.monotonic() - started > timeout:
            raise TimeoutError(f'Worker response timed out: {path}')
        time.sleep(.1)
    with path.open('rb') as f: value = pickle.load(f)
    path.unlink()
    return value


class FileWorker:
    def __init__(self, directory, node, log, initial, gpu_type, het_group=None):
        self.root = directory
        self.node, self.log, self.initial = node, log, initial
        self.gpu_type, self.het_group = gpu_type, het_group
        self.generation = 0
        self.max_requests = int(os.environ.get('MONTHLY_WORKER_MAX_REQUESTS', '12'))
        if self.max_requests < 1:
            raise ValueError('MONTHLY_WORKER_MAX_REQUESTS must be positive')
        self.start()

    def start(self):
        directory = self.root / f'generation_{self.generation}'
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.sequence = 0
        atomic_pickle(directory/'request_0.pkl', self.initial)
        node, log, gpu_type, het_group = self.node, self.log, self.gpu_type, self.het_group
        command = ['srun', '--exclusive', '--exact', '--nodes=1', '--ntasks=1',
                   '--cpus-per-task=8', '--mem='+os.environ.get('MONTHLY_WORKER_MEMORY', '64G'), '--gres=gpu:'+gpu_type+':1', '--nodelist='+node,
                   sys.executable, '-B', str(Path(__file__).resolve()), '--files', str(directory)]
        if het_group is not None: command.insert(1, f'--het-group={het_group}')
        self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

    def ready(self):
        result = await_pickle(self.directory/'response_0.pkl', self.process)
        self.sequence = 1
        return result

    def request(self, value):
        if self.sequence > self.max_requests:
            self.stop()
            self.generation += 1
            print(f'WORKER recycle: {self.root.name} generation={self.generation} '
                  f'after {self.max_requests} requests; releasing host/GPU caches', flush=True)
            self.start()
            ok, result = self.ready()
            if not ok:
                raise RuntimeError(f'Restarted worker failed: {result}')
        sequence = self.sequence
        atomic_pickle(self.directory/f'request_{sequence}.pkl', value)
        result = await_pickle(self.directory/f'response_{sequence}.pkl', self.process)
        self.sequence += 1
        return result

    def close(self):
        (self.directory/'STOP').touch()

    def stop(self):
        """Stop only this idle worker step; a slow teardown must not abort training."""
        self.close()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            step = (self.directory/'slurm_step.txt').read_text().strip()
            if re.fullmatch(r'[0-9]+\.[0-9]+', step) is None:
                raise RuntimeError(f'Refusing invalid worker step identity: {step!r}')
            print(f'WORKER shutdown timeout: {self.root.name}; killing idle step {step}', flush=True)
            subprocess.run(['scancel', '--signal=KILL', step], check=True, timeout=30)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # scancel has already targeted the remote step; reap a stuck launcher.
                self.process.kill()
                self.process.wait(timeout=30)



class MonthlyWorkers:
    def __init__(self, cfg, files, batch_size, seed, score_months, max_cfl):
        import jax
        self.jax = jax
        self.processes = []
        self.pool = ThreadPoolExecutor(max_workers=cfg.monthly_parallel_workers)
        directory = Path(cfg.train_output_file).parent / 'workers'
        directory.mkdir(parents=True, exist_ok=True)
        config = {k:v for k,v in vars(cfg).items() if not callable(v)}
        config['monthly_parallel_workers'] = 0
        config = jax.device_get(config)
        nodes = None
        if getattr(cfg, 'monthly_worker_nodes', False):
            nodes = []
            groups = int(os.environ.get('SLURM_HET_SIZE', '1'))
            types = os.environ.get('MONTHLY_GPU_TYPES', 'h100').split(':')
            for group in range(groups):
                node_list = os.environ.get(f'SLURM_JOB_NODELIST_HET_GROUP_{group}', os.environ['SLURM_JOB_NODELIST'])
                hosts = subprocess.check_output(['scontrol', 'show', 'hostnames', node_list], text=True).split()
                gpu_type, _, count = types[group].partition('x')
                nodes.extend((node, gpu_type, group if groups > 1 else None) for node in hosts for _ in range(int(count or 1)))
        if nodes is not None and len(nodes) < cfg.monthly_parallel_workers:
            raise ValueError('Need one allocated node per monthly worker')
        for index in range(cfg.monthly_parallel_workers):
            if nodes is not None:
                log = open(directory / f'worker_{index}.log', 'wb')
                remote = FileWorker(directory/f'node_{index}', nodes[index][0], log,
                    (config, files, batch_size, seed, score_months, max_cfl), nodes[index][1], nodes[index][2])
                print(f'WORKER {index}: node={nodes[index][0]} GPU={nodes[index][1]} one batch member', flush=True)
                self.processes.append((remote.process, remote, log))
                continue
            read_fd, write_fd = os.pipe()
            log = open(directory / f'worker_{index}.log', 'wb')
            process = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), str(write_fd)],
                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, pass_fds=(write_fd,))
            os.close(write_fd)
            stream = os.fdopen(read_fd, 'rb')
            self.processes.append((process, stream, log))
            pickle.dump((config, files, batch_size, seed, score_months, max_cfl), process.stdin)
            process.stdin.flush()
        atexit.register(self.close)
        for _, stream, _ in self.processes:
            ok, result = stream.ready() if isinstance(stream, FileWorker) else pickle.load(stream)
            if not ok:
                raise RuntimeError(result)
        print(f'WORKERS ready: {len(self.processes)} independent processes; logs in {directory}', flush=True)

    def request(self, worker, operation, month, params, options, denominators):
        process, stream, _ = worker
        value = (operation, month, params, options, denominators)
        if isinstance(stream, FileWorker):
            ok, result = stream.request(value)
        else:
            pickle.dump(value, process.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            process.stdin.flush()
            ok, result = pickle.load(stream)
        if not ok:
            raise RuntimeError(f'Month {month} worker failed:\n{result}')
        return result

    def map(self, operation, months, params, options, denominators):
        params, options, denominators = self.jax.device_get((params, options, denominators))
        results = []
        # Waves avoid sending two requests to the same worker concurrently.
        for offset in range(0, len(months), len(self.processes)):
            futures = [self.pool.submit(self.request, worker, operation, month, params, options, denominators)
                       for worker, month in zip(self.processes, months[offset:offset+len(self.processes)])]
            results.extend(f.result() for f in futures)
        return results

    def close(self):
        for process, stream, log in self.processes:
            if isinstance(stream, FileWorker):
                stream.close()
            elif process.stdin is not None:
                process.stdin.close()
        for process, stream, log in self.processes:
            if isinstance(stream, FileWorker):
                stream.stop()
                process = stream.process
            if process.poll() is None:
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill()
            if not isinstance(stream, FileWorker): stream.close()
            log.close()
        self.processes = []
        self.pool.shutdown(wait=False)


def worker(response_fd=None, directory=None):
    sequence = 0
    if directory is not None:
        # The actual step identity accounts for heterogeneous Slurm components.
        step = f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"
        (directory/'slurm_step.txt').write_text(step+'\n')
    response = os.fdopen(response_fd, 'wb') if directory is None else None
    def receive():
        if directory is None: return pickle.load(sys.stdin.buffer)
        path = directory/f'request_{sequence}.pkl'
        while not path.exists():
            if (directory/'STOP').exists(): raise EOFError
            time.sleep(.1)
        with path.open('rb') as f: value=pickle.load(f)
        path.unlink()
        return value
    def send(ok, value):
        nonlocal sequence
        if directory is None:
            pickle.dump((ok, value), response, protocol=pickle.HIGHEST_PROTOCOL);response.flush()
        else:
            atomic_pickle(directory/f'response_{sequence}.pkl', (ok,value))
            sequence += 1
    try:
        config, files, batch_size, seed, score_months, max_cfl = receive()
        source = Path(__file__).with_name('ad_benchmark_root.py')
        sys.argv = [str(source), '--diskless-mode']
        spec = importlib.util.spec_from_file_location('monthly_worker_benchmark', source)
        m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        cfg = SimpleNamespace(**config)
        sim = m.init_simulation(0, glorys_file=files[1][0], glorys_time_index=0,
            glorys_month_index=0, glorys_start_month=1,
            era5_forcing_file=cfg.monthly_initial_states['forcing_file'], setup_forcing_window=True)
        sim.era5_forcing_file = cfg.monthly_initial_states['forcing_file']
        _, raw_model, stats, _ = m.init_nn_benchmark(sim, cfg)
        from veros_control.training.stability import bound_corrections
        from veros_control.training.monthly_batches import MonthlyRollouts
        eps=m.jnp.asarray([cfg.epsilon]*2+[cfg.monthly_momentum_epsilon]*2)
        def model(params, features, coordinates, dzt):
            raw=raw_model(params,features,coordinates,dzt)
            if cfg.monthly_correction_bounds is None:
                return raw * (eps/cfg.epsilon)
            return bound_corrections(raw,eps,cfg.monthly_correction_bounds)/cfg.epsilon
        single = m.NNParamsTrainChunkedVJP(sim,model,stats,None)
        adapter = MonthlyRollouts(m,single,cfg,files,batch_size,seed,score_months,max_cfl)
        send(True, 'ready')
        while True:
            operation, month, params, options, denominators = receive()
            params=m.jax.tree_util.tree_map(m.jnp.asarray,params)
            adapter.loss_term_denominators=m.jnp.asarray(denominators)
            if operation == 'g':
                with adapter.member(month) as ad:
                    result=ad.g(sim.state,params,**options)
            elif operation == 'validate':
                result=adapter.validate_member(month,params)
            else: raise ValueError(operation)
            send(True,m.jax.device_get(result))
    except EOFError:
        # All responses have been written and copied to host before STOP is sent.
        # Avoid hanging in GPU/MPI interpreter finalizers in a disposable worker.
        if response is not None:
            response.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException:
        send(False,traceback.format_exc())
        raise


if __name__ == '__main__':
    if sys.argv[1] == '--files':
        worker(directory=Path(sys.argv[2]))
    else:
        worker(int(sys.argv[1]))
