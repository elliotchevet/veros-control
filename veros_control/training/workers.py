"""Persistent authenticated workers; commands may be local or Slurm launch prefixes."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing.connection import Client, Listener
from pathlib import Path
import os
import socket
import subprocess
import sys
import traceback


class Worker:
    def __init__(self, command, directory, initial, timeout):
        self.timeout = timeout
        self.connection = None
        self.process = None
        self.log = None
        directory.mkdir(parents=True)
        key = os.urandom(32)
        keyfile = directory/'auth.key'
        fd = os.open(keyfile, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(key)
        self.listener = Listener(('', 0), authkey=key)
        self.listener._listener._socket.settimeout(timeout)
        self.initial = initial
        self.log = (directory/'worker.log').open('wb')
        try:
            self.process = subprocess.Popen(command + [sys.executable, '-m',
                'veros_control.training.workers', socket.gethostname(),
                str(self.listener.address[1]), str(keyfile)], stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.close()
            raise

    def ready(self):
        self.connection = self.listener.accept()
        self.listener.close()
        self.connection.send(self.initial)
        self.receive()

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise TimeoutError(f'Worker did not respond within {self.timeout}s; see its log')
        ok, result = self.connection.recv()
        if not ok:
            raise RuntimeError(result)
        return result

    def request(self, message):
        self.connection.send(message)
        return self.receive()

    def close(self):
        if self.connection is not None:
            try:
                self.connection.send(None)
            except (OSError, EOFError):
                pass
            self.connection.close()
        self.listener.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.log is not None:
            self.log.close()


class ParallelObjective:
    def __init__(self, local):
        self.local = local
        self.config = local.config
        self.months, self.panel = local.months, local.panel
        self.metadata = local.metadata
        self.workers = []
        self.pool = ThreadPoolExecutor(max_workers=len(self.config.worker_commands))

    def member(self, month):
        self.local.member(month)
        self.names = self.local.names
        return self

    def start(self):
        if self.workers:
            return
        import jax
        initial = (replace(self.config, worker_commands=[]), jax.device_get(self.local.stats))
        try:
            for index, command in enumerate(self.config.worker_commands):
                self.workers.append(Worker(command, Path(self.config.output)/'workers'/str(index),
                    initial, self.config.worker_timeout))
            for worker in self.workers:
                worker.ready()
        except BaseException:
            self.close()
            raise

    def dispatch(self, operation, months, params, scale=None):
        import jax
        self.start()
        params, scale = jax.device_get((params, scale))
        groups = [tuple(months[i::len(self.workers)]) for i in range(len(self.workers))]
        futures = [(group, self.pool.submit(worker.request, (operation, group, params, scale)))
                   for worker, group in zip(self.workers, groups) if group]
        results = []
        for group, future in futures:
            value, metadata = future.result()
            self.metadata.update(metadata)
            results.append((len(group), value))
        return results

    def mean(self, months, params, scale, gradient=False):
        import jax
        import jax.numpy as jnp
        results = self.dispatch('gradient' if gradient else 'score', months, params, scale)
        total = None
        for count, value in results:
            weighted = jax.tree_util.tree_map(lambda x:jnp.asarray(x)*count/len(months), value)
            total = weighted if total is None else jax.tree_util.tree_map(lambda a,b:a+b,total,weighted)
        return total

    def validate(self, months, params):
        valid, reports = True, {}
        for _, (ok, report) in self.dispatch('validate', months, params):
            valid &= ok
            reports.update(report)
        return valid, reports

    def close(self):
        self.pool.shutdown(wait=True)
        for worker in self.workers:
            worker.close()
        self.workers.clear()
        self.local.close()


def main():
    import jax
    from veros_control.core.model import initialize
    from veros_control.core.correction import physical_model
    from veros_control.data.dataset import create_simulation
    from veros_control.training.batches import BatchObjective, case_configs
    host, port, keyfile = sys.argv[1:]
    with Client((host, int(port)), authkey=Path(keyfile).read_bytes()) as connection:
        try:
            config, stats = connection.recv()
            simulation, _ = create_simulation(next(iter(case_configs(config).values())))
            model, _ = initialize(config.model, simulation.state.settings.nz, config.seed)
            objective = BatchObjective(config, simulation,
                physical_model(model, config.correction, simulation.state.variables.zt), stats)
            connection.send((True, 'ready'))
            while True:
                message = connection.recv()
                if message is None:
                    break
                operation, months, params, scale = message
                params = jax.tree_util.tree_map(jax.numpy.asarray, params)
                if operation == 'validate':
                    value = objective.validate(months, params)
                elif operation in ('score', 'gradient'):
                    value = objective.mean(months, params, jax.numpy.asarray(scale), gradient=operation=='gradient')
                else:
                    raise ValueError(f'Unknown worker operation: {operation}')
                connection.send((True, (jax.device_get(value), objective.metadata)))
        except BaseException:
            connection.send((False, traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
