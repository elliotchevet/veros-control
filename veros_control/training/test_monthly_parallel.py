"""Host transport checks; no GPU allocation needed."""
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
spec=importlib.util.spec_from_file_location('parallel',Path(__file__).with_name('monthly_parallel.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class TransportTests(unittest.TestCase):
    def test_atomic_roundtrip_and_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'message.pkl'
            payload={'weights':np.arange(8,dtype=np.float32),'month':12}
            thread=threading.Thread(target=m.atomic_pickle,args=(p,payload));thread.start()
            actual=m.await_pickle(p,timeout=2);thread.join()
            np.testing.assert_array_equal(actual['weights'],payload['weights'])
            self.assertEqual(actual['month'],12)
            self.assertFalse(p.exists())
            self.assertFalse(p.with_suffix('.pending').exists())

    def test_worker_recycles_between_requests_without_losing_results(self):
        class Process:
            returncode = None
            def poll(self): return self.returncode
            def wait(self, timeout): self.returncode = 0; return 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'worker'
            def response(path, process):
                # Simulate ready, then an echo response from each new worker.
                request = path.with_name(path.name.replace('response_', 'request_'))
                value = m.await_pickle_original(request)
                return (True, 'ready' if path.name == 'response_0.pkl' else value)
            m.await_pickle_original = m.await_pickle
            try:
                with patch.dict(m.os.environ, {'MONTHLY_WORKER_MAX_REQUESTS':'2'}), \
                     patch.object(m.subprocess, 'Popen', side_effect=lambda *a, **k: Process()), \
                     patch.object(m, 'await_pickle', side_effect=response):
                    worker = m.FileWorker(root, 'test-node', None, ('config',), 'h100')
                    self.assertEqual(worker.ready(), (True, 'ready'))
                    first = worker.process
                    for value in (10, 20, 30):
                        self.assertEqual(worker.request(value), (True, value))
                    self.assertEqual(first.returncode, 0)
                    self.assertIsNot(first, worker.process)
                    self.assertEqual(worker.generation, 1)
                    self.assertTrue((root/'generation_0'/'STOP').exists())
                    worker.close()
                    self.assertTrue((root/'generation_1'/'STOP').exists())
            finally:
                del m.await_pickle_original

    def test_shutdown_timeout_kills_only_the_recorded_step(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            worker = object.__new__(m.FileWorker)
            worker.root = worker.directory = Path(directory)
            (worker.directory/'slurm_step.txt').write_text('12345.7\n')
            worker.process = Mock()
            worker.process.wait.side_effect = [m.subprocess.TimeoutExpired('srun', 60), 0]
            with patch.object(m.subprocess, 'run') as run:
                worker.stop()
            run.assert_called_once_with(['scancel', '--signal=KILL', '12345.7'], check=True, timeout=30)
            worker.process.kill.assert_not_called()
            self.assertTrue((worker.directory/'STOP').exists())

    def test_shutdown_refuses_allocation_wide_cancel(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            worker = object.__new__(m.FileWorker)
            worker.root = worker.directory = Path(directory)
            (worker.directory/'slurm_step.txt').write_text('12345')
            worker.process = Mock()
            worker.process.wait.side_effect = m.subprocess.TimeoutExpired('srun', 60)
            with patch.object(m.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError): worker.stop()
            run.assert_not_called()

    def test_dead_worker_is_not_waited_on(self):
        class Dead:
            returncode=1
            def poll(self): return 1
        with self.assertRaises(RuntimeError):
            m.await_pickle(Path('/tmp/absent-monthly-worker-response'),Dead(),timeout=1)

if __name__=='__main__': unittest.main()
