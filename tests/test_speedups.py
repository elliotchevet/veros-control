import importlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from multiprocessing.connection import Client
import numpy as np
from veros_control.training.config import TrainConfig, CurriculumStage, SchedulerConfig
from veros_control.training.scheduler import loss_test_next_learning_rate
from veros_control.training.workers import Worker


class SpeedupTests(unittest.TestCase):
    def test_smooth_curriculum_uses_best_and_stage_rates(self):
        module=importlib.import_module('veros_control.training.train')
        config=TrainConfig('out','setup:create','initial','target',transition='27/best.pkl',
            transition_checkpoint='best',fixed_loss_balance=True,
            scheduler=SchedulerConfig(mode='keep_update',start_update=1,recovery_interval=2,reject_worsening=True),
            curriculum=[CurriculumStage(h,200,1e-5 if h==243 else 1e-4) for h in (54,81,135,189,243)])
        seen=[]
        def stage(cfg):seen.append(cfg);return Path(cfg.output)/'latest.pkl'
        with patch.object(module,'train_stage',side_effect=stage):module.train(config)
        self.assertEqual(seen[0].transition,'27/best.pkl')
        for before,after in zip(seen,seen[1:]):
            self.assertEqual(after.transition,str(Path(before.output)/'best.pkl'))
        self.assertEqual(seen[-1].stage_initial_rate,1e-5)
        self.assertTrue(all(c.fixed_loss_balance and c.updates==200 for c in seen))

    def test_two_improvement_recovery_and_floor(self):
        rate,report=loss_test_next_learning_rate(1e-4,1.,2.,1,min_rate=1e-5,start_epoch=1,recovery_interval=2)
        self.assertEqual(rate,5e-5)
        for epoch in (2,3):rate,report=loss_test_next_learning_rate(rate,1.,.9,epoch,min_rate=1e-5,start_epoch=1,recovery_interval=2,clean_updates=report['lr_clean_updates'])
        self.assertAlmostEqual(rate,7.5e-5)
        rate,_=loss_test_next_learning_rate(1e-5,1.,2.,4,min_rate=1e-5,start_epoch=1)
        self.assertEqual(rate,1e-5)

    def test_worker_timeout_and_error_propagate(self):
        worker=object.__new__(Worker);worker.timeout=1;worker.connection=Mock()
        worker.connection.poll.return_value=False
        with self.assertRaises(TimeoutError):worker.receive()
        worker.connection.poll.return_value=True
        worker.connection.recv.return_value=(False,'remote failure')
        with self.assertRaisesRegex(RuntimeError,'remote failure'):worker.receive()

    def test_tcp_roundtrip_reuses_process_and_closes(self):
        threads=[]
        class Process:
            def wait(self,timeout):
                threads[-1].join(timeout)
                if threads[-1].is_alive():raise RuntimeError('peer did not stop')
        def launch(cmd,**kwargs):
            host,port,keyfile=cmd[-3:]
            def peer():
                with Client(('127.0.0.1',int(port)),authkey=Path(keyfile).read_bytes()) as c:
                    c.recv();c.send((True,'ready'))
                    while True:
                        x=c.recv()
                        if x is None:break
                        c.send((True,x))
            t=threading.Thread(target=peer,daemon=True);threads.append(t);t.start();return Process()
        with tempfile.TemporaryDirectory() as directory,patch('subprocess.Popen',side_effect=launch) as popen:
            w=Worker([],Path(directory)/'worker',('config',),3)
            try:
                w.ready()
                for i in range(3):np.testing.assert_array_equal(w.request(np.arange(100)+i),np.arange(100)+i)
                self.assertEqual(popen.call_count,1)
                self.assertEqual((Path(directory)/'worker/auth.key').stat().st_mode & 0o777,0o600)
            finally:w.close()

if __name__=='__main__':unittest.main()
