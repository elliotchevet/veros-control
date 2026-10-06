"""Regression checks for paired best/latest snapshots and optimizer alignment."""
import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('best_checkpoint',Path(__file__).resolve().parents[1]/'veros_control/core/checkpoint.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class BestCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,prefix='.checkpoint-test-')
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'train.pkl'
        self.writer=m.BestLatestCheckpoint(self.path)
    def payload(self, update):
        return dict(update=update,params={'weight':update},opt_state={'step':update},train_history=[{'update':i} for i in range(1,update+1)])
    def read(self,path):
        with path.open('rb') as stream:return pickle.load(stream)
    def test_improve_regress_and_resume(self):
        self.assertTrue(self.writer.save(self.payload(80),.83,{'temp':.8,'salt':.86}))
        self.assertTrue(self.writer.save(self.payload(81),.81,{'temp':.8,'salt':.82}))
        self.assertFalse(self.writer.save(self.payload(82),.95,{'temp':1.,'salt':.9}))
        best=self.read(m.best_path(self.path));last=self.read(self.path)
        self.assertEqual((best['update'],best['params']['weight'],best['opt_state']['step']),(81,81,81))
        self.assertEqual(last['update'],82);self.assertEqual(last['best_update'],81)
        self.assertEqual(best['checkpoint_loss'],.81)
        resumed=m.BestLatestCheckpoint(self.path)
        self.assertFalse(resumed.save(self.payload(83),.82,{}))
        self.assertEqual(self.read(m.best_path(self.path))['update'],81)
        self.assertTrue(resumed.save(self.payload(84),.79,{}))
        self.assertEqual(self.read(m.best_path(self.path))['update'],84)
    def test_shared_file_not_overwritten_by_worse_update(self):
        self.writer.save(self.payload(80),.83,{})
        self.assertTrue(self.path.samefile(m.best_path(self.path)))
        self.writer.save(self.payload(81),.9,{})
        self.assertFalse(self.path.samefile(m.best_path(self.path)))
        self.assertEqual(self.read(m.best_path(self.path))['params']['weight'],80)
    def test_nonfinite_and_history_mutation_leave_best_intact(self):
        payload=self.payload(80)
        self.writer.save(payload,.83,{})
        payload['train_history'].append({'update':81})
        self.assertEqual(len(self.writer.best_payload['train_history']),80)
        with self.assertRaises(FloatingPointError):self.writer.save(self.payload(81),float('nan'),{})
        self.assertEqual(self.read(self.path)['update'],80)
        self.assertEqual(self.read(m.best_path(self.path))['update'],80)

if __name__=='__main__':unittest.main()
