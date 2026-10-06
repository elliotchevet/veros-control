"""Regression checks for horizon-independent ERA5 selection and year wrapping."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

source=Path(__file__).resolve().parents[1]/'setup/global_1deg_control/forcing_window.py'
spec=importlib.util.spec_from_file_location('forcing_window',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PERIOD=2920
INTERVAL=10800.

class ForcingWindowTests(unittest.TestCase):
    def evaluate(self, window, times):
        # Distinct records make a shift or erroneous subset wrap observable.
        annual=np.sin(np.arange(PERIOD)*.31)+np.arange(PERIOD)*.01
        records=(window['start']+np.arange(window['length']))%PERIOD
        buffer=annual[records]
        q=np.asarray(times)/INTERVAL
        n1=np.floor(q).astype(int)%PERIOD;n2=(n1+1)%PERIOD
        fraction=q-np.floor(q)
        i1,i2,valid=m.local_forcing_indices(n1,n2,window['start'],PERIOD,window['length'],np)
        self.assertTrue(valid.all())
        actual=(1-fraction)*buffer[i1]+fraction*buffer[i2]
        expected=(1-fraction)*annual[n1]+fraction*annual[n2]
        np.testing.assert_array_equal(actual,expected)
        return actual

    def test_january_short_and_month_windows_identical_prefix(self):
        short=m.plan_forcing_window(0,27,3600,INTERVAL,PERIOD)
        month=m.plan_forcing_window(0,720,3600,INTERVAL,PERIOD)
        self.assertEqual(short,dict(start=0,length=11,period_count=PERIOD))
        self.assertEqual(month,dict(start=0,length=242,period_count=PERIOD))
        times=np.arange(28)*3600.
        np.testing.assert_array_equal(self.evaluate(short,times),self.evaluate(month,times))
        self.evaluate(month,np.arange(721)*3600.)

    def test_fractional_start_and_year_boundary(self):
        start=(PERIOD-1)*INTERVAL+1800
        w=m.plan_forcing_window(start,27,3600,INTERVAL,PERIOD)
        self.assertEqual(w['start'],PERIOD-1)
        self.evaluate(w,start+np.arange(28)*3600.)

    def test_legacy_padded_buffer_maps_january_correctly(self):
        for length in (16,250):
            w=dict(start=2918,length=length,period_count=PERIOD)
            self.evaluate(w,np.arange(28)*3600.)
            i1,i2,valid=m.local_forcing_indices(0,1,2918,PERIOD,length,np)
            self.assertEqual((int(i1),int(i2),bool(valid)),(2,3,True))

    def test_exhausted_window_is_not_recycled(self):
        w=m.plan_forcing_window(0,27,3600,INTERVAL,PERIOD)
        _,_,valid=m.local_forcing_indices(10,11,0,PERIOD,w['length'],np)
        self.assertFalse(valid)

    def test_full_cycle_and_longer_horizon(self):
        w=m.plan_forcing_window(5000,9000,3600,INTERVAL,PERIOD)
        self.assertEqual(w['length'],PERIOD)
        self.evaluate(w,5000+np.arange(9001)*3600.)

if __name__=='__main__':unittest.main()
