"""Chronological forcing: leap days, year boundaries, and dynamic buffer cleanup."""
import calendar,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
import h5netcdf
from veros_control.data import era5_sequence as m
class SequenceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
  for year in range(1993,1997):
   n=(366 if calendar.isleap(year) else 365)*24
   with h5netcdf.File(cls.root/f'ERA5_1deg_{year}_Flux_processed.nc','w') as f:
    f.dimensions={'time':n,'lat':1,'lon':1,'lat_u':1,'lon_u':1}
    t=f.create_variable('time',('time',),'f8');t[:]=np.arange(n);t.attrs['units']=f'hours since {year}-01-01 00:00:00'
    for k in ['lon','lat']:f.create_variable(k,(k,),'f8')[:]=[0]
    for k in ['lon_u','lat_u']:f.create_variable(k,(k,),'f8')[:]=[.5]
    for k in m.VARIABLES:
     dims=('time','lat','lon_u') if k=='ewss' else (('time','lat_u','lon') if k=='nsss' else ('time','lat','lon'))
     f.create_variable(k,dims,'f8')[:]=(np.arange(n)+10000*(year-1993))[:,None,None]
  cls.seq=m.ERA5Sequence(cls.root,1993,1996)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_leap_month(self):
  z,i=self.seq.load_month(1996,2);self.assertEqual(i,240)
  np.testing.assert_allclose(z[0][0,0],30000+m.month_hours(1996,2));self.assertEqual(z[0][0,0,-1],30000+60*24)
 def test_year_boundary(self):
  dec,_=self.seq.load_month(1993,12);jan,_=self.seq.load_month(1994,1)
  self.assertEqual(dec[0][0,0,-1],jan[0][0,0,0]);self.assertEqual(dec[0][0,0,-1],10000)
  self.assertEqual(jan[0][0,0,0],10000)
 def test_short_window_crosses_year_without_wrap(self):
  from datetime import datetime
  from veros_control.data.dated import load_window,forcing_positions
  start=datetime(1993,12,31)
  values,index=load_window(self.root,start,81,3600)
  expected_index,positions=forcing_positions(start,81,3600)
  self.assertEqual(index,expected_index)
  expected=np.asarray([hour+10000*(year-1993) for year,hour in positions])
  np.testing.assert_allclose(values[0][0,0],expected)
  self.assertEqual({year for year,hour in positions},{1993,1994})
 def test_missing_year_fails(self):
  with self.assertRaises(FileNotFoundError):m.ERA5Sequence(self.root,1993,1997)
 def test_final_endpoint_and_context(self):
  z,_=self.seq.load_month(1996,12);self.assertEqual(z[0][0,0,-1],30000+8783)
  class Sim:pass
  sim=Sim()
  with m.sequence_context(sim,z,2640):self.assertIs(sim._era5_sequence_buffers['ewss'],z[0])
  self.assertIsNone(sim._era5_sequence_buffers)
if __name__=='__main__':unittest.main()
