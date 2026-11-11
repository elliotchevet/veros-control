"""Read chronological ERA5 years onto 3-hour, 360-day model-month windows.

Each Gregorian month (including leap February) maps linearly onto 30 model
 days. Flux values retain their original units; accumulated one-hour values
are divided by 3600 by the existing setup, not by the stretched model interval.
"""
import calendar
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
import h5netcdf
import numpy as np

VARIABLES = ('ewss','nsss','str','slhf','sshf','ssr','e','tp')
PERIOD_COUNT = 2880  # 360 days, one sample every 3 model hours

def month_hours(year, month):
    start = (datetime(year,month,1)-datetime(year,1,1)).total_seconds()/3600
    return start + np.linspace(0,calendar.monthrange(year,month)[1]*24,241)

@contextmanager
def sequence_context(simulation, buffers, index_start):
    """Pass forcing as dynamic JAX arguments; never leave traced arrays cached."""
    old_buffers=getattr(simulation,'_era5_sequence_buffers',None)
    old_start=getattr(simulation,'_era5_sequence_index_start',None)
    simulation._era5_sequence_buffers=dict(zip(VARIABLES,buffers))
    simulation._era5_sequence_index_start=index_start
    try:
        yield
    finally:
        simulation._era5_sequence_buffers=old_buffers
        simulation._era5_sequence_index_start=old_start

class ERA5Sequence:
    def __init__(self, directory, first_year, last_year):
        self.directory=Path(directory);self.first_year=first_year;self.last_year=last_year
        if last_year<first_year:raise ValueError('Invalid year sequence')
        self.manifest=[];self.coordinates=None
        for year in range(first_year,last_year+1):
            path=self.path(year)
            with h5netcdf.File(path) as f:
                n=(366 if calendar.isleap(year) else 365)*24
                hours=np.asarray(f['time'][:]);attrs=f['time'].attrs
                epoch=str(attrs['units']).split('hours since ',1)
                if len(epoch)!=2:raise ValueError(f'Unsupported time units: {attrs}')
                first=datetime.fromisoformat(epoch[1])+timedelta(hours=float(hours[0]))
                if first!=datetime(year,1,1) or len(hours)!=n or not np.allclose(np.diff(hours),1):raise ValueError(f'Incomplete hourly year {year}')
                coords=tuple(np.asarray(f[k][:]) for k in ('lon','lat'))
                if self.coordinates is None:self.coordinates=coords
                if any(not np.array_equal(a,b) for a,b in zip(coords,self.coordinates)):raise ValueError('ERA5 grid changes between years')
                for k in VARIABLES:
                    expected_dims=('time','lat','lon_u') if k=='ewss' else (('time','lat_u','lon') if k=='nsss' else ('time','lat','lon'))
                    if f[k].dimensions!=expected_dims or f[k].shape!=(n,len(coords[1]),len(coords[0])):raise ValueError(f'Wrong dimensions: {year} {k}')
                    for axis in ['lon_u','lat_u']:
                        if axis in f[k].dimensions:
                            base=coords[0] if axis=='lon_u' else coords[1]
                            if not np.allclose(f[axis][:],base+.5):raise ValueError(f'Unexpected staggered coordinates: {year} {axis}')
                self.manifest.append(dict(year=year,path=str(path),hours=n,size_bytes=path.stat().st_size,time_units=str(attrs['units']),wind_dimensions={k:list(f[k].dimensions) for k in ('ewss','nsss')}))
    def path(self,year):return self.directory/f'ERA5_1deg_{year}_Flux_processed.nc'
    def load_month(self,year,month):
        if not self.first_year<=year<=self.last_year or not 1<=month<=12:raise ValueError('Outside forcing sequence')
        positions=month_hours(year,month);n=(366 if calendar.isleap(year) else 365)*24
        lo=np.floor(positions).astype(int);hi=np.ceil(positions).astype(int);weight=positions-lo
        indices=np.unique(np.r_[lo,hi]);current=indices[indices<n]
        result=[]
        with h5netcdf.File(self.path(year)) as f:
            for name in VARIABLES:
                values=np.asarray(f[name][current.tolist(),:,:],dtype=np.float64)
                if indices[-1]>=n:
                    if year<self.last_year:
                        with h5netcdf.File(self.path(year+1)) as following:last=np.asarray(following[name][0],dtype=np.float64)
                    else:
                        # Only the terminal endpoint is absent: persist Dec 31 23:00,
                        # never wrap to January of an earlier year.
                        last=np.asarray(f[name][-1],dtype=np.float64)
                    values=np.concatenate((values,last[None]),axis=0)
                z=(1-weight[:,None,None])*values[np.searchsorted(indices,lo)]+weight[:,None,None]*values[np.searchsorted(indices,hi)]
                if not np.isfinite(z).all():raise ValueError(f'Nonfinite forcing {year}-{month} {name}')
                result.append(z.transpose(2,1,0).copy())
        return tuple(result),(month-1)*240
