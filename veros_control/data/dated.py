"""Chronological monthly training samples and short ERA5 windows, no year looping."""
import calendar
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import h5netcdf
from veros_control.data.era5_sequence import VARIABLES

MODEL_YEAR = 360 * 86400

def model_timestamp(date):
    start = datetime(date.year, date.month, 1)
    fraction = (date-start).total_seconds() / (calendar.monthrange(date.year,date.month)[1]*86400)
    return ((date.year-1993)*360+(date.month-1+fraction)*30)*86400

def dated_files(directory, first, last, forcing_directory=None):
    if last < first: raise ValueError('Invalid training-year range')
    root=Path(directory)
    def path(y,m):
        p=root/f'GLORYS12_1deg_{y}{m:02d}_processed.nc'
        if not p.is_file(): raise FileNotFoundError(p)
        return str(p.resolve())
    pairs={}
    for year in range(first,last+1):
        forcing=Path(forcing_directory or root)/f'ERA5_1deg_{year}_Flux_processed.nc'
        if not forcing.is_file():raise FileNotFoundError(forcing)
        for month in range(1,13):
            pairs[len(pairs)+1]=(path(year,month),path(year+(month==12),month%12+1))
    return pairs

def forcing_positions(start, steps, dt):
    """3-hour model samples bracketing a rollout, mapped to Gregorian hours."""
    shift=model_timestamp(start)-(start.year-1993)*MODEL_YEAR
    first=max(0, int(np.floor(shift/10800))-1)
    last=int(np.floor((shift+steps*dt)/10800))+1
    positions=[]
    for i in range(first,last+1):
        year_delta, within=divmod(i*10800, MODEL_YEAR)
        month0, seconds=divmod(within,30*86400)
        year=start.year+year_delta;month=int(month0)+1
        hour=(datetime(year,month,1)-datetime(year,1,1)).total_seconds()/3600
        hour+=seconds/3600*calendar.monthrange(year,month)[1]/30
        positions.append((year,hour))
    return first%2880,positions

def load_window(directory,start,steps,dt):
    """Read only the needed hourly records, including real cross-year endpoints."""
    first,positions=forcing_positions(start,steps,dt)
    # Brackets are tuples (calendar year, hourly record); Jan 1 belongs to next file.
    brackets=[];needed={}
    for year,hour in positions:
        pair=[]
        for index in (int(np.floor(hour)),int(np.ceil(hour))):
            n=(366 if calendar.isleap(year) else 365)*24
            yy,ii=(year+1,index-n) if index>=n else (year,index)
            needed.setdefault(yy,set()).add(ii);pair.append((yy,ii))
        brackets.append((pair,hour-np.floor(hour)))
    tables={}
    for year,indices in needed.items():
        ids=sorted(indices)
        with h5netcdf.File(Path(directory)/f'ERA5_1deg_{year}_Flux_processed.nc') as f:
            for name in VARIABLES:
                data=np.asarray(f[name][ids,:,:],dtype=np.float64)
                tables[year,name]={i:data[j] for j,i in enumerate(ids)}
    result=[]
    for name in VARIABLES:
        array=np.stack([(1-w)*tables[a[0],name][a[1]]+w*tables[b[0],name][b[1]] for (a,b),w in brackets])
        if not np.isfinite(array).all():raise ValueError(f'Nonfinite forcing {name} at {start}')
        result.append(array.transpose(2,1,0).copy())
    return tuple(result),first


def validate_date_options(config):
    date = getattr(config, 'initial_date', None)
    if date:
        datetime.fromisoformat(date)
        if not config.forcing_directory:
            raise ValueError('initial_date requires forcing_directory')
        if getattr(config, 'forcing_time_shift', None) is not None:
            raise ValueError('Dated forcing computes its own clock offset')
    if getattr(config.model, 'use_date_embedding', False) and not (date or any(c.initial_date for c in getattr(config,'initial_states',[]))):
        raise ValueError('Date embedding requires dated initial conditions')
