"""Fixed-support multiyear ocean metrics from completed monthly archives."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from veros_control.diagnostics.annual import open_reference, model_record, year_indices

METRICS=['ke_mean_flow','volume_temperature','sst','sss','heat_content','ke_hourly_mean']

def common_support(run,assets,years,path):
    with xr.open_dataset(run/'baseline.averages.nc',decode_times=False) as b,xr.open_dataset(run/'trained.averages.nc',decode_times=False) as n:
        for key in ('xt','yt','xu','yu','zt','maskT','maskU','maskV'):
            if not np.array_equal(b[key],n[key]):raise ValueError('Paired grids differ: '+key)
        common={key:b[mask].values.astype(bool) for key,mask in [('temp','maskT'),('u','maskU'),('v','maskV')]}
        common['ssh']=common['temp'][-1].copy()
        for year in years:
            print(f'COMMON SUPPORT: {year}',flush=True)
            ib,in_=year_indices(b,year),year_indices(n,year)
            for month in range(1,13):
                records=[model_record(b,ib[month-1]),model_record(n,in_[month-1]),open_reference(assets/f'GLORYS12_1deg_{year}{month:02d}_processed.nc',b)]
                for data in records:
                    common['temp'] &= np.isfinite(data['temp']) & np.isfinite(data['salt'])
                    for key in ('u','v','ssh'):common[key] &= np.isfinite(data[key])
        if not common['temp'].any():raise ValueError('No shared wet volume')
        np.savez_compressed(path,**common,years=np.asarray(years))

def long_series(paths,folder):
    pieces=[]
    for path in paths:
        with xr.open_dataset(path) as ds:
            year=int(ds.attrs['year'])
            part=ds[METRICS].load().rename(month='sample')
            months=part.sample.values.copy()
            part=part.assign_coords(sample=year+(months-.5)/12,
                                    year=('sample',np.full(12,year)),month=('sample',months))
            pieces.append(part)
    joined=xr.concat(pieces,dim='sample',combine_attrs='drop_conflicts')
    joined.attrs.update(note='Monthly means matched by year/month; fixed common ocean support for every evaluated year. Reference is the actual GLORYS year; see run metadata for forcing provenance.',calendar='360_day model; Gregorian reference months')
    joined.to_netcdf(folder/'monthly_timeseries.nc',engine='h5netcdf')
    joined[METRICS[:-1]].to_dataframe().to_csv(folder/'monthly_timeseries.csv')
    joined[['ke_hourly_mean']].to_dataframe().to_csv(folder/'monthly_hourly_ke.csv')
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    specs=[('ke_mean_flow','Horizontal KE of monthly-mean flow','10¹⁸ J',1e18),
           ('volume_temperature','Volume-mean Conservative Temperature','°C',1),
           ('sst','Global mean SST (CT)','°C',1),('sss','Global mean SSS (SA)','g kg⁻¹',1),
           ('heat_content','Heat content relative to 0°C','10²⁴ J',1e24),
           ('ke_hourly_mean','Hourly-averaged horizontal KE (models only)','10¹⁸ J',1e18)]
    for ax,(key,title,unit,scale) in zip(axes.flat,specs):
        dim='model_trajectory' if key=='ke_hourly_mean' else 'trajectory'
        for label,color in zip(joined[dim].values,['#2166ac','#b2182b','black']):
            ax.plot(joined.sample,joined[key].sel({dim:label})/scale,color=color,label=str(label),lw=1.2)
        ax.set(title=title,ylabel=unit,xlabel='Year',xlim=(int(joined.year.min()),int(joined.year.max())+1));ax.grid(alpha=.2)
    axes[0,0].legend()
    fig.suptitle(f'Monthly paired evaluation — {int(joined.year.min())}–{int(joined.year.max())}\nActual GLORYS years; forcing provenance recorded with the rollout',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(folder/f'time_series_early_years.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
