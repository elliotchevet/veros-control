#!/usr/bin/env python
"""Prepare compact evaluation data from a completed paired monthly archive."""
import argparse
import calendar
import json
from pathlib import Path
import numpy as np
import xarray as xr
from veros_control.diagnostics.ocean import (BASINS, basin_masks, weighted_mean, thermodynamics,
                         mld_density, seasonal, overturning)

LABELS = ['no-NN', 'NN', 'GLORYS']
NAMES = {'temp':'thetao', 'salt':'so', 'u':'uo', 'v':'vo', 'ssh':'zos'}
DIMENSIONS = {'lat':'yt', 'lon':'xt', 'lon_u':'xu', 'lat_u':'yu'}

def open_reference(path, grid):
    ds = xr.open_dataset(path, decode_times=False).rename(DIMENSIONS)
    try:
        for name in ('xt','yt','xu','yu','zt'):
            delta = ds[name].values - grid[name].values
            if name in ('xt','xu'):
                delta = (delta + 180) % 360 - 180
            if ds[name].shape != grid[name].shape or not np.allclose(delta, 0, atol=1e-5):
                raise ValueError(f'{path}: {name} does not match the model grid; explicit remapping required')
        data = {name: np.asarray(ds[key].isel(time=0).values, float) for name,key in NAMES.items()}
        return data
    finally:
        ds.close()

def model_record(ds, index):
    return {name: np.asarray(ds[name].isel(Time=index).values, float) for name in NAMES}

def year_indices(ds, year):
    indices = np.flatnonzero(ds.simulation_year.values == year)
    if len(indices) != 12 or not np.array_equal(ds.simulation_month.values[indices], np.arange(1,13)):
        raise ValueError(f'Require all 12 completed months for {year}')
    if not np.all(ds.sample_count.values[indices] == 720):
        raise ValueError('Incomplete monthly averages')
    return indices

def run(args):
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir/'summary.json').read_text())
    if any(summary.get(k,{}).get('status') != 'complete' for k in ('baseline','trained')):
        raise ValueError('Paired rollout is incomplete; do not evaluate files still being written')
    refs = [Path(args.glorys_dir)/f'GLORYS12_1deg_{args.year}{m:02d}_processed.nc' for m in range(1,13)]
    missing = [str(p) for p in refs if not p.is_file()]
    if missing:
        raise FileNotFoundError('\n'.join(missing))
    datasets = [xr.open_dataset(run_dir/f'{label}.averages.nc', decode_times=False) for label in ('baseline','trained')]
    try:
        b,n = datasets
        indices = [year_indices(ds,args.year) for ds in datasets]
        for name in ('xt','yt','xu','yu','zt','dzt','maskT','maskU','maskV','area_t','area_u','area_v'):
            if not np.array_equal(b[name],n[name]):
                raise ValueError(f'Paired grids differ: {name}')
        lon,lat,depth = b.xt.values,b.yt.values,b.zt.values
        area = b.area_t.values
        thickness = b.dzt.values
        rho = float(b.attrs['rho0']); cp = float(b.attrs['heat_capacity_reference'])
        common = {name: b[mask].values.astype(bool) for name,mask in [('temp','maskT'),('u','maskU'),('v','maskV')]}
        common['ssh'] = common['temp'][-1].copy()
        # Fixed shared support across all months and all three trajectories.
        print('Checking common wet support across the complete evaluation year', flush=True)
        for month,path in enumerate(refs):
            records = [model_record(ds,ix[month]) for ds,ix in zip(datasets,indices)] + [open_reference(path,b)]
            for data in records:
                common['temp'] &= np.isfinite(data['temp']) & np.isfinite(data['salt'])
                for name in ('u','v','ssh'):
                    common[name] &= np.isfinite(data[name])
        if getattr(args,'common_mask_file',None):
            with np.load(args.common_mask_file) as saved:
                for name in common:
                    if saved[name].shape!=common[name].shape:raise ValueError('Common mask shape mismatch')
                    common[name] &= saved[name].astype(bool)
        volume = area[None]*thickness[:,None,None]*common['temp']
        uv_volumes = [b[f'area_{name}'].values[None]*thickness[:,None,None]*common[name] for name in ('u','v')]
        surface_weights = area*common['temp'][-1]
        ssh_weights = area*common['ssh']
        if volume.sum() <= 0 or surface_weights.sum() <= 0:
            raise ValueError('No common valid ocean support')
        basins = basin_masks(lon,lat)
        vbasins = basin_masks(lon,b.yu.values)
        # PDF overturning uses Southern, Atlantic and combined Indo-Pacific.
        moc_masks = np.stack([vbasins[0],vbasins[1],vbasins[2]|vbasins[3]])
        width = b.cosu.values[:,None]*b.dxt.values[None,:]
        annual = {key: np.zeros((3,*common['temp'].shape)) for key in ('temp','salt')}
        ssh = np.zeros((3,*area.shape))
        mld = np.empty((3,12,*area.shape))
        moc = np.zeros((3,3,len(depth),len(lat)))
        metrics = {key:np.zeros((3,12)) for key in ('ke_mean_flow','volume_temperature','sst','sss','heat_content')}
        hourly_ke = np.empty((2,12))
        weights = np.ones((3,12))/12
        days = np.array([calendar.monthrange(args.year,m)[1] for m in range(1,13)])
        weights[2] = days/days.sum()
        bottom = b.ht.values
        for month,path in enumerate(refs):
            print(f'Computing {args.year}-{month+1:02d} (three trajectories)', flush=True)
            records = [model_record(ds,ix[month]) for ds,ix in zip(datasets,indices)] + [open_reference(path,b)]
            for j,data in enumerate(records):
                convention = args.model_thermodynamics if j < 2 else 'pt-sp'
                ct,sa = thermodynamics(data['temp'],data['salt'],depth,lon,lat,convention)
                ct = np.where(common['temp'],ct,np.nan)
                sa = np.where(common['temp'],sa,np.nan)
                for key,field in [('temp',ct),('salt',sa)]:
                    annual[key][j] += np.nan_to_num(field)*weights[j,month]
                metrics['volume_temperature'][j,month] = weighted_mean(ct,volume)
                metrics['sst'][j,month] = weighted_mean(ct[-1],surface_weights)
                metrics['sss'][j,month] = weighted_mean(sa[-1],surface_weights)
                metrics['heat_content'][j,month] = rho*cp*np.nansum(ct*volume)
                metrics['ke_mean_flow'][j,month] = .5*rho*sum(np.sum(np.where(common[k],data[k]**2,0)*vol) for k,vol in zip(('u','v'),uv_volumes))
                ssh_field = np.where(common['ssh'],data['ssh'],np.nan)
                # Remove each monthly area-weighted datum before comparing SSH.
                ssh[j] += np.nan_to_num(ssh_field-weighted_mean(ssh_field,ssh_weights))*weights[j,month]
                mld[j,month] = mld_density(ct,sa,depth,bottom,lon,lat,'ct-sa')
                transport = overturning(data['v'],width,thickness,common['v'],moc_masks,depth)
                moc[j] += np.nan_to_num(transport)*weights[j,month]
            for j,(ds,ix) in enumerate(zip(datasets,indices)):
                hourly_ke[j,month] = .5*rho*sum(np.sum(np.where(common[k],ds[k+'2'].isel(Time=ix[month]).values,0)*vol) for k,vol in zip(('u','v'),uv_volumes))
        coords = dict(trajectory=LABELS,month=np.arange(1,13),zt=depth,yt=lat,xt=lon,
                      basin=list(BASINS),moc_basin=['Southern','Atlantic','Indo-Pacific'],yu=b.yu.values,zw=b.zw.values,
                      model_trajectory=LABELS[:2])
        output = xr.Dataset(coords=coords)
        for key,value in annual.items():
            value = np.where(common['temp'][None],value,np.nan)
            output['annual_'+key] = (('trajectory','zt','yt','xt'),value.astype('f4'))
            section = np.stack([weighted_mean(value,area[None,None]*common['temp'][None]*mask[None,None],axis=-1) for mask in basins],axis=1)
            output['zonal_'+key] = (('trajectory','basin','zt','yt'),section.astype('f4'))
        output['annual_ssh'] = (('trajectory','yt','xt'),np.where(common['ssh'][None],ssh,np.nan).astype('f4'))
        for winter in (True,False):
            values = []
            for j in range(3):
                # Gregorian day weighting for GLORYS seasonal means.
                def season(indices):
                    w = weights[j,indices]; w = w/w.sum()
                    return np.sum(mld[j,indices]*w[:,None,None],axis=0)
                north=season([0,1,2] if winter else [6,7,8]); south=season([6,7,8] if winter else [0,1,2])
                values.append(np.where(lat[:,None]>=0,north,south))
            output['mld_'+('winter' if winter else 'summer')] = (('trajectory','yt','xt'),np.asarray(values).astype('f4'))
        for k,values in metrics.items():
            output[k] = (('trajectory','month'),values)
        output['ke_hourly_mean'] = (('model_trajectory','month'),hourly_ke)
        active = np.stack([np.any(common['v'] & mask[None],axis=-1) for mask in moc_masks])
        output['moc'] = (('trajectory','moc_basin','zw','yu'),np.where(active[None],moc,np.nan).astype('f4'))
        output['area'] = (('yt','xt'),area)
        output['common_mask'] = (('zt','yt','xt'),common['temp'].astype('i1'))
        output.attrs.update(year=args.year,run_directory=str(run_dir.resolve()),model_thermodynamics=args.model_thermodynamics,
            reference_files=json.dumps([str(p) for p in refs]),
            averaging='Model: equal 30-day months; GLORYS: Gregorian day-weighted annual and seasonal means',
            basin_definition='Rectangular sectors: Atlantic [-70,20), Indian [20,120), Pacific complement; Southern south of -30. Latitude limits in README.',
            kinetic_energy='Horizontal resolved KE only. Three-curve metric uses monthly-mean velocities for all products; hourly second-moment model KE saved separately. Common fixed U/V support.',
            heat_content='rho0 * cp0 * integral(Conservative Temperature dV), relative to 0 degC; fixed common volume',
            rho0=rho,cp0=cp,mld_definition='First sigma0 increase of 0.03 kg/m3 relative to interpolated 10m value; monthly mean profiles; uncrossed profiles bottom-capped',
            ssh_definition='Each monthly SSH field has its own shared-area global mean removed',
            moc_definition='Resolved Eulerian volume overturning only; negative bottom-up integral of northward transport, Sv; no parameterized eddy transport',
            thermodynamics_note='Default ct-sa follows model EOS5 conventions; raw GLORYS thetao/so are converted from pt/SP to CT/SA. Historical initialization inserted raw GLORYS values without this conversion.')
        units = dict(ke_mean_flow='J',ke_hourly_mean='J',volume_temperature='degC',sst='degC',sss='g kg-1',heat_content='J',annual_temp='degC',annual_salt='g kg-1',annual_ssh='m',mld_winter='m',mld_summer='m',moc='Sv')
        for key,unit in units.items(): output[key].attrs['units']=unit
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix('.pending.nc')
        output.to_netcdf(temporary,engine='h5netcdf',encoding={k:{'zlib':True,'complevel':1} for k in output.data_vars})
        temporary.replace(dest)
        print(f'Saved {dest}',flush=True)
    finally:
        for ds in datasets: ds.close()

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--glorys-dir',required=True)
    p.add_argument('--year',type=int,default=1994)
    p.add_argument('--model-thermodynamics',choices=['ct-sa','pt-sp'],default='ct-sa')
    p.add_argument('--output',required=True)
    p.add_argument('--common-mask-file')
    run(p.parse_args())
