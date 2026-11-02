"""End-to-end synthetic data check; figures are test fixtures, not experiment results."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
import xarray as xr
from veros_control.diagnostics.annual import run
from veros_control.diagnostics.plots import run as plot_run


def check(folder):
    folder=Path(folder); refs=folder/'refs';refs.mkdir()
    lon=np.arange(15.,360.,30.);lat=np.arange(-75.,76.,15.);z=np.array([-2000.,-500.,-50.,-5.])
    nz,ny,nx=len(z),len(lat),len(lon)
    dims=('Time','zt','yt','xt')
    temp=np.broadcast_to((12+z[:,None,None]/200+np.cos(np.deg2rad(lat))[None,:,None]),(nz,ny,nx)).copy()
    salt=np.full_like(temp,35.)
    v=np.broadcast_to(np.array([-.01,-.01,.05,.05])[:,None,None],temp.shape).copy()
    u=np.full_like(v,.02)
    ds=xr.Dataset(coords={'Time':np.arange(12),'zt':z,'zw':[-1000.,-100.,-10.,0.],'yt':lat,'yu':lat+.5,'xt':lon,'xu':lon+.5})
    for name,data,spatial in [('temp',temp,('zt','yt','xt')),('salt',salt,('zt','yt','xt')),('u',u,('zt','yt','xu')),('v',v,('zt','yu','xt'))]:
        ds[name]=( ('Time',*spatial), np.broadcast_to(data,(12,*data.shape)))
    ds['ssh']=(('Time','yt','xt'),np.broadcast_to(np.sin(np.deg2rad(lon))[None,None,:],(12,ny,nx)))
    ds['u2']=ds.u**2*2;ds['v2']=ds.v**2*2
    for k,spatial in [('T',('zt','yt','xt')),('U',('zt','yt','xu')),('V',('zt','yu','xt'))]:
        ds['mask'+k]=(spatial,np.ones(temp.shape,dtype='i1'))
    for k,spatial in [('t',('yt','xt')),('u',('yt','xu')),('v',('yu','xt'))]:
        ds['area_'+k]=(spatial,np.full((ny,nx),1e10))
    ds['dzt']=(('zt',),[2000.,950.,90.,10.]);ds['ht']=(('yt','xt'),np.full((ny,nx),3050.))
    ds['dxt']=(('xt',),np.full(nx,1e5));ds['cosu']=(('yu',),np.cos(np.deg2rad(lat+.5)))
    ds['simulation_year']=(('Time',),np.full(12,1994));ds['simulation_month']=(('Time',),np.arange(1,13));ds['sample_count']=(('Time',),np.full(12,720))
    ds.attrs.update(rho0=1024.,heat_capacity_reference=3991.86795711963)
    ds.to_netcdf(folder/'baseline.averages.nc',engine='h5netcdf')
    nn=ds.copy(deep=True);nn['temp']=nn.temp+1
    nn.to_netcdf(folder/'trained.averages.nc',engine='h5netcdf')
    (folder/'summary.json').write_text(json.dumps({k:{'status':'complete'} for k in ('baseline','trained')}))
    for month in range(1,13):
        ref=ds[['temp','salt','u','v','ssh']].isel(Time=slice(month-1,month))
        ref=ref.rename({'temp':'thetao','salt':'so','u':'uo','v':'vo','ssh':'zos','Time':'time','yt':'lat','xt':'lon','xu':'lon_u','yu':'lat_u'})
        ref.to_netcdf(refs/f'GLORYS12_1deg_1994{month:02d}_processed.nc',engine='h5netcdf')
    run(argparse.Namespace(run_dir=str(folder),glorys_dir=str(refs),year=1994,model_thermodynamics='pt-sp',output=str(folder/'evaluation.nc')))
    with xr.open_dataset(folder/'evaluation.nc') as out:
        np.testing.assert_allclose(out.annual_temp[0],out.annual_temp[2],atol=1e-6)
        np.testing.assert_allclose(out.ke_hourly_mean[0],2*out.ke_mean_flow[0])
        assert np.all(out.volume_temperature[1]>out.volume_temperature[0])
        np.testing.assert_allclose(out.annual_ssh.mean(('yt','xt')),0,atol=1e-7)
    plot_run(argparse.Namespace(input=str(folder/'evaluation.nc'),output_dir=str(folder/'figures')))
    assert len(list((folder/'figures').glob('*.png')))==11
    print('PASS: synthetic input → calculations → eleven PNG/PDF figures and CSV outputs',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--keep-dir');args=p.parse_args()
    if args.keep_dir:
        Path(args.keep_dir).mkdir(parents=True,exist_ok=False);check(args.keep_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='evaluation_test_') as folder:check(folder)
