"""Correction profiles, hourly RMS and equivalent column heat/salt fluxes."""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_corrections(directory):
    root=Path(directory)
    path=root/'trained.averages.nc'
    with xr.open_dataset(path,decode_times=False) as ds, PdfPages(root/'corrections.pdf') as pdf:
        rho=float(ds.attrs['rho0']);cp=float(ds.attrs['heat_capacity_reference'])
        for name,mask,area in [('dtemp_nn','maskT','area_t'),('dsalt_nn','maskT','area_t'),
                               ('du_nn','maskU','area_u'),('dv_nn','maskV','area_v')]:
            if name not in ds:continue
            wet=ds[mask].values.astype(bool);weight=ds[area].values[None]*wet
            mean=ds[name].mean('Time').values
            square=ds[name+'2'].mean('Time').values
            denominator=weight.sum(axis=(1,2))
            signed=np.sum(np.where(wet,mean,0)*weight,axis=(1,2))/np.maximum(denominator,1.)
            rms=np.sqrt(np.sum(np.where(wet,square,0)*weight,axis=(1,2))/np.maximum(denominator,1.))
            fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
            depth=np.abs(ds.zt.values)
            axes[0].plot(signed*3600,depth,label='signed mean');axes[0].plot(rms*3600,depth,label='hourly RMS')
            axes[0].invert_yaxis();axes[0].legend();axes[0].set(xlabel='Field units / hour',ylabel='Depth (m)')
            lon=ds.xu.values if mask=='maskU' else ds.xt.values
            lat=ds.yu.values if mask=='maskV' else ds.yt.values
            for ax,field,title in [(axes[1],mean[-1]*3600,'Surface mean / hour'),
                (axes[2],np.sqrt(square[-1])*3600,'Surface hourly RMS / hour')]:
                image=ax.pcolormesh(lon,lat,np.where(wet[-1],field,np.nan),shading='auto',cmap='RdBu_r')
                fig.colorbar(image,ax=ax);ax.set_title(title)
            fig.suptitle(name);pdf.savefig(fig);fig.savefig(root/f'{name}.png',dpi=150);plt.close(fig)
            if name in ('dtemp_nn','dsalt_nn'):
                # Salinity uses g/kg: divide by 1000 to report kg salt / m2 / s.
                factor=rho*cp if name=='dtemp_nn' else rho/1000.
                flux=np.sum(np.where(wet,mean,0)*ds.dzt.values[:,None,None],axis=0)*factor
                fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
                image=ax.pcolormesh(lon,lat,np.where(wet[-1],flux,np.nan),shading='auto',cmap='RdBu_r')
                unit='W m⁻²' if name=='dtemp_nn' else 'kg salt m⁻² s⁻¹'
                fig.colorbar(image,ax=ax,label=unit);ax.set_title('Equivalent column-integrated '+name+' flux')
                pdf.savefig(fig);fig.savefig(root/f'{name}_flux.png',dpi=150);plt.close(fig)
