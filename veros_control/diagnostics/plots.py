#!/usr/bin/env python
"""Plot the applicable PDF panels: biases, NN fields, sections and time series."""
import argparse
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from veros_control.diagnostics.ocean import BASINS, LAT_LIMITS, weighted_mean

# Discrete levels transcribed from the supplied PDF; absent SSS absolute panel
# uses the salinity palette and range from its zonal-section example.
TEMP_BIAS = [-2,-1.5,-1,-.7,-.4,-.1,0,.1,.4,.7,1,1.5,2]
SALT_BIAS = [-1,-.7,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.7,1]
SALT_SECTION = [-.4,-.3,-.2,-.1,-.06,-.02,0,.02,.06,.1,.2,.3,.4]
SALT_FIELD = [33,34,34.2,34.4,34.6,34.7,34.8,34.9,35.1,35.3,35.5,35.8,36.2,36.5,36.9]
SSH_BIAS = [-1,-.7,-.5,-.3,-.2,-.1,-.05,0,.05,.1,.2,.3,.5,.7,1]
WINTER_BIAS = [-1000,-500,-300,-200,-100,-50,-20,0,20,50,100,200,300,500,1000]
WINTER_FIELD = [0,20,50,100,150,200,300,400,500,600,1000,1500,2000,4000]
MAPS = {
 'sst':('SST (Conservative Temperature)','°C',TEMP_BIAS,np.arange(0,32,2)),
 'sss':('SSS (Absolute Salinity)','g kg⁻¹',SALT_BIAS,SALT_FIELD),
 'ssh':('SSH (global mean removed)','m',SSH_BIAS,np.arange(-1.8,1.21,.3)),
 'mld_winter':('Winter MLD','m',WINTER_BIAS,WINTER_FIELD),
 'mld_summer':('Summer MLD','m',[-30,-20,-10,-5,0,5,10,20,30],np.arange(0,151,10))}

def discrete(levels, cmap):
    palette = plt.get_cmap(cmap, len(levels)-1)
    return palette, BoundaryNorm(levels,palette.N)

def save(fig, folder, name):
    fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight')
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)
    print('Plotted '+name,flush=True)

def map_panel(ax,field,lon,lat,levels,cmap,title,unit):
    order=np.argsort(lon%360)
    cyclic,x=add_cyclic_point(field[:,order],coord=(lon%360)[order])
    palette,norm=discrete(levels,cmap)
    img=ax.contourf(x,lat,cyclic,levels=levels,cmap=palette,norm=norm,extend='both',transform=ccrs.PlateCarree())
    ax.set_global();ax.set_extent([-180,180,-80,80],crs=ccrs.PlateCarree())
    ax.set_facecolor('#d3d3d3')
    coastline=Path(cartopy.config['data_dir'])/'shapefiles/natural_earth/physical/ne_110m_coastline.shp'
    if coastline.exists():
        ax.coastlines(resolution='110m',linewidth=.45)
    grid=ax.gridlines(draw_labels=True,linewidth=.2,color='gray',alpha=.4)
    grid.top_labels=False;grid.right_labels=False
    grid.xlabel_style={'size':7};grid.ylabel_style={'size':7}
    ax.set_title(title,fontsize=10)
    cb=plt.colorbar(img,ax=ax,orientation='horizontal',pad=.11,fraction=.06,shrink=.95)
    cb.set_label(unit,fontsize=8);cb.ax.tick_params(labelsize=7)

def bias_title(label,error,area):
    mean=weighted_mean(error,area);rmse=np.sqrt(weighted_mean(error**2,area))
    return f'{label} − GLORYS\nMean bias {mean:.3g}; RMSE {rmse:.3g}'

def maps(ds,folder,only=None,mld_trajectory="NN"):
    fields={'sst':ds.annual_temp.isel(zt=-1).values,'sss':ds.annual_salt.isel(zt=-1).values,
            'ssh':ds.annual_ssh.values,'mld_winter':ds.mld_winter.values,'mld_summer':ds.mld_summer.values}
    for key,data in fields.items():
        if only is not None and key not in only:
            continue
        name,unit,bias_levels,field_levels=MAPS[key]
        is_mld=key.startswith('mld')
        fig,axs=plt.subplots(1,3,figsize=(17,4.3),subplot_kw={'projection':ccrs.PlateCarree(central_longitude=210)})
        panels=[(data[1]-data[2],bias_levels,'RdBu_r',bias_title('NN',data[1]-data[2],ds.area.values)),(data[1],field_levels,'RdYlBu_r','NN yearly field')]
        if not is_mld:
            panels.insert(0,(data[0]-data[2],bias_levels,'RdBu_r',bias_title('no-NN',data[0]-data[2],ds.area.values)))
        else:
            index=list(ds.trajectory.values).index(mld_trajectory)
            error=data[index]-data[2]
            panels=[(error,bias_levels,'RdBu_r',bias_title(mld_trajectory,error,ds.area.values)),
                    (data[index],field_levels,'RdYlBu_r',f'{mld_trajectory} seasonal MLD'),
                    (data[2],field_levels,'RdYlBu_r','GLORYS seasonal MLD')]
        for ax,(field,levels,cmap,title) in zip(axs,panels):
            map_panel(ax,field,ds.xt.values,ds.yt.values,levels,cmap,title,unit)
        fig.suptitle(f'{name} — {ds.attrs["year"]}',fontsize=14,y=1.02)
        if is_mld:
            fig.text(.5,-.03,'Winter: JFM north / JAS south. Summer: JAS north / JFM south.\nDensity threshold 0.03 kg m⁻³ relative to 10 m, from monthly profiles.',ha='center',fontsize=9)
        suffix="_no_nn" if is_mld and mld_trajectory=="no-NN" else ""
        save(fig,folder,key+suffix)

def sections(ds,folder):
    z=np.abs(ds.zt.values);order=np.argsort(z);z=z[order]
    for key,unit,bias_levels,field_levels in [('temp','°C',TEMP_BIAS,np.arange(0,32,2)),('salt','g kg⁻¹',SALT_SECTION,SALT_FIELD)]:
        data=ds['zonal_'+key].values
        fig,axs=plt.subplots(3,4,figsize=(15,11),layout='constrained',sharey=True)
        for row in range(3):
            levels=bias_levels if row<2 else field_levels
            cmap,norm=discrete(levels,'RdBu_r' if row<2 else 'RdYlBu_r')
            for basin,ax in enumerate(axs[row]):
                field=(data[row,basin]-data[2,basin]) if row<2 else data[1,basin]
                img=ax.pcolormesh(ds.yt,z,field[order],cmap=cmap,norm=norm,shading='auto')
                ax.set_xlim(*LAT_LIMITS[basin]);ax.set_ylim(z.max(),0);ax.set_facecolor('#d3d3d3')
                ax.set_title(BASINS[basin]);ax.set_xlabel('Latitude (°N)')
                if basin==0:
                    label=['no-NN − GLORYS','NN − GLORYS','NN field'][row]
                    ax.set_ylabel(label+'\nDepth (m)')
            fig.colorbar(img,ax=list(axs[row]),label=unit,extend='both',shrink=.9)
        fig.suptitle(f'Annual zonal-mean {"Conservative Temperature" if key=="temp" else "Absolute Salinity"} — {ds.attrs["year"]}',fontsize=15)
        save(fig,folder,'zonal_'+key)

def moc_plot(ds,folder):
    levels=np.arange(-30,32,2);cmap,norm=discrete(levels,'RdBu_r')
    z=np.abs(ds.zw.values);order=np.argsort(z)
    fig,axs=plt.subplots(3,3,figsize=(14,11),layout='constrained',sharey=True)
    for row,label in enumerate(ds.trajectory.values):
        for basin,ax in enumerate(axs[row]):
            field=ds.moc.values[row,basin][order]
            img=ax.pcolormesh(ds.yu,z[order],field,cmap=cmap,norm=norm,shading='auto')
            if np.isfinite(field).any():
                ax.contour(ds.yu,z[order],field,levels=np.arange(-30,31,4),colors='black',linewidths=.4)
            ax.set_ylim(z.max(),0);ax.set_xlim(*[(-80,-30),(-30,80),(-30,60)][basin])
            ax.set_facecolor('#d3d3d3');ax.set_title(str(ds.moc_basin.values[basin]));ax.set_xlabel('Latitude (°N)')
            if basin==0:ax.set_ylabel(f'{label}\nDepth (m)')
        fig.colorbar(img,ax=list(axs[row]),label='Sv',extend='both',shrink=.9)
    fig.suptitle(f'Resolved meridional overturning — {ds.attrs["year"]}\nPositive clockwise; excludes parameterized eddy transport',fontsize=14)
    save(fig,folder,'overturning')

def time_series(ds,folder):
    specs=[('ke_mean_flow','Horizontal KE of monthly-mean flow','10¹⁸ J',1e18),
           ('volume_temperature','Global volume-mean Conservative Temperature','°C',1),
           ('sst','Global mean SST (CT)','°C',1),('sss','Global mean SSS (SA)','g kg⁻¹',1),
           ('heat_content','Ocean heat content relative to 0°C','10²⁴ J',1e24)]
    fig,axs=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for ax,(key,title,unit,scale) in zip(axs.flat,specs):
        for j,(label,color) in enumerate(zip(ds.trajectory.values,['#2166ac','#b2182b','black'])):
            ax.plot(ds.month,ds[key].values[j]/scale,'o-',ms=3,label=str(label),color=color)
        ax.set(title=title,ylabel=unit,xticks=np.arange(1,13),xlabel=f'Month ({ds.attrs["year"]})');ax.grid(alpha=.2)
    axs[0,0].legend()
    ax=axs.flat[-1]
    for j,(label,color) in enumerate(zip(ds.model_trajectory.values,['#2166ac','#b2182b'])):
        ax.plot(ds.month,ds.ke_hourly_mean.values[j]/1e18,'o-',ms=3,color=color,label=str(label))
    ax.set(title='Hourly-averaged horizontal KE (models only)',ylabel='10¹⁸ J',xlabel='Month',xticks=np.arange(1,13));ax.grid(alpha=.2)
    fig.suptitle('Monthly ocean diagnostics — fixed common ocean support',fontsize=15)
    save(fig,folder,'time_series')
    # CSVs permit numerical review independently of figures.
    ds[[s[0] for s in specs]].to_dataframe().to_csv(folder/'monthly_metrics.csv')
    ds[['ke_hourly_mean']].to_dataframe().to_csv(folder/'monthly_hourly_ke.csv')

def run(args):
    folder=Path(args.output_dir);folder.mkdir(parents=True,exist_ok=True)
    cached=Path(os.environ.get('VEROS_EVALUATION_COASTLINES',cartopy.config['data_dir']))
    cartopy.config['data_dir']=str(cached)
    coastline=cached/'shapefiles/natural_earth/physical/ne_110m_coastline.shp'
    with xr.open_dataset(args.input) as ds:
        time_series(ds,folder);maps(ds,folder)
        maps(ds,folder,only=["mld_winter","mld_summer"],mld_trajectory="no-NN")
        sections(ds,folder);moc_plot(ds,folder)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output-dir',required=True)
    run(p.parse_args())
