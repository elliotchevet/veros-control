"""Heat-budget equivalents of direct NN temperature sources, on Storto Fig. 5's scale."""
import argparse
import json
import os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('diagnostics',type=Path)
p.add_argument('--color-limit', type=float, default=200., help='Symmetric color limit in W/m²')
p.add_argument('--max-scale', action='store_true', help='Use separate symmetric maximum-absolute color limits, without clipping')
a=p.parse_args();out=a.diagnostics.resolve()
import math
if not math.isfinite(a.color_limit) or a.color_limit <= 0:
    p.error('--color-limit must be finite and positive')
os.environ['MPLCONFIGDIR']=str(out/'matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
cache=Path(__file__).resolve().parents[1]/'basin_maps/cartopy_data'
cartopy.config['data_dir']=cache;cartopy.config['pre_existing_data_dir']=cache
meta=json.loads((out/'metadata.json').read_text())
g=np.load(out/'grid_target.npz');corr=np.load(out/'trained_integrated.npz')
rho=1024.;cp=3991.86795711963
wet=g['wet_temp'].astype(bool);area=g['area_temp'];dz=g['dzt'];depth=g['depth'];top=int(np.argmin(depth))
tendency=np.where(wet,corr['mean_temp'],0.) # K/s; already averaged over the saved rollout
assert np.isfinite(tendency).all()
q_surface=rho*cp*dz[top]*tendency[...,top]
q_column=rho*cp*np.sum(tendency*dz[None,None,:],axis=2)
# Independently verify the conversion using accumulated direct source increments.
np.testing.assert_allclose(q_column,rho*cp*np.sum(np.where(wet,corr['integral_temp'],0.)*dz,axis=2)/(meta['n_steps']*meta['dt_seconds']),rtol=1e-10,atol=1e-8)
mask=wet[...,top].copy();mask[:2]=False;mask[-2:]=False;mask[:,:2]=False;mask[:,-2:]=False
lon=(g['lon']+180)%360-180;ix=np.argsort(lon[2:-2])+2
levels=np.linspace(-a.color_limit,a.color_limit,21);cmap=plt.get_cmap('bwr',len(levels)-1);norm=BoundaryNorm(levels,cmap.N)
fig,axes=plt.subplots(2,1,figsize=(13,10),subplot_kw={'projection':ccrs.PlateCarree()})
stats={}
images=[]
limits=[]
for ax,q,key,title in zip(axes,[q_surface,q_column],['surface_layer','full_column'],[
    f'(a) Surface-layer contribution (thickness {dz[top]:.1f} m)',
    '(b) Full-column net contribution (includes the depth taper)']):
    values=np.where(mask,q,np.nan)
    limit=float(np.max(np.abs(q[mask]))) if a.max_scale else a.color_limit
    if a.max_scale:
        norm=Normalize(vmin=-limit,vmax=limit)
        cmap=plt.get_cmap("bwr")
    im=ax.pcolormesh(lon[ix],g['lat'][2:-2],values[ix,2:-2].T,transform=ccrs.PlateCarree(),shading='auto',cmap=cmap,norm=norm,rasterized=True)
    images.append(im);limits.append(limit)
    ax.set_global();ax.set_facecolor('white')
    ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#dddddd',edgecolor='none',zorder=2)
    ax.coastlines(resolution='110m',linewidth=.4,color='#333333',zorder=3)
    ax.set_xticks(np.arange(-180,181,60),crs=ccrs.PlateCarree());ax.set_yticks([-60,-30,0,30,60],crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter());ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_title(title,loc='left',fontsize=13)
    weights=area[mask];v=q[mask]
    stats[key]=dict(mean_W_m2=float(np.average(v,weights=weights)),
        mean_absolute_W_m2=float(np.average(abs(v),weights=weights)),
        spatial_rms_of_time_mean_W_m2=float(np.sqrt(np.average(v*v,weights=weights))),
        min_W_m2=float(v.min()),max_W_m2=float(v.max()),
        area_percent_outside_color_scale=float(100*np.average(abs(v)>limit,weights=weights)))
fig.subplots_adjust(top=.87,bottom=.19,left=.045,right=.89,hspace=.25)
fig.suptitle('NN temperature correction expressed as an equivalent heat flux',fontsize=17,y=.975)
fig.text(.5,.925,f'Job {Path(meta["checkpoint"]).parent.name.removeprefix("job_")}, update {meta["update"]} | lambda = {meta.get("basin_pattern_lambda",0):g} | {meta["n_steps"]*meta["dt_seconds"]/3600:g}-hour mean',ha='center',fontsize=12)
if a.max_scale:
    for ax,im,limit in zip(axes,images,limits):
        pos=ax.get_position()
        cax=fig.add_axes([.91,pos.y0,.018,pos.height])
        cb=fig.colorbar(im,cax=cax,ticks=np.linspace(-limit,limit,5),format='%.1f')
        cb.set_label('Equivalent heat flux (W m$^{-2}$)')
else:
    cax=fig.add_axes([.91,.23,.018,.56]);cb=fig.colorbar(im,cax=cax,ticks=np.linspace(-a.color_limit,a.color_limit,11),extend='both');cb.set_label('Equivalent heat flux (W m$^{-2}$)')
caption=f'Positive (red): direct heating; negative (blue): direct cooling. Shared color scale for both panels.\nValues outside +/-{a.color_limit:g} W/m² are color-saturated; full values are retained in the data.\nThese are 27-hour means of volumetric NN sources, not seasonal means or actual surface-flux corrections.'
if a.max_scale:
    caption='Positive (red): direct heating; negative (blue): direct cooling. Separate symmetric scales for each panel.\nLimits equal the maximum absolute wet-cell value; no values are clipped.\nThese are 27-hour means of volumetric NN sources, not seasonal means or actual surface-flux corrections.'
fig.text(.055,.045,caption,fontsize=10,linespacing=1.6)
figure_name='heat_flux_equivalent_max_scale.png' if a.max_scale else ('heat_flux_equivalent_storto_scale.png' if a.color_limit == 200 else f'heat_flux_equivalent_clip{a.color_limit:g}.png')
fig.savefig(out/figure_name,dpi=200,facecolor='white');plt.close(fig)
np.savez_compressed(out/'heat_flux_equivalent.npz',lon=g['lon'],lat=g['lat'],wet=mask,surface_layer_W_m2=np.where(mask,q_surface,np.nan),full_column_W_m2=np.where(mask,q_column,np.nan))
report=dict(source_checkpoint=meta['checkpoint'],rho_kg_m3=rho,cp_J_kg_K=cp,averaging_hours=meta['n_steps']*meta['dt_seconds']/3600,statistics=stats)
(out/('heat_flux_equivalent_statistics.json' if not a.max_scale and a.color_limit == 200 else Path(figure_name).stem+'_statistics.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2));print(out/figure_name)
