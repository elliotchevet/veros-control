"""Render paired rollout artifacts without opening pickle checkpoints."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from metrics import basin_stats

p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output)
meta=json.loads((out/'metadata.json').read_text());g=np.load(out/'grid_target.npz')
h={k:np.load(out/f'{k}_history.npz') for k in ('baseline','trained')}
f={k:np.load(out/f'{k}_step27.npz') for k in h}
integ=np.load(out/'trained_integrated.npz')
names=[s.replace('_',' ').title() for s in meta['basins']]
z=g['depth'];order=np.argsort(z);top=np.asarray(meta['trained_level_indices']);keys=('temp','salt','u','v')
units={'temp':'°C','salt':'g/kg','u':'m/s','v':'m/s'}
figures=[]
def save(fig,name):
    fig.savefig(out/name,dpi=150,bbox_inches='tight');plt.close(fig);figures.append(name)

def heat(ax,data,title,signed=False):
    values=data[:,order].T
    lim=np.nanmax(np.abs(values)) or 1e-15
    im=ax.imshow(values,aspect='auto',origin='upper',cmap='RdBu_r' if signed else 'viridis',
        vmin=-lim if signed else 0,vmax=lim)
    ticks=np.arange(0,len(z),5)
    ax.set_yticks(ticks,[f'{z[order[t]]:.0f}' for t in ticks]);ax.set_ylabel('Depth (m; level spacing)')
    ax.set_xticks(range(6),names,rotation=35,ha='right');ax.set_title(title)
    plt.colorbar(im,ax=ax,shrink=.8)

for metric,title,signed in [('mean','Signed mean',True),('rms','RMS',False),('saturation','Fraction ≥95% bound',False)]:
    fig,axes=plt.subplots(2,2,figsize=(15,12),constrained_layout=True)
    for ax,k in zip(axes.flat,keys):
        if metric=='mean': data=np.mean(h['trained']['corr_mean_'+k],axis=0)
        elif metric=='rms': data=np.sqrt(np.mean(h['trained']['corr_rms_'+k]**2,axis=0))
        else:data=np.mean(h['trained']['saturation_'+k],axis=0)
        heat(ax,data,f'{k}: {title}'+(' ('+units[k]+'/hour)' if metric!='saturation' else ''),signed)
    save(fig,'correction_'+metric+'.png')
fig,axes=plt.subplots(2,2,figsize=(15,11),constrained_layout=True)
for ax,k in zip(axes.flat,keys):
    means=np.mean(h['trained']['corr_mean_'+k],axis=0)
    for b,name in enumerate(names):ax.plot(means[b,order],z[order],label=name)
    ax.invert_yaxis();ax.set_yscale('symlog',linthresh=100);ax.set_xlabel(units[k]+'/hour')
    ax.set_ylabel('Depth (m)');ax.set_title(k+' signed correction');ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);save(fig,'correction_profiles.png')
fig,axes=plt.subplots(2,2,figsize=(15,12),constrained_layout=True)
for row,k in enumerate(('temp','salt')):
    delta_rmse=h['trained']['rmse_'+k][-1]-h['baseline']['rmse_'+k][-1]
    delta_mse=h['trained']['bias_'+k][-1]**2-h['baseline']['bias_'+k][-1]**2
    heat(axes[row,0],delta_rmse,k+' Δ pointwise RMSE ('+units[k]+')',True)
    heat(axes[row,1],delta_mse,k+' Δ squared basin-mean error',True)
fig.suptitle('Trained − baseline at hour 27; blue means improvement')
save(fig,'error_improvement.png')
fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
for label in h:
    axes[0,0].plot(np.arange(28),h[label]['normalized_loss'],label=label)
    for j,k in enumerate(('temp','salt')):
        axes[0,1].plot(np.arange(28),h[label]['loss_terms'][:,j],label=label+' '+k)
    axes[1,0].plot(np.arange(1,28),np.max(h[label]['cfl'],axis=1),label=label)
    axes[1,1].plot(np.arange(1,28),h[label]['salt_min'],label=label)
for ax,title in zip(axes.flat,['Normalized basin objective','Raw basin losses (different units)','Maximum directional CFL','Minimum salinity (g/kg)']):
    ax.set_title(title);ax.set_xlabel('Hour');ax.legend();ax.grid(alpha=.2)
fig.suptitle('Error curves use the fixed hour-27 interpolated GLORYS target')
save(fig,'paired_curves.png')
# Geographic raster maps: avoid external map downloads; gray denotes masked land/halos.
def mapfield(ax,field,k,title,signed=True):
    lon=g['lon_u'] if k=='u' else g['lon'];lat=g['lat_v'] if k=='v' else g['lat']
    lon=(lon+180)%360-180;ix=np.argsort(lon[2:-2])+2
    value=field[ix,2:-2].T
    finite=np.abs(value[np.isfinite(value)])
    lim=float(np.quantile(finite,.99)) if finite.size else 1.
    lim=max(lim,1e-15)
    ax.set_facecolor('.8')
    im=ax.pcolormesh(lon[ix],lat[2:-2],value,shading='auto',cmap='RdBu_r' if signed else 'viridis',
        vmin=-lim if signed else 0,vmax=lim,rasterized=True)
    ax.set_title(title);ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    plt.colorbar(im,ax=ax,extend='both',shrink=.8)
for level in (order[0],order[4],order[min(10,len(order)-1)]):
    fig,axes=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
    for row,k in enumerate(('temp','salt')):
        mask=g['wet_'+k][...,level]
        values=[integ['mean_'+k][...,level]*3600,
            f['trained'][k][...,level]-f['baseline'][k][...,level],
            (f['trained'][k][...,level]-g['target_'+k][...,level])**2-(f['baseline'][k][...,level]-g['target_'+k][...,level])**2]
        for ax,val,title in zip(axes[row],values,['Mean correction/hour','Trained − baseline','Δ squared GLORYS error']):
            mapfield(ax,np.where(mask,val,np.nan),k,k+': '+title)
    fig.suptitle(f'Depth {z[level]:.1f} m; color limits clip at 99th percentile')
    save(fig,f'maps_depth_{z[level]:.0f}m.png')
fig,axes=plt.subplots(2,2,figsize=(15,9),constrained_layout=True)
for ax,k in zip(axes.flat,keys):
    level=order[0];mask=g['wet_'+k][...,level]
    mapfield(ax,np.where(mask,f['trained'][k][...,level]-f['baseline'][k][...,level],np.nan),k,k+' final change, surface')
save(fig,'surface_state_change.png')
fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
for ax,level in zip(axes,(order[0],order[min(10,len(order)-1)])):
    mask=g['wet_temp'][...,level]
    mapfield(ax,np.where(mask,f['trained']['w'][...,level]-f['baseline']['w'][...,level],np.nan),'temp',
        f'Vertical velocity change (m/s), level {z[level]:.1f} m')
fig.suptitle('w on native vertical interfaces; depth labels identify corresponding tracer levels')
save(fig,'vertical_velocity_change.png')
# Basin masks reconstructed from metadata's established rectangle definitions.
import importlib.util
spec=importlib.util.spec_from_file_location('basins',Path(__file__).resolve().parents[2]/'training/basin_loss.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
rows=[]
for ax,k in zip(axes.flat,keys):
    x=g['lon_u'] if k=='u' else g['lon'];y=g['lat_v'] if k=='v' else g['lat']
    masks=b.basin_masks(x,y);wet=g['wet_'+k];area=g['area_'+k]
    direct=integ['integral_'+k];change=f['trained'][k]-f['baseline'][k]
    dm,_,_=basin_stats(direct,area,wet,masks);cm,_,_=basin_stats(change,area,wet,masks)
    for bi,name in enumerate(names):
        line=ax.plot(cm[bi,order],z[order],label=name)[0]
        ax.plot(dm[bi,order],z[order],ls='--',color=line.get_color())
        for zi in order:
            rows.append(dict(channel=k,basin=name,depth_m=z[zi],trained_level=bool(zi in top),
                correction_mean_per_hour=np.mean(h['trained']['corr_mean_'+k][:,bi,zi]),
                correction_rms_per_hour=np.sqrt(np.mean(h['trained']['corr_rms_'+k][:,bi,zi]**2)),
                saturation_fraction=np.mean(h['trained']['saturation_'+k][:,bi,zi]),
                direct_integral=dm[bi,zi],paired_change=cm[bi,zi],coupled_residual=cm[bi,zi]-dm[bi,zi],
                baseline_bias=h['baseline']['bias_'+k][-1,bi,zi] if k in ('temp','salt') else '',
                trained_bias=h['trained']['bias_'+k][-1,bi,zi] if k in ('temp','salt') else '',
                baseline_rmse=h['baseline']['rmse_'+k][-1,bi,zi] if k in ('temp','salt') else '',
                trained_rmse=h['trained']['rmse_'+k][-1,bi,zi] if k in ('temp','salt') else ''))
    ax.invert_yaxis();ax.set_yscale('symlog',linthresh=100);ax.set_title(k+' ('+units[k]+')')
    ax.set_ylabel('Depth (m)');ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);fig.suptitle('Solid: paired state change; dashed: integrated direct NN source')
save(fig,'direct_vs_coupled.png')
with (out/'basin_depth.csv').open('w') as stream:
    w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=json.loads((out/'summary.json').read_text())
text='# Paired 27-hour diagnostic replay\n\n'
text+=f"Checkpoint update {meta['update']}; source SHA256 `{meta['sha256']}`.\n\n"
text+='Both runs use the same initialization, forcing and fixed interpolated target (27/720). Only the trained run applies bounded NN tendencies. No optimizer updates were performed.\n\n'
text+='| Run | Final normalized basin loss | Trajectory mean | Max CFL | Min salinity |\n|---|---:|---:|---:|---:|\n'
for k,s in summary.items():text+=f"| {k} | {s['final_normalized_loss']:.6g} | {s['trajectory_mean_normalized_loss']:.6g} | {s['max_cfl']:.6g} | {s['min_salinity']:.6g} |\n"
text+='\nBasin means can conceal local errors. Inspect pointwise RMSE alongside the trained objective. All depths are reported, including levels outside training. Momentum summaries use native staggered areas/masks. Map limits clip the strongest 1% for readability; raw fields are preserved. Direct-source residuals include coupled and numerical effects, not a closed process budget.\n\n'
for name in figures:text+=f'![{name}]({name})\n\n'
(out/'REPORT.md').write_text(text)
print('Saved report, CSV and',len(figures),'figures:',out/'REPORT.md')
