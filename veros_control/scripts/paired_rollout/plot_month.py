"""Plot complete or partially completed monthly paired inference, without JAX."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('diagnostics', type=Path)
p.add_argument('--output', type=Path, help='Optional separate plot directory')
a = p.parse_args()
root = a.diagnostics
out = a.output or root
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((root/'metadata.json').read_text())
summary = json.loads((root/'summary.json').read_text())
g = np.load(root/'grid_target.npz')
h = {label:np.load(root/f'{label}_history.npz') for label in ('baseline','trained') if (root/f'{label}_history.npz').exists()}
levels = np.asarray(meta['trained_level_indices'])
figures = []
def save(fig, name):
    fig.savefig(out/name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures.append(name)

fig, axes = plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
for label, hist in h.items():
    days = np.arange(len(hist['normalized_loss']))*meta['dt_seconds']/86400
    axes[0,0].plot(days,hist['normalized_loss'],label=label)
    if len(hist.get('cfl',[])):
        axes[0,1].plot(days[1:],np.max(hist['cfl'],axis=1),label=label)
        axes[1,0].plot(days[1:],hist['salt_min'],label=label)
        for i,k in enumerate(('temp','salt')):
            axes[1,1].plot(days,hist['loss_terms'][:,i]/meta['denominators'][k],label=f'{label}: {k}')
axes[0,1].axhline(1,color='k',ls='--',alpha=.5)
for ax,title in zip(axes.flat,['Normalized top-five-level objective','Maximum directional CFL','Minimum wet-cell salinity (g/kg)','Normalized temperature and salinity terms']):
    ax.set_title(title);ax.set_xlabel('Model days');ax.grid(alpha=.2);ax.legend()
fig.suptitle(f"Frozen {meta.get('training_n_steps',27)}-hour weights: longer inference\nErrors use interpolated monthly GLORYS; day 30 uses the next monthly field")
save(fig,'month_curves.png')

fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
colors=plt.get_cmap('tab10').colors
for row,k in enumerate(('temp','salt')):
    for label,hist in h.items():
        days=np.arange(len(hist['normalized_loss']))*meta['dt_seconds']/86400
        for b,name in enumerate(meta['basins']):
            for col,metric in enumerate(('bias','rmse')):
                values=hist[metric+'_'+k][:,b,:][:,levels]
                curve=np.nanmean(values,axis=1) if metric=='bias' else np.sqrt(np.nanmean(values**2,axis=1))
                axes[row,col].plot(days,curve,color=colors[b],ls='-' if label=='trained' else '--',label=name.replace('_',' ') if label=='trained' else None)
    for col,metric in enumerate(('signed bias','pointwise RMSE')):
        axes[row,col].set_title(f'{k}: top-five-level {metric}')
        axes[row,col].set_xlabel('Model days');axes[row,col].grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('Basin errors versus interpolated monthly GLORYS: solid trained, dashed no NN')
save(fig,'month_basin_errors.png')

# Only compare snapshots at the same elapsed time, including when one run failed.
def snapshots(label):
    return {int(p.stem.split('step')[-1]):p for p in root.glob(f'{label}_step*.npz')}
common=set(snapshots('baseline')) & set(snapshots('trained'))
comparison_step=max(common) if common else None
if comparison_step:
    fields={label:np.load(snapshots(label)[comparison_step]) for label in ('baseline','trained')}
    depth=np.asarray(g['depth']);level=int(np.argmin(depth))
    lon=(g['lon']+180)%360-180;ix=np.argsort(lon[2:-2])+2
    fig,axes=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
    rows=[]
    for row,k in enumerate(('temp','salt')):
        target=(1-comparison_step/720)*g['glorys_initial_'+k]+(comparison_step/720)*g['glorys_next_'+k]
        errors={label:fields[label][k]-target for label in fields}
        wet=g['wet_'+k][...,level]
        vals=[np.where(wet,errors[label][...,level],np.nan) for label in ('baseline','trained')]
        lim=max(float(np.nanquantile(np.abs(np.stack(vals)),.99)),1e-15)
        vals.append(np.where(wet,errors['trained'][...,level]**2-errors['baseline'][...,level]**2,np.nan))
        for col,(value,title) in enumerate(zip(vals,['No NN − GLORYS','Trained − GLORYS','Change in squared error (blue improves)'])):
            bound=lim if col<2 else max(float(np.nanquantile(np.abs(value),.99)),1e-15)
            ax=axes[row,col];ax.set_facecolor('.8')
            im=ax.pcolormesh(lon[ix],g['lat'][2:-2],value[ix,2:-2].T,cmap='RdBu_r',vmin=-bound,vmax=bound,shading='auto')
            ax.set_title(k+': '+title);plt.colorbar(im,ax=ax,shrink=.8,extend='both')
        for label in fields:
            weight=np.where(g['wet_'+k],g['area_'+k][...,None],0.)
            for zi in range(len(depth)):
                w=weight[...,zi];error=np.where(w>0,errors[label][...,zi],0.);den=w.sum()
                rows.append(dict(run=label,step=comparison_step,channel=k,depth_m=float(depth[zi]),
                    bias=float((w*error).sum()/den) if den else np.nan,
                    rmse=float(np.sqrt((w*error**2).sum()/den)) if den else np.nan))
    fig.suptitle(f'Latest shared snapshot: hour {comparison_step}, depth {depth[level]:.1f} m; display clips strongest 1%')
    save(fig,'month_surface_comparison.png')
    with (out/'month_depth_errors.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

report='# One-month frozen-weight inference\n\n'
report+=f"Source: `{meta['checkpoint']}`, update {meta['update']}; requested horizon {meta['n_steps']} hours.\n\n"
report+='The next monthly GLORYS field is the day-30 reference. Intermediate references are linear interpolations, not observed trajectories. Monthly means are not instantaneous month-boundary observations. Normalization remains that saved in the source training checkpoint.\n\n'
for label,status in summary.items():
    report+=f"- **{label}**: {status.get('status','unknown')}; {status.get('completed_steps','unknown')} completed steps. Failure: {status.get('failure')}.\n"
report+=f'\nLatest shared saved snapshot: {comparison_step} hours. Comparisons never mix different elapsed times.\n\n'
for name in figures:report+=f'![{name}]({name})\n\n'
(out/'MONTH_REPORT.md').write_text(report)
print(out/'MONTH_REPORT.md',flush=True)
