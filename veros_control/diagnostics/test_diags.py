"""Render paired evaluation curves and equal-time surface error maps."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from veros_control.diagnostics.metrics import depth_errors


def plot_evaluation(directory):
    out=Path(directory);config=json.loads((out/'config.json').read_text())
    histories={name:json.loads((out/f'{name}_history.json').read_text()) for name in ('baseline','trained') if (out/f'{name}_history.json').exists()}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for name,rows in histories.items():
        for ax,key in zip(axes.flat,('temp','salt','max_cfl','min_temperature')):
            ax.plot([r['step'] for r in rows],[r[key] for r in rows],label=name)
            ax.set_title(key);ax.set_xlabel('Step');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'rollout.png',dpi=150);plt.close(fig)
    if len(histories)!=2:return
    files={name:{int(p.stem.rsplit('_',1)[1]):p for p in out.glob(f'{name}_*.npz')} for name in histories}
    shared=set(files['baseline']) & set(files['trained'])
    if not shared:return
    step=max(shared);g=np.load(out/'grid.npz');level=int(np.argmin(g['depth']))
    alpha=step*config['dt_seconds']/(config['reference_hours']*3600)
    if config['loss']['interpolation']=='next':alpha=1.
    if config['loss']['interpolation']=='endpoint':alpha=config['n_steps']*config['dt_seconds']/(config['reference_hours']*3600)
    lon=(g['lon']+180)%360-180;indices=np.argsort(lon[2:-2])+2
    fields={k:np.load(v[step]) for k,v in files.items()};summary={}
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,key in enumerate(('temp','salt')):
        target=(1-alpha)*g['reference_initial_'+key]+alpha*g['reference_next_'+key]
        errors={name:np.where(g['wet'][...,level],data[key][...,level]-target[...,level],np.nan) for name,data in fields.items()}
        limit=max(float(np.nanquantile(np.abs(np.stack(list(errors.values()))),.99)),1e-12)
        for col,(name,error) in enumerate(errors.items()):
            ax=axes[row,col];ax.set_facecolor('.8')
            im=ax.pcolormesh(lon[indices],g['lat'][2:-2],error[indices,2:-2].T,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto')
            ax.set_title(f'{name}: {key} error');plt.colorbar(im,ax=ax)
            summary[name+'_'+key]={k:v.tolist() for k,v in depth_errors(fields[name][key],target,g['area'],g['wet']).items()}
    fig.suptitle(f'Shared step {step}; depth {g["depth"][level]:.1f} m; 99th-percentile color limits')
    fig.savefig(out/'surface_errors.png',dpi=150);plt.close(fig)
    (out/'depth_errors.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();plot_evaluation(a.directory)


if __name__=='__main__':main()
