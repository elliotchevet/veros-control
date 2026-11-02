"""Saved training objective evaluated on monthly means (not mean hourly loss)."""
from pathlib import Path
from types import SimpleNamespace
import csv
import numpy as np
import jax.numpy as jnp
import xarray as xr
from veros_control.core.checkpoint import load_checkpoint
from veros_control.training.loss import make_basin_loss
from veros_control.diagnostics.annual import open_reference


def compute(run_dir, glorys_dir, years):
    root=Path(run_dir)
    checkpoint=load_checkpoint(root/'checkpoint.pkl')
    config=checkpoint['config']['loss']
    scale=np.asarray([checkpoint['loss_term_denominators'][k] for k in ('temp','salt')])
    rows=[]
    for label in ('baseline','trained'):
        with xr.open_dataset(root/f'{label}.averages.nc',decode_times=False) as data:
            pad=lambda x:np.pad(np.asarray(x).T,((2,2),(2,2),(0,0)))
            variables=SimpleNamespace(xt=np.pad(data.xt.values,(2,2)),yt=np.pad(data.yt.values,(2,2)),
                zt=data.zt.values,dzt=data.dzt.values,maskT=pad(data.maskT),
                area_t=np.pad(data.area_t.values.T,((2,2),(2,2))),tau=0)
            for index,year in enumerate(data.simulation_year.values):
                if int(year) not in years:continue
                month=int(data.simulation_month.values[index])
                reference=open_reference(Path(glorys_dir)/f'GLORYS12_1deg_{year}{month:02d}_processed.nc',data)
                target={k:pad(reference[k]) for k in ('temp','salt')}
                for key in target:setattr(variables,key,jnp.asarray(pad(data[key].isel(Time=index))))
                hours=float(data.Time.values[index])*24
                _,_,terms,_=make_basin_loss(variables,target,levels=config['levels'],
                    pattern_lambda=config['pattern_lambda'],mode=config['mode'],
                    depth_loss=config.get('depth_loss'),spatial_filter=config.get('spatial_filter'),
                    lead_hours=lambda state:hours)
                raw=np.asarray(terms(SimpleNamespace(variables=variables)))
                if not np.isfinite(raw).all():raise ValueError('Nonfinite monthly loss')
                rows.append(dict(trajectory=label,year=int(year),month=month,
                    temp=float(raw[0]/scale[0]),salt=float(raw[1]/scale[1]),loss=float(np.mean(raw/scale))))
    if not rows:raise ValueError('No monthly records in selected years')
    with (root/'rollout_loss.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    for ax,key in zip(axes,('loss','temp','salt')):
        for label in ('baseline','trained'):
            selected=[r for r in rows if r['trajectory']==label]
            ax.plot([r['year']+(r['month']-.5)/12 for r in selected],[r[key] for r in selected],label=label)
        ax.set_ylabel(key);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('Model year')
    fig.suptitle('Training objective on monthly means; saved normalization')
    for ext in ('png','pdf'):fig.savefig(root/f'rollout_loss.{ext}',dpi=150)
    plt.close(fig)
    return rows
