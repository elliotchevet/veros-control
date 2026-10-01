"""Map endpoint surface basin improvements from saved paired diagnostics."""
import argparse
import csv
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('diagnostics', type=Path)
a = p.parse_args()
out = a.diagnostics.resolve()
os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

cache = Path(__file__).resolve().parents[1] / 'basin_maps/cartopy_data'
cartopy.config['data_dir'] = cache
cartopy.config['pre_existing_data_dir'] = cache
meta = json.loads((out / 'metadata.json').read_text())
with (out / 'basin_depth.csv').open() as f:
    rows = list(csv.DictReader(f))
depth = min(float(r['depth_m']) for r in rows)
rows = [r for r in rows if float(r['depth_m']) == depth and r['channel'] in ('temp', 'salt')]
records = []
for r in rows:
    record = dict(channel=r['channel'], basin=r['basin'], depth_m=depth)
    for metric in ('bias', 'rmse'):
        baseline, trained = (abs(float(r[f'{case}_{metric}'])) for case in ('baseline', 'trained'))
        record[f'baseline_{metric}'] = baseline
        record[f'trained_{metric}'] = trained
        record[f'{metric}_improvement_pct'] = 100 * (1 - trained / baseline) if baseline > 0 else float('nan')
    records.append(record)
with (out / 'surface_basin_improvement.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader(); writer.writerows(records)
boxes = {
    'North Atlantic': [(-70, 0, 90, 90)],
    'South Atlantic': [(-70, -60, 90, 60)],
    'North Pacific': [(-180, 0, 110, 90), (120, 0, 60, 90)],
    'South Pacific': [(-180, -60, 110, 60), (120, -60, 60, 60)],
    'Indian': [(20, -60, 100, 90)],
    'Southern': [(-180, -90, 360, 30)],
}
labels = {'North Atlantic': (-30, 35), 'South Atlantic': (-28, -30),
          'North Pacific': (-140, 35), 'South Pacific': (-140, -30),
          'Indian': (75, -25), 'Southern': (0, -73)}
source_job = Path(meta['checkpoint']).parent.name.replace('job_', '')
for metric, description in [('bias', 'Absolute basin-mean bias'), ('rmse', 'Area-weighted spatial RMSE')]:
    values = [r[f'{metric}_improvement_pct'] for r in records]
    extent = max(100., max(abs(v) for v in values))
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent)
    cmap = plt.get_cmap('BrBG')
    fig, axes = plt.subplots(2, 1, figsize=(13, 11), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax, channel, title in zip(axes, ('temp', 'salt'), ('Temperature', 'Salinity')):
        ax.set_global(); ax.set_facecolor('white')
        for r in records:
            if r['channel'] != channel:
                continue
            value = r[f'{metric}_improvement_pct']
            for x, y, w, h in boxes[r['basin']]:
                ax.add_patch(Rectangle((x,y), w,h, facecolor=cmap(norm(value)), edgecolor='none', transform=ccrs.PlateCarree(), zorder=0))
        ax.add_feature(cfeature.LAND.with_scale('110m'), facecolor='#c6c6c6', edgecolor='none', zorder=1)
        ax.coastlines(resolution='110m', linewidth=.35, color='#666666', zorder=2)
        for r in records:
            if r['channel'] != channel:
                continue
            for x,y,w,h in boxes[r['basin']]:
                ax.add_patch(Rectangle((x,y), w,h, fill=False, edgecolor='#333333', linewidth=.8, transform=ccrs.PlateCarree(), zorder=3))
            x,y = labels[r['basin']]
            value = r[f'{metric}_improvement_pct']
            label = r['basin'] + '\n' + f'{value:+.1f}%'
            ax.text(x,y,label,ha='center',va='center',fontsize=11,fontweight='bold',transform=ccrs.PlateCarree(),bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=3),zorder=4)
        ax.set_title(title, fontsize=15, loc='left')
    fig.subplots_adjust(top=.88,bottom=.13,hspace=.17,left=.035,right=.965)
    fig.suptitle(f'Surface improvement over the no-NN baseline\n{description}',fontsize=18,y=.98)
    fig.text(.5,.915,f'Training job {source_job}, update {meta["update"]} | Endpoint: {meta["n_steps"]} hours | Surface: {depth:.1f} m',ha='center',fontsize=11)
    cax = fig.add_axes([.25,.09,.50,.017])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal')
    cb.set_label('Improvement (%) — negative: worse; positive: better',fontsize=10)
    fig.text(.5,.028,'100 × (baseline error − trained error) / baseline error. Wet-cell area weighting; same interpolated GLORYS target.\nRectangular training domains; Pacific wraps across the map edge. White ocean areas are outside the selected domains.',ha='center',fontsize=9)
    path = out / f'surface_basin_{metric}_improvement.png'
    fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)
    print(path)
for r in records:
    print(r['channel'],r['basin'],f'bias {r["bias_improvement_pct"]:+.2f}%',f'RMSE {r["rmse_improvement_pct"]:+.2f}%')
