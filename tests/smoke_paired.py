"""Check real monthly paired output, including correction second moments."""
import argparse
from pathlib import Path
import json
import numpy as np
import xarray as xr
from veros_control.diagnostics.paired import PairedConfig, run
from veros_control.diagnostics.corrections import plot_corrections


def check(training, output):
    training=Path(training).resolve()
    config=PairedConfig(output=output, setup=str(Path(__file__).with_name('smoke_setup.py').resolve())+':create_setup',
        initial=str(training/'initial.nc'),target=str(training/'target.nc'),
        checkpoint=str(training/'surface/window_243/latest.pkl'),months=1,max_cfl=1.)
    root=run(config)
    summary=json.loads((root/'summary.json').read_text())
    assert all(summary[k]['status']=='complete' for k in ('baseline','trained')),summary
    for label in ('baseline','trained'):
        with xr.open_dataset(root/f'{label}.averages.nc',decode_times=False) as ds:
            assert ds.sizes['Time']==1 and int(ds.sample_count[0])==720
            assert np.isfinite(ds.temp.where(ds.maskT.astype(bool),0)).all()
            assert 'dtemp_nn2' in ds
            if label=='baseline':np.testing.assert_allclose(ds.dtemp_nn2.fillna(0),0.)
    plot_corrections(root)
    assert (root/'corrections.pdf').is_file()
    print('PASS: paired real model, monthly archive and correction report')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('training');p.add_argument('output');a=p.parse_args();check(a.training,a.output)
