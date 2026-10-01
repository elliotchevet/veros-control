"""Add paired no-NN endpoint references to training history plots, without loading JAX."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('diagnostics',type=Path)
a=p.parse_args()
meta=json.loads((a.diagnostics/'metadata.json').read_text())
summary=json.loads((a.diagnostics/'summary.json').read_text())['baseline']
out=Path(meta['checkpoint']).parent
os.environ['MPLCONFIGDIR']=str(out/'matplotlib')
os.environ['MPLBACKEND']='Agg'
source=Path(__file__).resolve().parents[2]/'training/plot_train_checkpoint.py'
spec=importlib.util.spec_from_file_location('plot_checkpoint',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with (out/'train_history.csv').open() as f:
    rows=[{k:float(v) if v else float('nan') for k,v in r.items()} for r in csv.DictReader(f)]
assert meta['sample_steps']==meta['n_steps'], 'Endpoint reference requires endpoint-only training'
terms=dict(zip(('temp','salt'),summary['final_loss_terms']))
baseline=dict(normalized_loss=summary['final_normalized_loss'],terms=terms,
              normalized_terms={k:v/meta['denominators'][k] for k,v in terms.items()})
checkpoint=dict(n_steps=meta['n_steps'],loss_mode=f"GLORYS endpoint | lambda={meta['basin_pattern_lambda']:g} | job {out.name}")
m.plot_history(rows,checkpoint,str(out/'train_loss_with_baseline.png'),baseline)
m.plot_loss_terms(rows,checkpoint,str(out/'train_loss_terms_with_baseline.png'),baseline)
print(json.dumps(baseline,indent=2))
print(out/'train_loss_with_baseline.png')
print(out/'train_loss_terms_with_baseline.png')
