"""Frozen-weight paired trajectories; hourly metrics and recoverable snapshots."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
from veros_control.core.checkpoint import load_checkpoint, model_from_checkpoint
from veros_control.core.correction import physical_model
from veros_control.core.rollout import Rollout, solver_cache_scope
from veros_control.training.config import load_config
from veros_control.training.loss import build_loss
from veros_control.training.stability import state_metrics, valid_metrics, report_metrics
from veros_control.evaluation.config import EvaluationConfig
from veros_control.evaluation.testset import evaluation_case


def infer(config):
    from veros import runtime_settings
    if runtime_settings.backend != "jax":
        raise ValueError("Evaluation requires VEROS_BACKEND=jax before importing Veros")
    output=Path(config.output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a new evaluation directory')
    output.mkdir(parents=True,exist_ok=True)
    source=load_checkpoint(config.checkpoint)
    # Read architecture before setup so forcing feature settings are consistent.
    _,config.model,config.correction,stats,config.normalization=model_from_checkpoint(source,1)
    simulation,reference=evaluation_case(config)
    model,_,_,_,_=model_from_checkpoint(source,simulation.state.settings.nz)
    physical=physical_model(model,config.correction,simulation.state.variables.zt)
    names,terms,metadata=build_loss(simulation.state,reference,config.loss,config.n_steps)
    config_data=asdict(config)
    config_data["dt_seconds"]=float(simulation.state.settings.dt_tracer)
    config_data['checkpoint_sha256']=hashlib.sha256(Path(config.checkpoint).read_bytes()).hexdigest()
    (output/'config.json').write_text(json.dumps(config_data,indent=2)+'\n')
    v=simulation.state.variables
    np.savez_compressed(output/'grid.npz',lon=np.asarray(v.xt),lat=np.asarray(v.yt),depth=np.abs(np.asarray(v.zt)),
        area=np.asarray(v.area_t),wet=np.asarray(v.maskT),**{'reference_initial_'+k:np.asarray(x) for k,x in reference.initial.items()},
        **{'reference_next_'+k:np.asarray(x) for k,x in reference.target.items()})
    summary={}
    cases=[('baseline',lambda p,x,c,d:jnp.zeros((*x.shape[:2],4),dtype=x.dtype))] if config.baseline else []
    cases.append(('trained',physical))
    for label,apply in cases:
        rollout=Rollout(simulation,apply,stats,config.normalization,config.model.use_forcings)
        state=rollout.initial
        history=[]
        status='complete'
        for step in range(config.n_steps+1):
            if step:
                with solver_cache_scope():
                    state=rollout.step(state,source['params'])
                    jax.block_until_ready(state.variables.temp)
            values=np.asarray(state_metrics(state));loss_terms=np.asarray(terms(state))
            history.append(dict(step=step,**report_metrics(values),**dict(zip(names,map(float,loss_terms)))))
            valid=valid_metrics(values,config.max_cfl) and np.isfinite(loss_terms).all()
            if not valid:status='failed'
            if step % config.snapshot_every==0 or step==config.n_steps or not valid:
                vs=state.variables
                np.savez_compressed(output/f'{label}_{step:05d}.npz',**{k:np.asarray(getattr(vs,k)[...,vs.tau]) for k in ('temp','salt','u','v','w')})
                (output/f'{label}_history.json').write_text(json.dumps(history,indent=2)+'\n')
            print(label,history[-1],flush=True)
            if not valid:break
        summary[label]=dict(status=status,last_step=step,last_metrics=history[-1])
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        del rollout,state
        jax.clear_caches()
    if hashlib.sha256(Path(config.checkpoint).read_bytes()).hexdigest()!=config_data['checkpoint_sha256']:
        raise RuntimeError('Source checkpoint changed during evaluation')
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');a=p.parse_args()
    print(infer(load_config(a.config,EvaluationConfig)))


if __name__=='__main__':main()
