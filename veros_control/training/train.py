"""Train a column correction with one sampled, checkpointed reverse-mode path."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import shutil
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from veros_control.core.checkpoint import BestLatestCheckpoint, best_path, load_checkpoint, model_from_checkpoint
from veros_control.core.model import initialize
from veros_control.core.correction import physical_model
from veros_control.core.preprocess import fit_statistics
from veros_control.core.rollout import Rollout, solver_cache_scope
from veros_control.training.config import load_config, objective_signature
from veros_control.training.trainset import training_case
from veros_control.training.loss import build_loss
from veros_control.training.stability import make_validator, backtrack


def train(config):
    from veros import runtime_settings
    if runtime_settings.backend != "jax":
        raise ValueError("Training requires VEROS_BACKEND=jax before importing Veros")
    config.validate_options()
    output = Path(config.output)
    output.mkdir(parents=True,exist_ok=True)
    latest = output/'latest.pkl'
    if latest.exists() or best_path(latest).exists():
        raise FileExistsError('Use a new output directory; resume reads a separate checkpoint')
    source = load_checkpoint(config.resume or config.weights) if config.resume or config.weights else None
    if config.resume:
        if source.get('format_version') != 1:
            raise ValueError('Legacy checkpoints support weights-only initialization; full resume requires the public format')
        if source['objective_signature'] != objective_signature(config):
            raise ValueError('Resume objective differs: use weights for a fresh objective')
    simulation,reference = training_case(config)
    if source:
        model,model_config,_,stats,normalization = model_from_checkpoint(source,simulation.state.settings.nz)
        # Architecture and preprocessing must match the loaded parameters.
        config.model,config.normalization = model_config,normalization
        params = source['params']
    else:
        model,params = initialize(config.model,simulation.state.settings.nz,config.seed)
        stats = None
    with simulation.state.settings.unlock():
        simulation.state.settings.use_forcings = config.model.use_forcings
    simulation.set_forcing(simulation.state)
    if source is None and config.normalization != 'none':
        stats = fit_statistics(simulation.state,config.normalization,config.model.use_forcings)
    apply_model = physical_model(model,config.correction,simulation.state.variables.zt)
    rollout = Rollout(simulation,apply_model,stats,config.normalization,config.model.use_forcings)
    names,terms,loss_metadata = build_loss(simulation.state,reference,config.loss,config.n_steps)
    evaluate,gradient = rollout.objective(terms,config.n_steps,config.sample_steps,config.chunk_base)
    optimizer = optax.adamw(1.)
    if config.gradient_clip is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip),optimizer)
    if config.resume and source['config'].get('gradient_clip') != config.gradient_clip:
        raise ValueError('Changing gradient clipping changes optimizer state; use weights instead')
    opt_state = source['opt_state'] if config.resume else optimizer.init(params)
    start = source['update'] if config.resume else 0
    history = list(source['train_history']) if config.resume else []
    validate = make_validator(rollout,config.n_steps,config.max_cfl) if config.validate else lambda p:(True,{})
    valid,report = validate(params)
    if not valid:raise FloatingPointError(f'Initial rollout failed: {report}')
    with solver_cache_scope():
        _,raw = evaluate(params,jnp.ones(len(names)))
        raw = np.asarray(raw)
    if not np.isfinite(raw).all():raise FloatingPointError('Nonfinite initial loss')
    denominators = source['loss_term_denominators'] if config.resume else dict(zip(names,raw+1e-12 if config.loss.balance else np.ones(len(names))))
    scale = jnp.asarray([denominators[k] for k in names])
    if config.resume and best_path(config.resume).exists():
        shutil.copy2(best_path(config.resume),best_path(latest))
    tracker = BestLatestCheckpoint(latest)
    rate = source.get("learning_rate",config.learning_rate) if config.resume and config.learning_rate == source["config"]["learning_rate"] else config.learning_rate
    signature = objective_signature(config)
    (output/'config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
    def save(update,loss,raw):
        payload = dict(format_version=1,params=params,opt_state=opt_state,update=update,
            config=asdict(config),objective_signature=signature,stats_features=stats,
            loss_term_denominators=denominators,loss_metadata=loss_metadata,train_history=history,
            learning_rate=rate)
        tracker.save(payload,loss,dict(zip(names,map(float,raw))))
        print(f'update={update} loss={float(loss):.9g} best={tracker.best_loss:.9g} best_update={tracker.best_payload["update"]}',flush=True)
    save(start,float(np.mean(raw/np.asarray(scale))),raw)
    for update in range(start+1,start+config.updates+1):
        started = time.monotonic()
        with solver_cache_scope():
            (loss,_),grads = gradient(params,scale)
            jax.block_until_ready(grads)
        if not np.isfinite(float(loss)) or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(grads)):
            raise FloatingPointError('Nonfinite loss/gradient; saved states preserved')
        direction,new_opt_state = optimizer.update(grads,opt_state,params)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(new_opt_state)):
            raise FloatingPointError('Nonfinite optimizer state')
        candidate,rate,report = backtrack(params,direction,rate,validate,config.candidate_retries)
        with solver_cache_scope():
            post_loss,post_terms = evaluate(candidate,scale)
            post_loss,post_terms = float(post_loss),np.asarray(post_terms)
        if not np.isfinite(post_loss) or not np.isfinite(post_terms).all():
            raise FloatingPointError('Nonfinite candidate loss; saved states preserved')
        params,opt_state = candidate,new_opt_state
        history.append(dict(update=update,loss=float(loss),post_update_loss=post_loss,
            post_update_loss_terms=dict(zip(names,map(float,post_terms))),learning_rate=rate,
            gradient_norm=float(optax.global_norm(grads)),validation=report,seconds=time.monotonic()-started))
        save(update,post_loss,post_terms)
    return latest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',help='JSON training configuration')
    args=parser.parse_args()
    print(train(load_config(args.config)))


if __name__=='__main__':main()
