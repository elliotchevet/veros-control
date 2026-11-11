"""Train with monthly batches, fixed-panel checkpoints, and resumable curriculum."""
from dataclasses import asdict, replace
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
from veros_control.training.config import load_config, objective_signature
from veros_control.training.trainset import training_case
from veros_control.training.batches import BatchObjective, batch_months, case_configs
from veros_control.training.scheduler import backtrack_candidate, loss_test_next_learning_rate, recover_learning_rate


def finite(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))


def train(config):
    """Run one stage, or an ordered curriculum carrying latest Adam/weights/LR."""
    config.validate_options()
    if config.curriculum:
        previous = config.transition
        result = None
        for index, stage in enumerate(config.curriculum):
            stage_config = replace(config, curriculum=[], n_steps=stage.n_steps,
                updates=stage.updates, output=str(Path(config.output)/f'window_{stage.n_steps}'),
                transition=previous, resume=config.resume if index == 0 else None,
                weights=config.weights if index == 0 else None,
                stage_initial_rate=stage.initial_rate)
            result = train_stage(stage_config)
            previous = str(best_path(result) if config.transition_checkpoint == 'best' else result)
        return result
    return train_stage(config)


def train_stage(config):
    from veros import runtime_settings
    if runtime_settings.backend != 'jax':
        raise ValueError('Training requires VEROS_BACKEND=jax before importing Veros')
    config.validate_options()
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output/'latest.pkl'
    if latest.exists() or best_path(latest).exists():
        raise FileExistsError('Use a new output directory; resume reads a separate checkpoint')
    source_path = config.resume or config.transition or config.weights
    source = load_checkpoint(source_path) if source_path else None
    carry = bool(config.resume or config.transition)
    if carry:
        if source.get('format_version') != 1:
            raise ValueError('Full resume/transition requires a public-format checkpoint')
        expected, actual = dict(source['objective_signature']), objective_signature(config)
        expected['scheduler'] = dict(expected['scheduler'])
        expected['scheduler'].setdefault('recovery_interval', 10)
        expected['scheduler'].setdefault('reject_worsening', False)
        if config.transition:
            expected.pop('n_steps', None); actual.pop('n_steps', None)
            expected.pop('scheduler', None); actual.pop('scheduler', None)
        if expected != actual:
            raise ValueError('Resume/transition objective differs; use weights for a fresh experiment')
        if source['config'].get('gradient_clip') != config.gradient_clip:
            raise ValueError('Changing gradient clipping changes Adam state')
    simulation, _ = training_case(next(iter(case_configs(config).values())))
    if source:
        model, config.model, _, stats, config.normalization = model_from_checkpoint(source, simulation.state.settings.nz)
        params = source['params']
    else:
        model, params = initialize(config.model, simulation.state.settings.nz, config.seed)
        stats = None
    config.validate_options()
    with simulation.state.settings.unlock():
        simulation.state.settings.use_forcings = config.model.use_forcings
        simulation.state.settings.use_date_embedding = config.model.use_date_embedding
    simulation.set_forcing(simulation.state)
    if source is None and config.normalization != 'none':
        stats = fit_statistics(simulation.state, config.normalization, config.model.use_forcings)
    physical = physical_model(model, config.correction, simulation.state.variables.zt)
    objective = BatchObjective(config, simulation, physical, stats)
    if config.worker_commands:
        from veros_control.training.workers import ParallelObjective
        objective = ParallelObjective(objective)
    try:
        objective.member(objective.months[0])
        names = objective.names
        optimizer = optax.adamw(1.)
        if config.gradient_clip is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(config.gradient_clip), optimizer)
        opt_state = source['opt_state'] if carry else optimizer.init(params)
        start = source['update'] if config.resume else 0
        history = list(source['train_history']) if config.resume else []
        rate = source.get('learning_rate', config.learning_rate) if carry else config.learning_rate
        if carry and source['config']['learning_rate'] != config.learning_rate:
            raise ValueError('Resume/transition must preserve the LR ceiling')
        if config.stage_initial_rate is not None:
            rate = config.stage_initial_rate
        streak = source.get('lr_clean_updates', 0) if config.resume else 0
        valid, report = objective.validate(objective.panel, params)
        if not valid:
            raise FloatingPointError(f'Initial rollout failed: {report}')
        if config.resume or (config.transition and config.fixed_loss_balance):
            denominators = source['loss_term_denominators']
            calibration = source.get('baseline_calibration')
        else:
            calibration = None
            if config.loss.balance and config.loss.normalization != 'none':
                calibration_params = jax.tree_util.tree_map(jnp.zeros_like, params) if config.loss.normalization == 'no_nn' else params
                _, raw = objective.mean(objective.months, calibration_params, jnp.ones(len(names)))
                raw = np.asarray(raw)
                if not finite(raw) or np.any(raw < 0) or (config.loss.normalization == 'no_nn' and np.any(raw <= 0)):
                    raise FloatingPointError('Invalid loss calibration')
                values = raw if config.loss.normalization == 'no_nn' else raw + 1e-12
                denominators = dict(zip(names, map(float, values)))
                calibration = dict(mode=config.loss.normalization, months=list(objective.months),
                    n_steps=config.n_steps, denominators=denominators)
            else:
                denominators = dict.fromkeys(names, 1.)
        if calibration is not None:
            (output/'loss_calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
        scale = jnp.asarray([denominators[k] for k in names])
        if config.resume and best_path(config.resume).exists():
            shutil.copy2(best_path(config.resume), best_path(latest))
        tracker = BestLatestCheckpoint(latest)
        signature = objective_signature(config)
        (output/'config.json').write_text(json.dumps(asdict(config), indent=2)+'\n')

        def save(update, loss, raw):
            payload = dict(format_version=1, params=params, opt_state=opt_state, update=update,
                config=asdict(config), objective_signature=signature, stats_features=stats,
                loss_term_denominators=denominators, loss_metadata=objective.metadata, train_history=history,
                learning_rate=rate, lr_clean_updates=streak, baseline_calibration=calibration)
            tracker.save(payload, loss, dict(zip(names, map(float, raw))))
            print(f'update={update} reference_loss={float(loss):.9g} next_lr={rate:.6g} best_update={tracker.best_payload["update"]}', flush=True)

        panel_loss, panel_terms = objective.mean(objective.panel, params, scale)
        save(start, panel_loss, panel_terms)
        for update in range(start+1, start+config.updates+1):
            started = time.monotonic()
            months = batch_months(objective.months, config.batch_size, config.seed, update-1)
            (loss, before_terms), grads = objective.mean(months, params, scale, gradient=True)
            if not finite((loss, grads)):
                raise FloatingPointError('Nonfinite loss/gradient; saved states preserved')
            direction, new_opt_state = optimizer.update(grads, opt_state, params)
            if not finite(new_opt_state):
                raise FloatingPointError('Nonfinite optimizer state')
            score = lambda p: objective.mean(months, p, scale)
            validate = lambda p: objective.validate(months, p)
            mode = config.scheduler.mode
            candidate, _, applied_rate, retries, report = backtrack_candidate(
                params, direction, rate, validate,
                max_retries=0 if mode == 'keep_update' else config.candidate_retries,
                score=score, reference_loss=float(loss),
                min_learning_rate=config.scheduler.min_rate if mode != 'constant' else 0.,
                enforce_loss_decrease=mode == 'backtracking' and update >= config.scheduler.start_update)
            post_loss, post_terms = float(report['loss']), np.asarray(report['loss_terms'])
            scheduler_report = {}
            if mode == 'keep_update':
                rate, scheduler_report = loss_test_next_learning_rate(applied_rate, float(loss), post_loss, update,
                    min_rate=config.scheduler.min_rate, max_rate=config.learning_rate,
                    clean_updates=streak, start_epoch=config.scheduler.start_update,
                    recovery_interval=config.scheduler.recovery_interval)
                streak = scheduler_report['lr_clean_updates']
                if config.scheduler.reject_worsening and scheduler_report['loss_test_active'] and not scheduler_report['loss_test_passed']:
                    candidate, new_opt_state = params, opt_state
                    post_loss, post_terms = float(loss), np.asarray(before_terms)
                    scheduler_report['update_kept'] = False
            elif mode == 'backtracking':
                backoff, streak = recover_learning_rate(rate/config.learning_rate, retries, streak)
                rate = max(config.scheduler.min_rate, config.learning_rate*backoff)
            else:
                rate = applied_rate
            # Score before committing so failed fixed-panel evaluations leave the saved state intact.
            panel_loss, panel_terms = objective.mean(objective.panel, candidate, scale)
            if not finite((panel_loss, panel_terms)):
                raise FloatingPointError('Nonfinite fixed-panel score')
            params, opt_state = candidate, new_opt_state
            history.append(dict(update=update, loss=float(loss), post_update_loss=post_loss,
                post_update_loss_terms=dict(zip(names, map(float, post_terms))), learning_rate=applied_rate,
                reference_loss=float(panel_loss), reference_loss_terms=dict(zip(names, map(float, panel_terms))),
                batch_months=list(months), next_lr=rate, scheduler=scheduler_report,
                gradient_norm=float(optax.global_norm(grads)), validation=report, seconds=time.monotonic()-started))
            save(update, panel_loss, panel_terms)
        return latest
    finally:
        objective.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='JSON training configuration')
    print(train(load_config(parser.parse_args().config)))


if __name__ == '__main__':
    main()
