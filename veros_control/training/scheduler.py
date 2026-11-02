"""Candidate acceptance and LR schedules shared with main (9f4fb66)."""
import jax
import jax.numpy as jnp
import numpy as np

def backtrack_candidate(params, direction, learning_rate, validate, max_retries=8, factor=0.5,
                        score=None, reference_loss=None, loss_rtol=1e-8, loss_atol=1e-10,
                        min_learning_rate=0., enforce_loss_decrease=True):
    """Try one Adam direction at smaller rates; never mutate optimizer state.

    validate returns (valid, diagnostic). Finite parameters are checked before
    simulation. Exhaustion raises, leaving the caller's parameters/state intact.
    """
    if score is not None and (reference_loss is None or not np.isfinite(reference_loss)):
        raise ValueError('Loss backtracking requires a finite reference loss')
    if not np.isfinite(min_learning_rate) or min_learning_rate < 0:
        raise ValueError("Minimum learning rate must be finite and nonnegative")
    rate = max(learning_rate, min_learning_rate)
    for retry in range(max_retries + 1):
        updates = jax.tree_util.tree_map(lambda x: rate * x, direction)
        candidate = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(candidate))
        valid, diagnostic = validate(candidate) if finite else (False, 'nonfinite candidate parameters')
        if valid and score is not None:
            value, terms = score(candidate)
            value = float(value)
            terms = np.asarray(terms)
            limit = reference_loss + loss_atol + loss_rtol * abs(reference_loss)
            require_decrease = enforce_loss_decrease and (min_learning_rate == 0 or rate > min_learning_rate)
            valid = bool(np.isfinite(value) and np.isfinite(terms).all() and (not require_decrease or value <= limit))
            diagnostic = dict(diagnostic, loss=value, loss_limit=limit, loss_terms=terms.tolist())
        print(f'  candidate lr={rate:.6e} valid={valid}: {diagnostic}', flush=True)
        if valid:
            return candidate, updates, rate, retry, diagnostic
        if min_learning_rate > 0 and rate <= min_learning_rate:
            break
        rate = max(min_learning_rate, rate * factor)
    raise FloatingPointError(f'Candidate rollout rejected after {max_retries + 1} attempts; previous parameters and optimizer state preserved')



def loss_test_next_learning_rate(rate, before, after, epoch, *, min_rate=1e-5,
                                 max_rate=1e-3, clean_updates=0, start_epoch=20,
                                 rtol=1e-8, atol=1e-10):
    """Keep stable updates; halve next LR on failure, recover after 10 improvements."""
    if not all(np.isfinite(x) for x in (rate, before, after, min_rate, max_rate)):
        raise ValueError('Loss scheduler requires finite rates and losses')
    if not 0 <= min_rate <= rate <= max_rate or rate <= 0 or epoch < 1 or start_epoch < 1 or clean_updates < 0:
        raise ValueError('Invalid rate, floor, ceiling, counter, or epoch')
    passed = bool(after <= before + atol + rtol * abs(before))
    tested = epoch >= start_epoch
    next_rate = rate
    action = 'warmup' if not tested else 'hold'
    if not tested:
        clean_updates = 0
    elif not passed:
        next_rate = max(min_rate, rate * .5)
        clean_updates = 0
        action = 'halve'
    else:
        clean_updates = clean_updates + 1 if after < before else 0
        if clean_updates >= 10:
            next_rate = min(max_rate, rate * 1.5)
            clean_updates = 0
            action = 'recover'
    return next_rate, dict(loss_test_active=tested, loss_test_passed=passed,
                           next_lr=next_rate, update_kept=True,
                           lr_clean_updates=clean_updates, lr_action=action)


def recover_learning_rate(backoff, retries, clean_updates):
    """Persist reductions; recover 50% after five consecutive first-try acceptances."""
    backoff *= 0.5 ** retries
    clean_updates = 0 if retries else clean_updates + 1
    if clean_updates >= 5:
        backoff = min(1., backoff * 1.5)
        clean_updates = 0
    return backoff, clean_updates
