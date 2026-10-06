"""ERA5 subset planning and annual-to-window index conversion."""
import math


def plan_forcing_window(forcing_time, n_steps, dt, interval, period_count, margin=0):
    """Include both interpolation records through the rollout endpoint.

    Start at the record at or immediately before the initial time. A complete
    annual window also covers rollouts longer than one forcing cycle.
    """
    if n_steps < 0 or margin < 0 or dt <= 0 or interval <= 0 or period_count < 2:
        raise ValueError('Invalid forcing-window duration, margin, or period')
    start_time = forcing_time % (interval * period_count)
    first = math.floor(start_time / interval)
    last = math.floor((start_time + n_steps * dt) / interval)
    length = min(period_count, last - first + 2 + margin)
    return dict(start=first % period_count, length=length, period_count=period_count)


def local_forcing_indices(n1, n2, start, period_count, length, xp):
    """Return local indices and coverage flag; never wrap by subset length."""
    i1 = xp.mod(n1 - start, period_count)
    i2 = xp.mod(n2 - start, period_count)
    return i1, i2, (i1 < length) & (i2 < length)
