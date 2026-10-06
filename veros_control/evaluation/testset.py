"""Evaluation uses the same grid mapping and forcing as training."""
from veros_control.data.dataset import create_simulation


def evaluation_case(config):
    return create_simulation(config)
