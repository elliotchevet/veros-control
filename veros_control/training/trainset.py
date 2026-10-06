"""A training case uses the shared initialization and reference loader."""
from veros_control.data.dataset import create_simulation


def training_case(config):
    return create_simulation(config)
