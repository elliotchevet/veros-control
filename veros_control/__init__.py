"""Differentiable column corrections for Veros."""
from veros_control.control import control
from veros_control.set_inits import set_inits
from veros_control.variables import VARIABLES
from veros_control.settings import SETTINGS

__version__ = '0.2.0'
__VEROS_INTERFACE__ = dict(name='veros_control',setup_entrypoint=set_inits,
    run_entrypoint=control,settings=SETTINGS,variables=VARIABLES)
