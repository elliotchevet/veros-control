try:
    import veros  
except ImportError:
    raise RuntimeError(
        'Plugin needs Veros to be installed (try `pip install veros`)'
    )

__version__ ="0.1.0" 

from veros_control.control import control
from veros_control.set_inits import set_inits
from veros_control.variables import VARIABLES
from veros_control.settings import SETTINGS


__VEROS_INTERFACE__ = dict(
    name = 'veros_control',
    setup_entrypoint = set_inits,
    run_entrypoint = control,
    settings = SETTINGS,
    variables = VARIABLES,
)
