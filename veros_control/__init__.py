import os
import sys


def _truthy_env(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


def _diskless_mode_requested():
    return (
        "--diskless-mode" in sys.argv
        or _truthy_env("VEROS_CONTROL_DISKLESS_MODE")
        or _truthy_env("VEROS_DISKLESS_MODE")
    )


try:
    import veros
except ImportError:
    raise RuntimeError(
        'Plugin needs Veros to be installed (try `pip install veros`)'
    )


if _diskless_mode_requested():
    from veros import runtime_settings

    try:
        setattr(runtime_settings, "diskless_mode", True)
    except RuntimeError as exc:
        if getattr(runtime_settings, "diskless_mode") is not True:
            raise exc

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

