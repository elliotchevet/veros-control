"""Asset-free small Veros setup for integration checks."""
import veros_control
from veros import veros_routine
from veros.setups.acc_basic import ACCBasicSetup
from veros.core.operators import numpy as npx, update, at
from veros_control.control import apply_correction
from veros.variables import Variable

class SmokeSetup(ACCBasicSetup):
    __veros_plugins__ = (veros_control,)

    @veros_routine
    def set_parameter(self,state):
        super().set_parameter(state)
        s=state.settings
        s.nx,s.ny,s.nz=8,8,4
        s.dt_mom=s.dt_tracer=3600.
        s.enable_streamfunction=False
        s.enable_tempsalt_sources=s.enable_momentum_sources=True
        s.enable_neutral_diffusion=s.enable_skew_diffusion=False
        s.enable_hor_friction=s.enable_tke=s.enable_bottom_friction=False
        s.restart_output_filename=''
        state.var_meta['forcing_time_shift']=Variable('Forcing offset',tuple(),'s','Forcing clock offset')

    @veros_routine
    def set_grid(self,state):
        v=state.variables
        v.dxt=update(v.dxt,at[:],45.)
        v.dyt=update(v.dyt,at[:],10.)
        v.dzt=update(v.dzt,at[:],npx.array([400.,300.,200.,100.]))

    @veros_routine
    def set_forcing(self,state):
        v=state.variables
        v.temp_source=update(v.temp_source,at[:],1e-7*v.maskT)
        v.salt_source=update(v.salt_source,at[:],0.)
        v.u_source=update(v.u_source,at[:],0.)
        v.v_source=update(v.v_source,at[:],0.)
        apply_correction(self,state)

    @veros_routine
    def set_diagnostics(self,state):
        for diagnostic in state.diagnostics.values():
            diagnostic.output_frequency=diagnostic.sampling_frequency=0

def create_setup():
    simulation=SmokeSetup();simulation.setup();return simulation


class DynamicForcingSetup(SmokeSetup):
    def training_forcing(self):
        return (npx.asarray(1e-7)+self.state.variables.forcing_time_shift*1e-11,)

    @veros_routine
    def set_forcing(self,state):
        forcing=getattr(self,'_training_forcing',None)
        if forcing is None:forcing=self.training_forcing()
        v=state.variables
        v.temp_source=update(v.temp_source,at[:],forcing[0]*v.maskT)
        v.salt_source=update(v.salt_source,at[:],0.)
        v.u_source=update(v.u_source,at[:],0.)
        v.v_source=update(v.v_source,at[:],0.)
        apply_correction(self,state)


def create_dynamic_setup():
    simulation=DynamicForcingSetup();simulation.setup();return simulation
