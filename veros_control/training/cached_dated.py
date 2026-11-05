"""Persistent date-independent compiled gradient, score and stability rollouts."""
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path
import numpy as np

class CachedDatedRollouts:
    def __init__(self, adapter):
        cfg = adapter.cfg
        if not (getattr(cfg, 'surface_only', False)
                and cfg.monthly_initial_states.get('training_years')
                and cfg.basin_levels == 1 and cfg.basin_pattern_lambda == 1.
                and tuple(cfg.keys_loss) == ('temp', 'salt')):
            raise ValueError('Compiled dated rollouts require dated surface T/S training, one basin level and pattern lambda=1')
        self.a=adapter;self.m=adapter.m;self.sim=adapter.simulation;self.cfg=cfg
        self.template=adapter.base_state.copy();self.functions={};self.terms=None;self.traces={}

    @contextmanager
    def inputs(self, sample):
        from veros_control.training.train import load_glorys_state,load_glorys_snapshot
        from veros_control.training.monthly_batches import snapshot_date,MONTH_DAYS
        from veros_control.training.dated_months import model_timestamp,load_window,MODEL_YEAR
        from veros_control.training.basin_loss import make_basin_loss
        sim=self.sim;old=sim.state;jnp=self.m.jnp
        first,last=self.a.files[sample];start,end=snapshot_date(first),snapshot_date(last)
        try:
            sim.state=self.template
            state=load_glorys_state(sim,first,config=SimpleNamespace(glorys_month_lengths=MONTH_DAYS,glorys_start_month=start.month))
            with state.variables.unlock():
                state.variables.forcing_time_shift=jnp.asarray(model_timestamp(start)-(start.year-1993)*MODEL_YEAR)
                state.variables.date_year_offset=jnp.asarray(float(start.year-1993))
            sim.state=state
            initial=load_glorys_snapshot(sim,first,('temp','salt'));target=load_glorys_snapshot(sim,last,('temp','salt'))
            loaded,index=load_window(Path(first).parent,start,self.a.steps,state.settings.dt_tracer)
            length=int(np.ceil(self.a.steps*state.settings.dt_tracer/10800))+3
            buffers=tuple(jnp.pad(jnp.asarray(z),((0,0),(0,0),(0,length-z.shape[-1])),mode='edge') for z in loaded)
            if self.terms is None:
                _,_,self.terms,_=make_basin_loss(state.variables,target,('temp','salt'),1,pattern_lambda=1.,initial_target=initial,target_alpha=lambda s:s.variables.time/(30*86400.),depth_loss=self.cfg.depth_loss,spatial_filter=self.cfg.spatial_loss_filter,lead_hours=lambda s:s.variables.time/3600.)
            variables={key:getattr(state.variables,key) for key in self.m.get_all_active_keys(state)}
            args=(variables,buffers,jnp.asarray(index,dtype=jnp.int64),initial,target,jnp.asarray(model_timestamp(end)-model_timestamp(start)),self.a.loss_term_denominators)
            yield args
        finally:
            sim.state=old

    def function(self, mode, options=None):
        if mode in self.functions:return self.functions[mode]
        m,sim,cfg=self.m,self.sim,self.cfg;jnp=m.jnp
        from veros_control.setup.global_1deg_control.era5_sequence import sequence_context
        from veros_control.training.stability import make_rollout_validator
        options=dict(options or {})
        def body(params,variables,buffers,index,initial,target,interval,den):
            self.traces[mode]=self.traces.get(mode,0)+1
            print(f'COMPILE mode={mode} trace={self.traces[mode]} horizon={self.a.steps}',flush=True)
            state=m.inject_dynamic_vars(self.template,variables)
            def terms(s):return self.terms(s,target,initial,interval)
            with sequence_context(sim,buffers,index):
                if mode=='validate':
                    old=sim.state
                    try:
                        sim.state=state
                        raw=make_rollout_validator(sim,self.a.single.model_batched_unscaled,self.a.single.stats,cfg.epsilon,
                            m.make_chunk_sizes(self.a.steps,base=cfg.chunk_base),m.nn_epsilon_step,m._temporary_solver_cache,
                            max_cfl=self.a.max_cfl,score_terms=terms if cfg.monthly_fused_validation else None,
                            sample_steps=cfg.sample_steps,score_denominators=den,raw_rollout=True)
                        return raw(params)
                    finally:sim.state=old
                ad=m.NNParamsTrainChunkedVJP(sim,self.a.single.model_batched_unscaled,self.a.single.stats,lambda s:jnp.mean(terms(s)),
                    freeze_vars=cfg.freeze_vars,loss_reduction=cfg.loss_reduction,loss_terms_function=terms,loss_term_count=2)
                ad.loss_term_denominators=den
                return ad.g(state,params,**options)
        self.functions[mode]=m.jax.jit(body)
        return self.functions[mode]

    def g(self,sample,params,options):
        mode='forward' if options.get('forward_only') else 'gradient'
        with self.inputs(sample) as args, self.m._temporary_solver_cache():
            result=self.function(mode,options)(params,*args)
            self.m.jax.block_until_ready(result)
            return result

    def validate(self,sample,params):
        with self.inputs(sample) as args, self.m._temporary_solver_cache():
            result,total=self.m.jax.device_get(self.function('validate')(params,*args))
        report=dict(nonfinite_count=int(result[0]),min_salinity=float(result[1]),max_cfl=float(result[2]))
        valid=bool(np.all(np.isfinite(result)) and result[0]==0 and result[1]>0 and result[2]<=self.a.max_cfl)
        if self.cfg.monthly_fused_validation:
            terms=np.asarray(total)/(self.a.steps//self.cfg.sample_steps)
            loss=float(np.mean(terms/np.asarray(self.a.loss_term_denominators)))
            report.update(loss=loss,loss_terms=terms.tolist())
            valid=valid and np.isfinite(loss) and np.isfinite(terms).all()
        return bool(valid),report
