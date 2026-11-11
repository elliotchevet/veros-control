"""Small, strict configuration; unknown options are errors."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
from veros_control.core.model import ModelConfig
from veros_control.core.correction import CorrectionConfig


@dataclass(frozen=True)
class LossConfig:
    mode: str = 'basin'
    levels: int = 0
    pattern_lambda: float = 1.
    balance: bool = True
    interpolation: str = 'sampled'  # sampled, endpoint, or next
    normalization: str = 'initial'  # initial, no_nn, or none (balance=False also disables)
    depth_loss: dict | None = None
    spatial_filter: dict | None = None


@dataclass(frozen=True)
class SchedulerConfig:
    mode: str = 'constant'  # constant, backtracking, keep_update
    min_rate: float = 1e-5
    start_update: int = 20
    recovery_interval: int = 10
    reject_worsening: bool = False


@dataclass(frozen=True)
class InitialState:
    initial: str
    target: str
    month: int = 1
    reference_hours: float = 720.
    forcing_time_shift: float | None = None
    initial_date: str | None = None


@dataclass(frozen=True)
class CurriculumStage:
    n_steps: int
    updates: int
    initial_rate: float | None = None


@dataclass
class TrainConfig:
    output: str
    setup: str
    initial: str
    target: str
    n_steps: int = 243
    sample_steps: int = 27
    updates: int = 80                 # additional updates when resuming
    learning_rate: float = .0005      # effective rate; no hidden backoff factor
    reuse_rollouts: bool = False
    worker_commands: list[list[str]] = field(default_factory=list)
    worker_timeout: float = 600.
    transition_checkpoint: str = 'latest'
    fixed_loss_balance: bool = False
    stage_initial_rate: float | None = None
    gradient_clip: float | None = None
    chunk_base: int = 3
    seed: int = 0
    normalization: str = 'none'
    reference_hours: float = 720.
    start_month: int = 1
    time_index: int = 0
    forcing_time_shift: float | None = None
    initial_date: str | None = None
    resume: str | None = None
    weights: str | None = None
    training_years: list[int] | None = None
    reference_directory: str | None = None
    forcing_directory: str | None = None
    disable_surface_restoring: bool = False
    correct_evaporation_sign: bool = False
    initial_states: list[InitialState] = field(default_factory=list)
    batch_size: int = 1
    checkpoint_months: list[int] = field(default_factory=list)
    curriculum: list[CurriculumStage] = field(default_factory=list)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    transition: str | None = None  # curriculum only: carry parameters, Adam, statistics and LR
    validate: bool = True
    max_cfl: float = 1.
    candidate_retries: int = 8
    model: ModelConfig = field(default_factory=ModelConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate_options(self):
        if sum(bool(x) for x in (self.resume, self.weights, self.transition)) > 1:
            raise ValueError('Choose resume (full state) or weights (fresh optimizer)')
        if self.n_steps < 1 or self.sample_steps < 1 or self.n_steps % self.sample_steps:
            raise ValueError('n_steps must be a positive multiple of sample_steps')
        if self.transition_checkpoint not in ('latest', 'best'):
            raise ValueError('transition_checkpoint must be latest or best')
        if self.scheduler.recovery_interval < 1 or (self.scheduler.reject_worsening and self.scheduler.mode != 'keep_update'):
            raise ValueError('Invalid rejection or recovery configuration')
        if not math.isfinite(self.worker_timeout) or self.worker_timeout <= 0:
            raise ValueError('worker_timeout must be positive and finite')
        if any(not isinstance(c, list) or any(not isinstance(x, str) for x in c) for c in self.worker_commands):
            raise ValueError('worker_commands must be argument lists, without shell expansion')
        rates = [self.stage_initial_rate] + [s.initial_rate for s in self.curriculum]
        if any(v is not None and (not math.isfinite(v) or not self.scheduler.min_rate <= v <= self.learning_rate) for v in rates):
            raise ValueError('Initial rates must lie between floor and ceiling')
        if self.resume and self.stage_initial_rate is not None:
            raise ValueError('Resume preserves saved LR; cannot reset stage_initial_rate')
        if self.updates < 1 or self.chunk_base < 2:
            raise ValueError('updates must be positive; chunk_base must be >=2')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError('gradient_clip must be positive')
        if self.model.hidden_dim < 1 or self.model.depth < 1:
            raise ValueError("Model width and depth must be positive")
        if self.loss.levels < 0 or not math.isfinite(self.loss.pattern_lambda) or self.loss.pattern_lambda < 0:
            raise ValueError('Invalid depth selection or pattern weight')
        if self.loss.mode not in ('basin','pointwise') or self.loss.interpolation not in ('sampled','endpoint','next'):
            raise ValueError('Invalid loss mode or interpolation')
        if self.normalization not in ('none','meanstd','minmax'):
            raise ValueError('Invalid feature normalization')
        if not math.isfinite(self.reference_hours) or self.reference_hours <= 0 or not 1 <= self.start_month <= 12:
            raise ValueError('Invalid reference interval or start month')
        if self.model.surface_only and self.loss.levels != 1:
            raise ValueError('Surface-only training requires loss.levels=1')
        if self.scheduler.mode not in ('constant', 'backtracking', 'keep_update'):
            raise ValueError('Unknown scheduler mode')
        if (self.scheduler.mode != 'constant' and not 0 <= self.scheduler.min_rate <= self.learning_rate) or self.scheduler.start_update < 1:
            raise ValueError('Invalid scheduler floor or start update')
        if self.scheduler.mode != 'constant' and not self.validate:
            raise ValueError('Adaptive schedulers require candidate validation')
        if self.loss.normalization not in ('initial', 'no_nn', 'none'):
            raise ValueError('Unknown loss normalization')
        if self.training_years is not None:
            if len(self.training_years) != 2 or self.training_years[1] < self.training_years[0]:
                raise ValueError('training_years requires an inclusive first/last year')
            if not self.reference_directory or not self.forcing_directory:
                raise ValueError('Dated training requires reference_directory and forcing_directory')
            if not self.initial_states:
                from veros_control.data.dated import dated_files
                from veros_control.data.reference import snapshot_date
                pairs = dated_files(self.reference_directory, *self.training_years, forcing_directory=self.forcing_directory)
                self.initial_states = [InitialState(a,b,i,initial_date=snapshot_date(a).isoformat()) for i,(a,b) in pairs.items()]
        from veros_control.data.dated import validate_date_options
        validate_date_options(self)
        for case in self.initial_states:
            if case.initial_date:
                from datetime import datetime
                datetime.fromisoformat(case.initial_date)
                if not self.forcing_directory: raise ValueError('Dated initial states require forcing_directory')
        months = [case.month for case in self.initial_states] or [self.start_month]
        if len(set(months)) != len(months) or any(month < 1 for month in months) or any(case.month > 12 and not case.initial_date for case in self.initial_states):
            raise ValueError('Initial-state months must be distinct and in 1..12')
        if not 1 <= self.batch_size <= len(months):
            raise ValueError('Invalid initial-state batch size')
        if self.checkpoint_months and not set(self.checkpoint_months) <= set(months):
            raise ValueError('Checkpoint months must be training months')
        horizons = [stage.n_steps for stage in self.curriculum]
        if horizons != sorted(set(horizons)):
            raise ValueError('Curriculum horizons must be strictly increasing')
        if self.model.residual_blocks < 0:
            raise ValueError('Residual block count must be nonnegative')
        if self.loss.spatial_filter is not None:
            from veros_control.training.spatial_filter import validate_filter
            validate_filter(self.loss.spatial_filter)
        for stage in self.curriculum:
            if stage.n_steps < 1 or stage.n_steps % self.sample_steps or stage.updates < 1:
                raise ValueError('Invalid curriculum stage')
        if self.candidate_retries < 0 or self.max_cfl <= 0:
            raise ValueError('Invalid candidate validation limits')


def load_config(path, cls=TrainConfig):
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    for key in ('output','initial','target','resume','weights','checkpoint','transition','reference_directory','forcing_directory'):
        if values.get(key):
            values[key] = str((path.parent/values[key]).resolve())
    if 'setup' in values:
        module, factory = values['setup'].rsplit(':',1)
        if module.endswith('.py'):
            values['setup'] = str((path.parent/module).resolve())+':'+factory
    for case in values.get('initial_states', []):
        for key in ('initial', 'target'):
            case[key] = str((path.parent/case[key]).resolve())
    if 'initial_states' in values:
        values['initial_states'] = [InitialState(**case) for case in values['initial_states']]
    if 'curriculum' in values:
        values['curriculum'] = [CurriculumStage(**stage) for stage in values['curriculum']]
    for key, schema in [('scheduler',SchedulerConfig),('model',ModelConfig),('correction',CorrectionConfig),('loss',LossConfig)]:
        if key in values:
            values[key] = schema(**values[key])
    config = cls(**values)
    config.validate_options()
    return config


def objective_signature(config):
    keys = ('setup','initial','target','n_steps','sample_steps','normalization','reference_hours','start_month','time_index','forcing_time_shift','model','correction','loss','initial_states','batch_size','checkpoint_months','scheduler','seed')
    values=asdict(config)
    signature = {key:values[key] for key in keys}
    if not signature['model'].get('use_date_embedding'): signature['model'].pop('use_date_embedding', None)
    for case in signature['initial_states']:
        if case.get('initial_date') is None: case.pop('initial_date', None)
    for key in ('initial_date','forcing_directory','reference_directory','training_years','disable_surface_restoring','correct_evaporation_sign'):
        if values[key] not in (None, False): signature[key] = values[key]
    return json.loads(json.dumps(signature))
