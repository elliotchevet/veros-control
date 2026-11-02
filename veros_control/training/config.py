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


@dataclass(frozen=True)
class InitialState:
    initial: str
    target: str
    month: int = 1
    reference_hours: float = 720.
    forcing_time_shift: float | None = None


@dataclass(frozen=True)
class CurriculumStage:
    n_steps: int
    updates: int


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
    gradient_clip: float | None = None
    chunk_base: int = 3
    seed: int = 0
    normalization: str = 'none'
    reference_hours: float = 720.
    start_month: int = 1
    time_index: int = 0
    forcing_time_shift: float | None = None
    resume: str | None = None
    weights: str | None = None
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
        months = [case.month for case in self.initial_states] or [self.start_month]
        if len(set(months)) != len(months) or any(not 1 <= month <= 12 for month in months):
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
    for key in ('output','initial','target','resume','weights','checkpoint','transition'):
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
    return json.loads(json.dumps({key:values[key] for key in keys}))
