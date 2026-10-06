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
    resume: str | None = None
    weights: str | None = None
    validate: bool = True
    max_cfl: float = 1.
    candidate_retries: int = 8
    model: ModelConfig = field(default_factory=ModelConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate_options(self):
        if self.resume and self.weights:
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
        if self.candidate_retries < 0 or self.max_cfl <= 0:
            raise ValueError('Invalid candidate validation limits')


def load_config(path, cls=TrainConfig):
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    for key in ('output','initial','target','resume','weights','checkpoint'):
        if values.get(key):
            values[key] = str((path.parent/values[key]).resolve())
    if 'setup' in values:
        module, factory = values['setup'].rsplit(':',1)
        if module.endswith('.py'):
            values['setup'] = str((path.parent/module).resolve())+':'+factory
    for key, schema in [('model',ModelConfig),('correction',CorrectionConfig),('loss',LossConfig)]:
        if key in values:
            values[key] = schema(**values[key])
    config = cls(**values)
    config.validate_options()
    return config


def objective_signature(config):
    keys = ('setup','initial','target','n_steps','sample_steps','normalization','reference_hours','start_month','time_index','model','correction','loss')
    values=asdict(config)
    return json.loads(json.dumps({key:values[key] for key in keys}))
