from dataclasses import dataclass, field
from veros_control.core.model import ModelConfig
from veros_control.core.correction import CorrectionConfig
from veros_control.training.config import LossConfig


@dataclass
class EvaluationConfig:
    output: str
    setup: str
    initial: str
    target: str
    checkpoint: str
    n_steps: int = 720
    snapshot_every: int = 24
    baseline: bool = True
    reference_hours: float = 720.
    start_month: int = 1
    time_index: int = 0
    max_cfl: float = 1.
    initial_date: str | None = None
    forcing_directory: str | None = None
    disable_surface_restoring: bool = False
    correct_evaporation_sign: bool = False
    normalization: str = 'none'  # restored from checkpoint
    model: ModelConfig = field(default_factory=ModelConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    def validate_options(self):
        from veros_control.data.dated import validate_date_options
        validate_date_options(self)
        if self.n_steps < 1 or self.snapshot_every < 1 or self.max_cfl <= 0:
            raise ValueError('Invalid evaluation duration, snapshot interval, or CFL limit')
        if self.reference_hours <= 0 or not 1 <= self.start_month <= 12:
            raise ValueError('Invalid reference interval or start month')
