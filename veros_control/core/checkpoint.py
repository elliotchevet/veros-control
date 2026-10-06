"""Atomic latest/best snapshots with scores belonging to the saved parameters."""
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile


def best_path(path):
    path = Path(path)
    return path.with_name('best.pkl') if path.name == 'latest.pkl' else path.with_name(path.stem + '.best' + path.suffix)


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class BestLatestCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.best_path = best_path(path)
        self.best_payload = None
        self.best_loss = math.inf
        if self.best_path.exists():
            with self.best_path.open('rb') as stream:
                self.best_payload = pickle.load(stream)
            self.best_loss = float(self.best_payload['checkpoint_loss'])
            if not math.isfinite(self.best_loss):
                raise ValueError('Existing best checkpoint has a nonfinite score')

    def save(self, payload, loss, loss_terms):
        loss = float(loss)
        if not math.isfinite(loss):
            raise FloatingPointError('Cannot save a nonfinite scored checkpoint')
        snapshot = dict(payload)
        snapshot['train_history'] = list(payload.get('train_history', []))
        snapshot.update(checkpoint_loss=loss, checkpoint_loss_terms=dict(loss_terms),
                        checkpoint_loss_timing='evaluated at saved parameters')
        improved = loss < self.best_loss
        snapshot.update(best_loss=loss if improved else self.best_loss,
                        best_update=snapshot['update'] if improved else self.best_payload['update'])
        atomic_save(self.path, snapshot)
        if improved:
            # Atomic replacement of latest leaves an older linked best untouched.
            temporary = self.best_path.with_name('.' + self.best_path.name + '.link')
            try:
                if temporary.exists():
                    temporary.unlink()
                try:
                    os.link(self.path, temporary)
                except OSError:
                    shutil.copyfile(self.path, temporary)
                os.replace(temporary, self.best_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            self.best_payload = snapshot
            self.best_loss = loss
        return improved


def load_checkpoint(path):
    """Load a trusted local pickle checkpoint."""
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def model_from_checkpoint(checkpoint, nz):
    """Load the public format or a recent single-MLP research checkpoint."""
    from dataclasses import asdict
    from veros_control.core.model import ModelConfig, VerticalCorrectionMLP
    from veros_control.core.correction import CorrectionConfig
    if checkpoint.get('format_version') == 1:
        model_config = ModelConfig(**checkpoint['config']['model'])
        correction = CorrectionConfig(**checkpoint['config']['correction'])
        normalization = checkpoint['config']['normalization']
        stats = checkpoint.get('stats_features')
    else:
        if checkpoint.get('num_mlps',1) != 1 or any(checkpoint.get(k,False) for k in ('TS','use_one_hot','conservative_mixing','train_epsilon')):
            raise ValueError('Checkpoint uses an experimental architecture that is not supported')
        model_config = ModelConfig(hidden_dim=checkpoint.get('hidden_dim_nn',64),depth=checkpoint.get('depth_nn',2),
            use_lon_lat=checkpoint.get('use_lon_lat',False),use_forcings=checkpoint.get('use_forcings',False),
            zero_output_init=checkpoint.get('zero_output_init',False))
        correction = CorrectionConfig(tracer_epsilon=checkpoint.get('tracer_epsilon',checkpoint.get('epsilon',1e-6)),
            momentum_epsilon=checkpoint.get('momentum_epsilon',checkpoint.get('epsilon',1e-6)),
            bounds_per_hour=checkpoint.get('correction_bounds_per_hour'),depth_taper=checkpoint.get('correction_depth_taper'))
        stats = checkpoint.get('stats_features')
        normalization = checkpoint.get('scaling_method_features','meanstd') if stats is not None and checkpoint.get('normalize_inputs',True) else 'none'
    model = VerticalCorrectionMLP(nz=nz,**asdict(model_config))
    return model,model_config,correction,stats,normalization
