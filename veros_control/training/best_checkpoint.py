"""Atomic latest/best snapshots with scores belonging to the saved parameters."""
import math
import os
from pathlib import Path
import pickle
import shutil
import tempfile


def best_path(path):
    path = Path(path)
    return path.with_name(path.stem + '.best' + path.suffix)


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
