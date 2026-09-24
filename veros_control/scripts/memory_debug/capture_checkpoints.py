"""Preserve completed diagnostic checkpoints without modifying training code."""
from pathlib import Path
import re
import sys
import time

root = Path(__file__).resolve().parent
job = sys.argv[1]
log = root / f'slurm-{job}.out'
folder = root / f'job_{job}'
position = 0
pending = ''
update = 0
captured = set()
deadline = time.monotonic() + 2400
while time.monotonic() < deadline:
    if log.exists():
        with log.open() as stream:
            stream.seek(position)
            pending += stream.read()
            position = stream.tell()
        lines = pending.split('\n')
        pending = lines.pop()
        for line in lines:
            match = re.match(r'=== train update (\d+)/(\d+) ===', line)
            if match:
                update, total = map(int, match.groups())
            if line.startswith('  memory after update:') and update not in captured:
                output = folder / f'train_update_{update}.pkl'
                output.write_bytes((folder / 'train.pkl').read_bytes())
                captured.add(update)
                print('Captured', output, flush=True)
                if update == total:
                    raise SystemExit(0)
    time.sleep(3)
raise SystemExit('Timed out waiting for diagnostic checkpoints')
