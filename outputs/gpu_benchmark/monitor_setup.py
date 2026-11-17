"""Show dependency-download activity in the same visible progress log."""
from pathlib import Path
import json
import time

root = Path(__file__).resolve().parent
cache = Path.home() / '.cache/uv'
last = 0
while not (root / 'requirements-resolved.txt').exists():
    if (root / 'status.json').exists() and json.loads((root / 'status.json').read_text()).get('stage') == 'failed':
        break
    size = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
    with (root / 'progress.log').open('a') as log:
        log.write(f'[{time.strftime("%H:%M:%S")}] Setup active: dependency cache {size/1024**2:.1f} MiB; change {max(0,size-last)/1024**2:.1f} MiB since previous check. Training has not started.\n')
    last = size
    time.sleep(30)
