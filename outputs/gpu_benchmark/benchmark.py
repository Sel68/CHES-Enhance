"""Visible, timestamped benchmark wrapper; project training behavior is unchanged."""
import csv
import json
import os
from pathlib import Path
import sys
import time
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs/gpu_benchmark/matplotlib')

def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)

def heartbeat():
    while True:
        time.sleep(20)
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                log(f'GPU heartbeat: utilization %, VRAM used/total MiB, power W = {result.stdout.strip()}')
                with open(ROOT / 'outputs/gpu_benchmark/gpu_samples.csv', 'a') as sample:
                    sample.write(f'{time.time()},{result.stdout.strip()}\n')
        except (OSError, subprocess.TimeoutExpired):
            log('Benchmark process is active')

threading.Thread(target=heartbeat, daemon=True).start()

import jax
import jax.numpy as jnp
import optax
log(f'Python {sys.version.split()[0]}; JAX {jax.__version__}; Optax {optax.__version__}; devices: {jax.devices()}')
assert jax.default_backend() == 'gpu', 'GPU backend required'
started = time.perf_counter()
jax.jit(lambda x: x @ x)(jnp.ones((512, 512))).block_until_ready()
log(f'Compiled GPU check passed in {time.perf_counter() - started:.2f}s')

import run
from agent import SAC
from data import Config, make_dataset
from env import Env

original_update = SAC.update
def visible_update(self, batch, imitation=False):
    if not hasattr(self, '_benchmark_update_started'):
        self._benchmark_update_started = time.perf_counter()
        log('Compiling first learning update; this first call can take longer')
    result = original_update(self, batch, imitation=imitation)
    if imitation:
        self._benchmark_bc = getattr(self, '_benchmark_bc', 0) + 1
        if self._benchmark_bc % 20 == 0:
            log(f'Behavior-cloning warm start: {self._benchmark_bc}/100 updates')
    elif self.updates % 100 == 0:
        log(f'SAC updates: {self.updates}; critic loss={result["critic_loss"]:.4f}')
    return result
SAC.update = visible_update

original_step = Env.step
def visible_step(self, action):
    if not hasattr(self, '_benchmark_started'):
        self._benchmark_started = time.perf_counter()
        log(f'Rollout {self.trace["id"]}: {len(self.trace["tasks"])} tasks')
    done = original_step(self, action)
    if self.cursor % 500 == 0 or done:
        log(f'  {self.trace["id"]}: {self.cursor}/{len(self.trace["tasks"])} tasks; rollout elapsed {time.perf_counter() - self._benchmark_started:.1f}s')
    return done
Env.step = visible_step

timings = []
original_append = run.append_csv
def visible_append(path, row):
    original_append(path, row)
    if Path(path).name == 'training.csv':
        timings.append(dict(episode=row['episode'], elapsed_seconds=time.perf_counter() - train_started))
        log(f'EPISODE {row["episode"]}/10 DONE: latency={row["avg_latency"]:.2f}s, on-time={row["completion_ratio"]:.1%}, hit={row["hit_rate"]:.1%}, updates={row["updates"]}; training elapsed={timings[-1]["elapsed_seconds"]:.1f}s')
        (ROOT / 'outputs/gpu_benchmark/episode_timings.json').write_text(json.dumps(timings, indent=2))
run.append_csv = visible_append

data = 'outputs/gpu_benchmark/data_full'
output = 'outputs/gpu_benchmark/run'
if not (ROOT / data / 'manifest.json').exists():
    log('Generating 64 train / 4 validation / 8 test traces, 400 slots each')
    make_dataset(data, Config(slots=400), (64, 4, 8))

log('Starting LRU/tight/seed 0, 10 episodes; default width=64, history=8, batch=128')
train_started = time.perf_counter()
sys.argv = ['run.py', 'train', '--data', data, '--output', output, '--device', 'gpu', '--policies', 'LRU', '--disks', 'tight', '--seeds', '0', '--episodes', '10', '--eval-every', '5']
run.main()
training_seconds = time.perf_counter() - train_started
log(f'TRAINING FINISHED: {training_seconds:.1f}s ({training_seconds/60:.2f} min), including warm start and validation')

log('Evaluating best checkpoint and five heuristics on 8 test traces')
eval_started = time.perf_counter()
sys.argv = ['run.py', 'evaluate', '--data', data, '--output', output, '--device', 'gpu', '--policies', 'LRU', '--disks', 'tight', '--seeds', '0']
run.main()
with open(ROOT / output / 'results.csv', newline='') as f:
    rows = list(csv.DictReader(f))
summary = {}
for method in dict.fromkeys(r['method'] for r in rows):
    selected = [r for r in rows if r['method'] == method]
    summary[method] = {k: sum(float(r[k]) for r in selected) / len(selected) for k in ['avg_latency', 'completion_ratio', 'hit_rate', 'rejection_rate', 'evictions']}
    log(f'{method}: {summary[method]}')
report = dict(training_seconds=training_seconds, evaluation_seconds=time.perf_counter()-eval_started, versions=dict(python=sys.version, jax=jax.__version__, optax=optax.__version__), devices=[str(d) for d in jax.devices()], episode_timings=timings, results=summary)
(ROOT / 'outputs/gpu_benchmark/benchmark_summary.json').write_text(json.dumps(report, indent=2))
log('BENCHMARK COMPLETE; report: outputs/gpu_benchmark/benchmark_summary.json')
