"""Finish setup, verify learning on CUDA, then run the visible benchmark."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.chdir(ROOT)

def log(message):
    with (HERE / 'progress.log').open('a') as stream:
        stream.write(f'[{time.strftime("%H:%M:%S")}] {message}\n')

def state(stage, **details):
    temporary = HERE / 'status.tmp.json'
    temporary.write_text(json.dumps(dict(stage=stage, updated=time.strftime('%Y-%m-%d %H:%M:%S %z'), **details), indent=2))
    temporary.replace(HERE / 'status.json')

lock = (HERE / 'pipeline.lock').open('w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('Benchmark pipeline is already running')

if (HERE / 'benchmark_summary.json').exists():
    raise SystemExit('Benchmark already complete; see benchmark_summary.json')

try:
    state('waiting_for_dependency_installation')
    log('Automatic pipeline active: waiting for setup, then GPU learning checks, training, and evaluation.')
    while True:
        installers = []
        for process in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args = process.read_bytes().split(b'\0')
                if args and b'ches-enhance-benchmark/tools/' in args[0] and b'pip' in args and b'install' in args:
                    installers.append(process)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
        if not installers:
            break
        time.sleep(10)
    if not (HERE / 'requirements-resolved.txt').exists():
        state('installing_dependencies')
        log('Continuing dependency installation with timeouts adjusted for the slow connection.')
        subprocess.run(['bash', str(HERE / 'install_dependencies.sh')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    python = str(Path.home() / '.local/share/ches-enhance-benchmark/venv/bin/python')
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false', MPLCONFIGDIR=str(HERE / 'matplotlib'))
    state('checking_gpu_learning')
    log('Checking masked SAC gradients, checkpoint loading, and actor/critic learning on the GPU.')
    with (HERE / 'progress.log').open('a') as stream:
        subprocess.run([python, '-u', '-m', 'pytest', '-q', 'test_core.py::test_history_sampling_and_sac_masked_gradients', 'test_core.py::test_actor_and_forced_rejection_critic_learn'], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    state('running_benchmark', episodes=10, policy='LRU', disk='tight', seed=0)
    subprocess.run(['bash', str(HERE / 'run.sh')], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    state('complete', summary=str(HERE / 'benchmark_summary.json'))
    log('Pipeline finished successfully. Results are in benchmark_summary.json.')
except BaseException as error:
    state('failed', error=str(error))
    log('Pipeline stopped with an error; training will never silently fall back to CPU.')
    with (HERE / 'progress.log').open('a') as stream:
        traceback.print_exc(file=stream)
    raise
