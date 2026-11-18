#!/usr/bin/env bash
set -euo pipefail
cd '/mnt/c/CODES/Projects/OS Project/CHES-Enhance'
exec > >(tee -a outputs/gpu_benchmark/progress.log) 2>&1
base="$HOME/.local/share/ches-enhance-benchmark"
export UV_HTTP_TIMEOUT=14400
export UV_CONCURRENT_DOWNLOADS=3
echo "[$(date -Is)] Installing dependencies with 3 parallel downloads and a 4-hour timeout for the slow connection"
"$base/tools/uv-x86_64-unknown-linux-gnu/uv" pip install --python "$base/venv/bin/python" 'jax[cuda13]' 'optax==0.2.8' 'numpy>=1.26,<3' 'matplotlib>=3.9,<4' 'pillow>=10' 'pytest>=6.2.5'
"$base/tools/uv-x86_64-unknown-linux-gnu/uv" pip freeze --python "$base/venv/bin/python" > outputs/gpu_benchmark/requirements-resolved.txt
echo "[$(date -Is)] Setup complete"
