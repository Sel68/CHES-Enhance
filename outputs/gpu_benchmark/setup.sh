#!/usr/bin/env bash
set -euo pipefail
cd '/mnt/c/CODES/Projects/OS Project/CHES-Enhance'
exec > >(tee -a outputs/gpu_benchmark/progress.log) 2>&1
base="$HOME/.local/share/ches-enhance-benchmark"
mkdir -p "$base/tools"
echo "[$(date -Is)] Downloading isolated environment manager"
curl -fL --retry 3 https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz -o "$base/tools/uv.tar.gz"
tar -xzf "$base/tools/uv.tar.gz" -C "$base/tools"
uv="$base/tools/uv-x86_64-unknown-linux-gnu/uv"
"$uv" venv --python 3.12 "$base/venv"
echo "[$(date -Is)] Installing JAX CUDA 13 and benchmark dependencies"
"$uv" pip install --python "$base/venv/bin/python" 'jax[cuda13]' 'optax==0.2.8' 'numpy>=1.26,<3' 'matplotlib>=3.9,<4' 'pillow>=10' 'pytest>=6.2.5'
"$uv" pip freeze --python "$base/venv/bin/python" > outputs/gpu_benchmark/requirements-resolved.txt
echo "[$(date -Is)] Setup complete"
