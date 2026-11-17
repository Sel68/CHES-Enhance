#!/usr/bin/env bash
set -euo pipefail
cd '/mnt/c/CODES/Projects/OS Project/CHES-Enhance'
exec > >(tee -a outputs/gpu_benchmark/progress.log) 2>&1
"$HOME/.local/share/ches-enhance-benchmark/venv/bin/python" -u outputs/gpu_benchmark/benchmark.py
