$ErrorActionPreference = 'Stop'
& wsl.exe -d Ubuntu --exec python3 -u '/mnt/c/CODES/Projects/OS Project/CHES-Enhance/outputs/gpu_benchmark/pipeline.py'
exit $LASTEXITCODE
