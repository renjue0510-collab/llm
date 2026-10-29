#!/usr/bin/env bash
# Read-only check of the GPU node before the first run. Prints versions and GPU state.
set -uo pipefail
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
python3 - <<'PY'
import importlib
for m in ["torch", "vllm", "verl", "transformers", "ray", "flash_attn", "math_verify", "datasets", "tensordict"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:14s} MISSING ({type(e).__name__})")
import torch
print("cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
PY
df -h /nfs 2>/dev/null | tail -1
