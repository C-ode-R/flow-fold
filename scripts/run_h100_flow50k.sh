#!/usr/bin/env bash
# Run from the repository root. Designed for nohup; rerunning resumes data
# collection, embedding, and training from their completed checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v mmseqs >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get update
      DEBIAN_FRONTEND=noninteractive apt-get install -y mmseqs2
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo DEBIAN_FRONTEND=noninteractive apt-get install -y mmseqs2
    fi
  fi
fi
command -v mmseqs >/dev/null 2>&1 || { echo 'mmseqs2 is required' >&2; exit 1; }

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r outputs/requirements.txt
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch/H100 unavailable"; print(torch.cuda.get_device_name(0))'
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python data/collect_afdb_short.py --out work/afdb_short \
  --target 70000 --max-candidates 500000 --workers 8
.venv/bin/python data/finalize_flow_manifest.py \
  --candidates work/afdb_short/candidates.json \
  --holdout-manifest data/manifest_1000train_200holdout.json \
  --out work/flow50k/manifest.json --target 50000 --threads 16
.venv/bin/python data/download_structures.py
.venv/bin/python data/audit_manifest.py \
  --manifest work/flow50k/manifest.json --check-files \
  --report work/flow50k/audit.json
.venv/bin/python data/embed_flow_shards.py \
  --manifest work/flow50k/manifest.json --out work/flow50k/cache \
  --device cuda --batch-size 32 --shard-size 256

cd outputs
resume=()
if [ -f run_flow_50k_batched/last.pt ]; then
  resume=(--resume run_flow_50k_batched/last.pt)
fi
../.venv/bin/python train_flow_large.py \
  --cache ../work/flow50k/cache --out run_flow_50k_batched \
  --epochs 10 --batch-size 8 --accumulate 4 --ode-steps 25 --device cuda "${resume[@]}"
../.venv/bin/python evaluate_flow_large.py \
  --cache ../work/flow50k/cache --checkpoint run_flow_50k_batched/best.pt \
  --out run_flow_50k_batched/test.json --device cuda
