#!/usr/bin/env bash
# Run from the repository root after copying the CPU-prepared dataset.
# Designed for nohup; embedding and training resume from completed checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f work/flow50k/manifest.json ] || {
  echo 'Missing CPU-prepared work/flow50k/manifest.json; do not rent GPU for collection' >&2
  exit 1
}
python3 data/audit_manifest.py \
  --manifest work/flow50k/manifest.json --check-files \
  --report work/flow50k/audit.json
python3 data/flow50k_filelist.py \
  --manifest work/flow50k/manifest.json \
  --out work/flow50k/received_files.txt

python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r outputs/requirements.txt
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch/H100 unavailable"; print(torch.cuda.get_device_name(0))'
.venv/bin/python -m unittest discover -s tests -v

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
