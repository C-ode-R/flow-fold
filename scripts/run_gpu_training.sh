#!/usr/bin/env bash
# Single CUDA GPU; never evaluates or prepares test by default.
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET_DIR="${DATASET_DIR:-data/dataset_v1_final}"
CACHE_DIR="${CACHE_DIR:-data/cache150}"
RUN_DIR="${RUN_DIR:-outputs/fold150_full_v1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PRECISION="${PRECISION:-fp32}"
CUDA_EXTRA="${CUDA_EXTRA:-cu126}"
READY_DIR="${READY_DIR:-data/gpu_ready}"
if [[ -f "$READY_DIR/READY.json" ]]; then
  python3 scripts/verify_gpu_ready.py --cache "$CACHE_DIR" --ready "$READY_DIR"
fi
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi not found; use a CUDA GPU host.' >&2; exit 1; }
nvidia-smi
uv sync --locked --python 3.14 --extra "$CUDA_EXTRA"
uv run --locked --extra "$CUDA_EXTRA" python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
if [[ ! -f "$CACHE_DIR/train/index.json" || ! -f "$CACHE_DIR/validation/index.json" ]]; then
  uv run --locked --extra "$CUDA_EXTRA" python -u scripts/prepare_dataset.py \
    --dataset "$DATASET_DIR" --out "$CACHE_DIR" --device cuda --batch-size "$BATCH_SIZE"
fi
uv run --locked --extra "$CUDA_EXTRA" python -u scripts/preflight_gpu.py \
  --cache "$CACHE_DIR" --device cuda --precision "$PRECISION" --batch-size "$BATCH_SIZE" \
  --out "${RUN_DIR}_preflight.json"
resume=()
if [[ -f "$RUN_DIR/last.pt" ]]; then resume=(--resume); fi
uv run --locked --extra "$CUDA_EXTRA" python -u scripts/train_dataset.py \
  --cache "$CACHE_DIR" --out "$RUN_DIR" --device cuda --precision "$PRECISION" \
  --epochs 100 --batch-size "$BATCH_SIZE" --workers 2 "${resume[@]}"
