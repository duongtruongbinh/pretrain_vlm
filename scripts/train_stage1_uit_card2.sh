#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

/home/shared/miniconda3/envs/nhantd_env/bin/accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION:-no}" \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  --dynamo_backend no \
  train.py --config-section "${CONFIG_SECTION:-train_uit_only}" "$@"
