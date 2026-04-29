#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-nhantd_env}"
STAGE="${1:-stage1}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

echo "Project: $ROOT_DIR"
echo "Conda env: $CONDA_ENV"
echo "Visible GPU: $CUDA_VISIBLE_DEVICES"
echo "Stage: $STAGE"

case "$STAGE" in
  stage1|caption)
    conda run --no-capture-output -n "$CONDA_ENV" python train.py
    ;;
  stage2|instruction)
    conda run --no-capture-output -n "$CONDA_ENV" python train_instruction.py
    ;;
  *)
    echo "Usage: $0 [stage1|stage2]"
    echo "  stage1: train projector caption model with train.py"
    echo "  stage2: instruction finetune with train_instruction.py"
    exit 2
    ;;
esac
