#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/shared/miniconda3/envs/nhantd_env/bin/python}"

: "${KTVIC_ANNOTATIONS:?Set KTVIC_ANNOTATIONS to the KTVIC test annotation JSON/JSONL path}"
: "${KTVIC_IMAGE_ROOT:?Set KTVIC_IMAGE_ROOT to the KTVIC image directory}"
: "${STAGE1_CHECKPOINT:?Set STAGE1_CHECKPOINT to a Stage-1 checkpoint path}"
: "${VIET_CULTURAL_VQA_ANNOTATIONS:?Set VIET_CULTURAL_VQA_ANNOTATIONS to the Viet Cultural VQA test JSON path}"
: "${VIET_CULTURAL_VQA_IMAGE_ROOT:?Set VIET_CULTURAL_VQA_IMAGE_ROOT to the Viet Cultural VQA image directory}"
: "${STAGE2_CHECKPOINT:?Set STAGE2_CHECKPOINT to a Stage-2 checkpoint path}"

BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-outputs/benchmarks}"
DEVICE="${DEVICE:-auto}"
KTVIC_MAX_SAMPLES="${KTVIC_MAX_SAMPLES:-}"
VIET_CULTURAL_VQA_MAX_SAMPLES="${VIET_CULTURAL_VQA_MAX_SAMPLES:-}"

ktvic_args=(
  scripts/evaluate_ktvic.py
  --annotations "$KTVIC_ANNOTATIONS"
  --image-root "$KTVIC_IMAGE_ROOT"
  --checkpoint "$STAGE1_CHECKPOINT"
  --output-dir "$BENCH_OUTPUT_ROOT/ktvic"
  --device "$DEVICE"
)

if [[ -n "$KTVIC_MAX_SAMPLES" ]]; then
  ktvic_args+=(--max-samples "$KTVIC_MAX_SAMPLES")
fi

viet_cultural_vqa_args=(
  scripts/evaluate_viet_cultural_vqa.py
  --annotations "$VIET_CULTURAL_VQA_ANNOTATIONS"
  --image-root "$VIET_CULTURAL_VQA_IMAGE_ROOT"
  --checkpoint "$STAGE2_CHECKPOINT"
  --output-dir "$BENCH_OUTPUT_ROOT/viet_cultural_vqa"
  --device "$DEVICE"
)

if [[ -n "$VIET_CULTURAL_VQA_MAX_SAMPLES" ]]; then
  viet_cultural_vqa_args+=(--max-samples "$VIET_CULTURAL_VQA_MAX_SAMPLES")
fi

echo "[benchmark] Stage 1 KTVIC"
"$PYTHON_BIN" "${ktvic_args[@]}"

echo "[benchmark] Stage 2 Viet Cultural VQA"
"$PYTHON_BIN" "${viet_cultural_vqa_args[@]}"

echo "[benchmark] Done"
echo "KTVIC metrics:             $BENCH_OUTPUT_ROOT/ktvic/metrics.json"
echo "Viet Cultural VQA metrics: $BENCH_OUTPUT_ROOT/viet_cultural_vqa/metrics.json"
