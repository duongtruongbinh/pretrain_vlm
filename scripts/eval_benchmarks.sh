#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/shared/miniconda3/envs/nhantd_env/bin/python}"
BENCH_ENV_FILE="${BENCH_ENV_FILE:-data/benchmarks/benchmark_env.sh}"
STAGE1_OUTPUT_DIR="${STAGE1_OUTPUT_DIR:-outputs/stage_1_projector_coco_uit_test}"
STAGE2_OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-outputs/stage_2_instruction_final}"

checkpoint_from_pointer() {
  local output_dir="$1"
  local pointer_name="$2"
  "$PYTHON_BIN" - "$output_dir" "$pointer_name" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
pointer_name = sys.argv[2]
pointer_path = output_dir / f"{pointer_name}_checkpoint.json"
if not pointer_path.exists():
    raise SystemExit(0)
payload = json.loads(pointer_path.read_text(encoding="utf-8"))
checkpoint = Path(str(payload.get("checkpoint", "")))
if not checkpoint.is_absolute():
    checkpoint = output_dir / checkpoint
print(checkpoint)
PY
}

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[eval] Missing $name."
    echo "[eval] Run this first: bash scripts/download_benchmarks.sh"
    echo "[eval] Then run: source $BENCH_ENV_FILE"
    exit 1
  fi
}

if [[ -f "$BENCH_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$BENCH_ENV_FILE"
else
  echo "[eval] Benchmark env file not found: $BENCH_ENV_FILE"
  echo "[eval] Run this first: bash scripts/download_benchmarks.sh"
  exit 1
fi

export STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-$(checkpoint_from_pointer "$STAGE1_OUTPUT_DIR" best)}"
export STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-$(checkpoint_from_pointer "$STAGE2_OUTPUT_DIR" best)}"
export DEVICE="${DEVICE:-auto}"
export BENCH_OUTPUT_ROOT="${BENCH_OUTPUT_ROOT:-outputs/benchmarks}"

require_var KTVIC_ANNOTATIONS
require_var KTVIC_IMAGE_ROOT
require_var VIET_CULTURAL_VQA_ANNOTATIONS
require_var VIET_CULTURAL_VQA_IMAGE_ROOT
require_var STAGE1_CHECKPOINT
require_var STAGE2_CHECKPOINT

echo "[eval] KTVIC annotations:        $KTVIC_ANNOTATIONS"
echo "[eval] KTVIC images:             $KTVIC_IMAGE_ROOT"
echo "[eval] Viet Cultural annotations: $VIET_CULTURAL_VQA_ANNOTATIONS"
echo "[eval] Viet Cultural images:      $VIET_CULTURAL_VQA_IMAGE_ROOT"
echo "[eval] Stage 1 checkpoint:        $STAGE1_CHECKPOINT"
echo "[eval] Stage 2 checkpoint:        $STAGE2_CHECKPOINT"
echo "[eval] Device:                    $DEVICE"
echo "[eval] Output root:               $BENCH_OUTPUT_ROOT"

bash scripts/run_benchmarks.sh
