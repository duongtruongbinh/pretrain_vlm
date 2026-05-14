#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/home/shared/miniconda3/envs/nhantd_env/bin/python}"

"$PYTHON_BIN" scripts/download_benchmarks.py "$@"
