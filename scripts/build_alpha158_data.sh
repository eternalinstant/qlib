#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
START_DATE="${START_DATE:-20160101}"
END_DATE="${END_DATE:-$(date +%Y%m%d)}"
WORKERS="${WORKERS:-4}"
VERIFY_SAMPLE_SIZE="${VERIFY_SAMPLE_SIZE:-32}"
ALPHA158_BATCH_SIZE="${ALPHA158_BATCH_SIZE:-100}"
ALPHA158_REBUILD_ONLY="${ALPHA158_REBUILD_ONLY:-0}"

if [[ -z "${TUSHARE_TOKEN:-}" && -f env.sh ]]; then
  set -a
  # shellcheck disable=SC1091
  source env.sh
  set +a
fi

if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  echo "TUSHARE_TOKEN is required."
  echo "Example: export TUSHARE_TOKEN=your_token_here"
  exit 2
fi

echo "=========================================="
echo "Build Qlib data for official Alpha158"
echo "range:   ${START_DATE} ~ ${END_DATE}"
echo "workers: ${WORKERS}"
echo "batch:   ${ALPHA158_BATCH_SIZE}"
if [[ "${ALPHA158_REBUILD_ONLY}" == "1" ]]; then
  echo "mode:    rebuild provider from existing raw_data"
else
  echo "mode:    update raw_data + rebuild provider"
fi
echo "=========================================="

BUILD_ARGS=(
  build
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --workers "${WORKERS}" \
  --skip-factor-data \
  --batch-size "${ALPHA158_BATCH_SIZE}"
)

if [[ "${ALPHA158_REBUILD_ONLY}" == "1" ]]; then
  BUILD_ARGS+=(--skip-download --skip-raw)
fi

"${PYTHON_BIN}" scripts/build_qlib_data.py "${BUILD_ARGS[@]}"

"${PYTHON_BIN}" scripts/verify_alpha158_data.py \
  --provider-uri data/qlib_data/cn_data \
  --end "${END_DATE}" \
  --sample-size "${VERIFY_SAMPLE_SIZE}"

echo "Alpha158 data build finished."
