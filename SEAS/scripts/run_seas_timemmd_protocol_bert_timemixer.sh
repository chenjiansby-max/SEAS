#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-7}"
SEED="${2:-2024}"
DOMAINS="${3:-Agriculture,Climate,Economy,Energy,Environment,Health,Security,SocialGood,Traffic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DATA_ROOT="${SEAS_TIMEMMD_DATA_ROOT:-${REPO_DIR}/data/timemmd_mmts_tats}"
OUT_DIR="${SEAS_TIMEMMD_OUT_DIR:-${REPO_DIR}/outputs/timemmd_protocol/seas_bert_timemixer_seed${SEED}}"
STDOUT_DIR="${SEAS_TIMEMMD_STDOUT_DIR:-${OUT_DIR}/stdout}"

python scripts/prepare_seas_timemmd_from_mmts.py \
  --output-root "${DATA_ROOT}" \
  --text-col "${SEAS_TIMEMMD_TEXT_COL:-Final_Search_4}"

mkdir -p "${OUT_DIR}" "${STDOUT_DIR}"

SEAS_DATA_ROOT="${DATA_ROOT}" \
SEAS_FREQLOOKBACK_OUT_DIR="${OUT_DIR}" \
SEAS_STDOUT_DIR="${STDOUT_DIR}" \
bash scripts/run_seas_freqlookback_bert_timemixer.sh "${GPUS}" "${SEED}" "${DOMAINS}"
