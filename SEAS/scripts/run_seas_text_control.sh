#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-4,6}"
SEED="${2:-2024}"
DOMAINS="${3:-Security,Climate,SocialGood,Traffic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

CONDA_ENV="${SEAS_CONDA_ENV:-tsfm_env}"
activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    return 0
  fi
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  fi
}
activate_conda_env

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_ROOT="${SEAS_TIMEMMD_DATA_ROOT:-${REPO_DIR}/data/timemmd_mmts_tats}"
BASE_OUT="${SEAS_TEXT_CONTROL_BASE_OUT:-${REPO_DIR}/outputs/timemmd_protocol/seas_text_control_seed${SEED}}"

python scripts/prepare_seas_timemmd_from_mmts.py \
  --output-root "${DATA_ROOT}" \
  --text-col "${SEAS_TIMEMMD_TEXT_COL:-Final_Search_4}"

run_variant() {
  local tag="$1"
  local mode="$2"
  local shift_steps="${3:-1}"
  local out_dir="${BASE_OUT}/${tag}"
  local stdout_dir="${out_dir}/stdout"

  mkdir -p "${out_dir}" "${stdout_dir}"
  echo "=== SEAS text control: ${tag} | mode=${mode} | GPUs=${GPUS} | domains=${DOMAINS} ==="

  SEAS_DATA_ROOT="${DATA_ROOT}" \
  SEAS_FREQLOOKBACK_OUT_DIR="${out_dir}" \
  SEAS_STDOUT_DIR="${stdout_dir}" \
  SEAS_ABLATION_TAG="text_${tag}" \
  SEAS_TEXT_CONTROL_MODE="${mode}" \
  SEAS_TEXT_SHIFT_STEPS="${shift_steps}" \
  SEAS_TEXT_CONTROL_SEED="${SEED}" \
  bash scripts/run_seas_freqlookback_bert_timemixer.sh "${GPUS}" "${SEED}" "${DOMAINS}"
}

run_variant "aligned" "aligned" 1
run_variant "no_text" "no_text" 1
run_variant "shuffled_text" "shuffled_text" 1
run_variant "lagged_text" "lagged_text" "${SEAS_TEXT_LAG_STEPS:-1}"
run_variant "random_text" "random_text" 1

echo "Finished all SEAS TimeMMD text-control experiments."
echo "Metric summaries: ${BASE_OUT}"
