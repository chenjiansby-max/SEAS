#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-0,1,6}"
SEED="${2:-2024}"
DOMAINS="${3:-Climate,Security,Traffic,Health}"

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
BASE_OUT="${SEAS_ABLATION_BASE_OUT:-${REPO_DIR}/outputs/timemmd_protocol/seas_ablation_seed${SEED}}"

python scripts/prepare_seas_timemmd_from_mmts.py \
  --output-root "${DATA_ROOT}" \
  --text-col "${SEAS_TIMEMMD_TEXT_COL:-Final_Search_4}"

run_variant() {
  local tag="$1"
  local disable_asd="$2"
  local disable_sed="$3"
  local disable_sfm="$4"
  local disable_hsg="$5"
  local disable_text_s4d="$6"
  local out_dir="${BASE_OUT}/${tag}"
  local stdout_dir="${out_dir}/stdout"

  mkdir -p "${out_dir}" "${stdout_dir}"
  echo "=== SEAS ablation: ${tag} | GPUs=${GPUS} | domains=${DOMAINS} ==="

  SEAS_DATA_ROOT="${DATA_ROOT}" \
  SEAS_FREQLOOKBACK_OUT_DIR="${out_dir}" \
  SEAS_STDOUT_DIR="${stdout_dir}" \
  SEAS_ABLATION_TAG="${tag}" \
  SEAS_DISABLE_ASD="${disable_asd}" \
  SEAS_DISABLE_SED="${disable_sed}" \
  SEAS_DISABLE_SFM="${disable_sfm}" \
  SEAS_DISABLE_HSG="${disable_hsg}" \
  SEAS_DISABLE_TEXT_S4D="${disable_text_s4d}" \
  bash scripts/run_seas_freqlookback_bert_timemixer.sh "${GPUS}" "${SEED}" "${DOMAINS}"
}

run_variant "full" 0 0 0 0 0
run_variant "wo_asd" 1 0 0 0 0
run_variant "wo_sed" 0 1 0 0 0
run_variant "wo_sfm" 0 0 1 0 0
run_variant "wo_hsg" 0 0 0 1 0
run_variant "wo_text_s4d" 0 0 0 0 1

echo "Finished all SEAS TimeMMD ablation experiments."
echo "Metric summaries: ${BASE_OUT}"
