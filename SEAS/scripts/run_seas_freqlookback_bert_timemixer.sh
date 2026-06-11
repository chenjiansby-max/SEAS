#!/usr/bin/env bash
set -u

GPUS="${1:-7}"
SEEDS="${2:-2024}"
DOMAINS="${3:-Agriculture,Climate,Economy,Energy,Environment,Health,Security,SocialGood,Traffic}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}" || exit 1

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

DATA_ROOT="${SEAS_DATA_ROOT:-./data/timemmd_mmts_tats}"
LLM_PATH="${SEAS_LLM_PATH:-./pretrained/bert}"
OUT_DIR="${SEAS_FREQLOOKBACK_OUT_DIR:-./outputs/seas_freqlookback_bert}"
STDOUT_DIR="${SEAS_STDOUT_DIR:-./outputs/stdout}"
mkdir -p "${OUT_DIR}" "${STDOUT_DIR}"
ABLATION_TAG="${SEAS_ABLATION_TAG:-}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a DOMAIN_LIST <<< "${DOMAINS}"

MODEL_NAME="TimeMixer"
LABEL_LEN=0
BATCH_SIZE="${SEAS_BATCH_SIZE:-32}"
TRAIN_EPOCHS="${SEAS_TRAIN_EPOCHS:-50}"
PATIENCE="${SEAS_PATIENCE:-20}"
LEARNING_RATE="${SEAS_LR:-0.005}"
LEARNING_RATE2="${SEAS_LR2:-0.005}"
LEARNING_RATE3="${SEAS_LR3:-0.001}"
LEARNING_RATE_WEIGHT="${SEAS_LR_WEIGHT:-0.0001}"
PROMPT_WEIGHT="${SEAS_PROMPT_WEIGHT:-0.1}"
TEXT_EMB="${SEAS_TEXT_EMB:-32}"
MM_EMB_SIZE="${SEAS_MM_EMB_SIZE:-16}"
MM_HIDDEN_SIZE="${SEAS_MM_HIDDEN_SIZE:-16}"
D_MODEL="${SEAS_D_MODEL:-512}"
D_FF="${SEAS_D_FF:-1024}"
E_LAYERS="${SEAS_E_LAYERS:-2}"
D_LAYERS="${SEAS_D_LAYERS:-1}"
FACTOR="${SEAS_FACTOR:-3}"
DROPOUT="${SEAS_DROPOUT:-0.1}"
TEXT_DROPOUT="${SEAS_TEXT_DROPOUT:-0.2}"
LLM_EMB_SIZE=768

echo "SEAS frequency-aware lookback run with BERT"
echo "Repo: ${REPO_DIR}"
echo "Python: $(which python) ($(python --version 2>&1))"
echo "Conda env: ${CONDA_DEFAULT_ENV:-NA}"
echo "GPUs: ${GPUS}"
echo "Seeds: ${SEEDS}"
echo "Domains: ${DOMAINS}"
echo "Lookback protocol: Monthly=12, Weekly=24, Daily=96"
echo "LLM path: ${LLM_PATH}"
echo "Output: ${OUT_DIR}"
echo "Text control: ${SEAS_TEXT_CONTROL_MODE:-aligned}"
if [ -n "${ABLATION_TAG}" ]; then
  echo "Ablation tag: ${ABLATION_TAG}"
fi

horizons_for_domain() {
  case "$1" in
    Energy|Health) echo "12 24 36 48" ;;
    Environment) echo "48 96 192 336" ;;
    *) echo "6 8 10 12" ;;
  esac
}

seq_len_for_domain() {
  case "$1" in
    Energy|Health) echo "24" ;;
    Environment) echo "96" ;;
    *) echo "12" ;;
  esac
}

run_one() {
  local gpu="$1"
  local seed="$2"
  local domain="$3"
  local pred_len="$4"
  local seq_len
  seq_len="$(seq_len_for_domain "${domain}")"
  local name_prefix="seas_freqlookback_bert"
  local des_suffix="SEAS_freqlookback_BERT"
  if [ -n "${ABLATION_TAG}" ]; then
    name_prefix="seas_${ABLATION_TAG}_freqlookback_bert"
    des_suffix="SEAS_${ABLATION_TAG}_freqlookback_BERT"
  fi
  local save_name="${OUT_DIR}/result_${name_prefix}_${domain}.txt"
  local log_name="${STDOUT_DIR}/${name_prefix}_${domain}_s${seed}_sl${seq_len}_p${pred_len}.log"

  echo "[SEAS-freqlookback-BERT][gpu ${gpu}] ${domain} seed=${seed} seq_len=${seq_len} pred_len=${pred_len}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${DATA_ROOT}" \
    --data_path "${domain}.csv" \
    --model_id "${name_prefix}_${domain}_s${seed}_sl${seq_len}_p${pred_len}" \
    --model "${MODEL_NAME}" \
    --data custom \
    --features S \
    --target OT \
    --seq_len "${seq_len}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${pred_len}" \
    --des "S${seed}_${des_suffix}" \
    --seed "${seed}" \
    --batch_size "${BATCH_SIZE}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --patience "${PATIENCE}" \
    --learning_rate "${LEARNING_RATE}" \
    --learning_rate2 "${LEARNING_RATE2}" \
    --learning_rate3 "${LEARNING_RATE3}" \
    --learning_rate_weight "${LEARNING_RATE_WEIGHT}" \
    --method SEAS \
    --llm_model BERT \
    --llm_path "${LLM_PATH}" \
    --llm_emb_size "${LLM_EMB_SIZE}" \
    --llm_layers 6 \
    --text_name fact \
    --text_control_mode "${SEAS_TEXT_CONTROL_MODE:-aligned}" \
    --text_shift_steps "${SEAS_TEXT_SHIFT_STEPS:-1}" \
    --text_control_seed "${SEAS_TEXT_CONTROL_SEED:-${seed}}" \
    --text_emb "${TEXT_EMB}" \
    --prompt_weight "${PROMPT_WEIGHT}" \
    --pool_type avg \
    --n_ts_features 1 \
    --mm_emb_size "${MM_EMB_SIZE}" \
    --mm_hidden_size "${MM_HIDDEN_SIZE}" \
    --text_dropout "${TEXT_DROPOUT}" \
    --dropout "${DROPOUT}" \
    --num_workers "${SEAS_NUM_WORKERS:-4}" \
    --use_fullmodel 0 \
    --down_sampling_layers 2 \
    --down_sampling_method avg \
    --down_sampling_window 2 \
    --e_layers "${E_LAYERS}" \
    --d_layers "${D_LAYERS}" \
    --factor "${FACTOR}" \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model "${D_MODEL}" \
    --d_ff "${D_FF}" \
    --proj_per_freq \
    --fuse_history \
    --use_product \
    --use_text_s4d_spectral 1 \
    --text_s4d_state "${SEAS_TEXT_S4D_STATE:-64}" \
    --text_s4d_dropout "${SEAS_TEXT_S4D_DROPOUT:-0.1}" \
    --use_seas 1 \
    --seas_num_bands "${SEAS_NUM_BANDS:-3}" \
    --seas_num_events "${SEAS_NUM_EVENTS:-4}" \
    --seas_hidden_dim "${SEAS_HIDDEN_DIM:-64}" \
    --seas_max_gain "${SEAS_MAX_GAIN:-0.5}" \
    --seas_max_phase "${SEAS_MAX_PHASE:-0.5}" \
    --seas_max_residual "${SEAS_MAX_RESIDUAL:-0.3}" \
    --seas_decomp_residual "${SEAS_DECOMP_RESIDUAL:-0.25}" \
    --seas_disable_asd "${SEAS_DISABLE_ASD:-0}" \
    --seas_disable_sed "${SEAS_DISABLE_SED:-0}" \
    --seas_disable_sfm "${SEAS_DISABLE_SFM:-0}" \
    --seas_disable_hsg "${SEAS_DISABLE_HSG:-0}" \
    --seas_disable_text_s4d "${SEAS_DISABLE_TEXT_S4D:-0}" \
    --save_name "${save_name}" \
    > "${log_name}" 2>&1
}

running=0
gpu_cursor=0
failed=0
for seed in "${SEED_LIST[@]}"; do
  for domain in "${DOMAIN_LIST[@]}"; do
    for pred_len in $(horizons_for_domain "${domain}"); do
      gpu="${GPU_LIST[$((gpu_cursor % ${#GPU_LIST[@]}))]}"
      gpu_cursor=$((gpu_cursor + 1))
      run_one "${gpu}" "${seed}" "${domain}" "${pred_len}" &
      running=$((running + 1))
      if [ "${running}" -ge "${#GPU_LIST[@]}" ]; then
        wait -n || failed=1
        running=$((running - 1))
      fi
    done
  done
done

while [ "${running}" -gt 0 ]; do
  wait -n || failed=1
  running=$((running - 1))
done

if [ "${failed}" -ne 0 ]; then
  echo "Finished with some failed SEAS frequency-aware lookback experiments."
  exit 1
fi

echo "Finished all SEAS frequency-aware lookback experiments."
echo "Metric summaries: ${OUT_DIR}"
echo "Full stdout logs: ${STDOUT_DIR}"
