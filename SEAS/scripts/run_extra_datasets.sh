#!/usr/bin/env bash
set -euo pipefail

GPU0="${1:-0}"
GPU2="${2:-2}"
SEED="${3:-2024}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

DATA_ROOT="${EXTRA2_DATA_ROOT:-${REPO_DIR}/data/icde_experiment2_extra}"
OUT_ROOT="${EXTRA2_OUT_ROOT:-${REPO_DIR}/outputs/experiment2_extra/seed_${SEED}}"
RUN_GROUPS="${EXTRA2_RUN_GROUPS:-both}"
RUN_MODELS="${EXTRA2_RUN_MODELS:-all}"
SKIP_EXISTING="${EXTRA2_SKIP_EXISTING:-1}"
DOMAIN_FILTER="${EXTRA2_DOMAINS:-all}"
SEAS_CONDA_ENV="${SEAS_CONDA_ENV:-tsfm_env}"
SEAS_LLM_PATH="${SEAS_LLM_PATH:-${REPO_DIR}/pretrained/bert}"
EXTRA2_GPT2_PATH="${EXTRA2_GPT2_PATH:-${REPO_DIR}/pretrained/gpt2}"
TIMEXER_ROOT="${EXTRA2_TIMEXER_ROOT:-${REPO_DIR}/../TimeXer-main}"
AURORA_ROOT="${EXTRA2_AURORA_ROOT:-${REPO_DIR}/../Aurora-main/VoT-main}"
MMTSFLIB_ROOT="${EXTRA2_MMTSFLIB_ROOT:-${REPO_DIR}/../MM-TSFlib-main/MM-TSFlib-main}"
TIMELLM_ROOT="${EXTRA2_TIMELLM_ROOT:-${REPO_DIR}/../Time-LLM-main/Time-LLM-main}"

# Follow the supplementary-dataset protocol already used in MEMOIR_V2.
SEQ_LEN_DAILY="${EXTRA2_SEQ_LEN_DAILY:-24}"
PRED_LENS_DAILY="${EXTRA2_PRED_LENS_DAILY:-6 8 10 12}"
SEQ_LEN_HOURLY="${EXTRA2_SEQ_LEN_HOURLY:-48}"
PRED_LENS_HOURLY="${EXTRA2_PRED_LENS_HOURLY:-12 24 36 48}"
LABEL_LEN="${EXTRA2_LABEL_LEN:-0}"

python "${SCRIPT_DIR}/prepare_extra_datasets.py" --output-root "${DATA_ROOT}"

activate_conda_env() {
  if ! command -v conda >/dev/null 2>&1; then
    return 0
  fi
  local conda_base
  conda_base="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${conda_base}" && -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${SEAS_CONDA_ENV}"
  fi
}

require_dir() {
  local dir_path="$1"
  local dir_name="$2"
  if [[ ! -d "${dir_path}" ]]; then
    echo "[error] Missing ${dir_name}: ${dir_path}" >&2
    return 1
  fi
}

should_run_model() {
  local model_name="$1"
  if [[ "${RUN_MODELS}" == "all" ]]; then
    return 0
  fi
  [[ ",${RUN_MODELS}," == *",${model_name},"* ]]
}

result_has_model_id() {
  local result_file="$1"
  local model_id="$2"
  if [[ ! -f "${result_file}" ]]; then
    return 1
  fi
  if command -v rg >/dev/null 2>&1; then
    rg -q "${model_id}" "${result_file}"
  else
    grep -q "${model_id}" "${result_file}"
  fi
}

should_run_domain() {
  local domain_name="$1"
  if [[ "${DOMAIN_FILTER}" == "all" ]]; then
    return 0
  fi
  [[ ",${DOMAIN_FILTER}," == *",${domain_name},"* ]]
}

run_seas() {
  activate_conda_env
  export PYTHONNOUSERSITE=1
  export TOKENIZERS_PARALLELISM=true
  cd "${REPO_DIR}"
  local out_dir="${OUT_ROOT}/seas"
  local stdout_dir="${out_dir}/stdout"
  mkdir -p "${out_dir}" "${stdout_dir}"
  for domain in TTC_Climate FNF_AULoad_NSW; do
    if ! should_run_domain "${domain}"; then
      continue
    fi
    local seq_len="${SEQ_LEN_DAILY}"
    local pred_lens="${PRED_LENS_DAILY}"
    local freq="d"
    if [[ "${domain}" == "FNF_AULoad_NSW" ]]; then
      seq_len="${SEQ_LEN_HOURLY}"
      pred_lens="${PRED_LENS_HOURLY}"
      freq="h"
    fi
    for pred_len in ${pred_lens}; do
      local model_id="seas_freqlookback_bert_${domain}_s${SEED}_sl${seq_len}_p${pred_len}"
      local log_file="${stdout_dir}/${model_id}.log"
      local result_file="${out_dir}/result_seas_${domain}.txt"
      if [[ "${SKIP_EXISTING}" == "1" ]] && result_has_model_id "${result_file}" "${model_id}"; then
        echo "[SEAS][skip] ${domain} sl=${seq_len} pl=${pred_len}"
        continue
      fi
      echo "[SEAS][gpu ${GPU0}] ${domain} sl=${seq_len} pl=${pred_len}"
      CUDA_VISIBLE_DEVICES="${GPU0}" python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${DATA_ROOT}" \
        --data_path "${domain}.csv" \
        --model_id "${model_id}" \
        --model TimeMixer \
        --data custom \
        --features S \
        --target OT \
        --freq "${freq}" \
        --seq_len "${seq_len}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --des "S${SEED}_SEAS_extra2" \
        --seed "${SEED}" \
        --batch_size 32 \
        --train_epochs 50 \
        --patience 20 \
        --learning_rate 0.005 \
        --learning_rate2 0.005 \
        --learning_rate3 0.001 \
        --learning_rate_weight 0.0001 \
        --method SEAS \
        --llm_model BERT \
        --llm_path "${SEAS_LLM_PATH}" \
        --llm_emb_size 768 \
        --llm_layers 6 \
        --text_name fact \
        --text_emb 32 \
        --prompt_weight 0.1 \
        --pool_type avg \
        --n_ts_features 1 \
        --mm_emb_size 16 \
        --mm_hidden_size 16 \
        --text_dropout 0.2 \
        --dropout 0.1 \
        --num_workers 0 \
        --use_fullmodel 0 \
        --down_sampling_layers 2 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --d_model 512 \
        --d_ff 1024 \
        --proj_per_freq \
        --fuse_history \
        --use_product \
        --use_seas 1 \
        --save_name "${result_file}" \
        > "${log_file}" 2>&1
    done
  done
}

run_spectral_baseline() {
  activate_conda_env
  export PYTHONNOUSERSITE=1
  export TOKENIZERS_PARALLELISM=true
  cd "${REPO_DIR}"
  local out_dir="${OUT_ROOT}/spectral_baseline"
  local stdout_dir="${out_dir}/stdout"
  mkdir -p "${out_dir}" "${stdout_dir}"
  for domain in TTC_Climate FNF_AULoad_NSW; do
    if ! should_run_domain "${domain}"; then
      continue
    fi
    local seq_len="${SEQ_LEN_DAILY}"
    local pred_lens="${PRED_LENS_DAILY}"
    local freq="d"
    if [[ "${domain}" == "FNF_AULoad_NSW" ]]; then
      seq_len="${SEQ_LEN_HOURLY}"
      pred_lens="${PRED_LENS_HOURLY}"
      freq="h"
    fi
    for pred_len in ${pred_lens}; do
      local model_id="seas_freqlookback_bert_${domain}_s${SEED}_sl${seq_len}_p${pred_len}"
      local log_file="${stdout_dir}/${model_id}.log"
      local result_file="${out_dir}/result_spectral_baseline_${domain}.txt"
      if [[ "${SKIP_EXISTING}" == "1" ]] && result_has_model_id "${result_file}" "${model_id}"; then
        echo "[SpectralBaseline][skip] ${domain} sl=${seq_len} pl=${pred_len}"
        continue
      fi
      echo "[SpectralBaseline][gpu ${GPU0}] ${domain} sl=${seq_len} pl=${pred_len}"
      CUDA_VISIBLE_DEVICES="${GPU0}" python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${DATA_ROOT}" \
        --data_path "${domain}.csv" \
        --model_id "${model_id}" \
        --model TimeMixer \
        --data custom \
        --features S \
        --target OT \
        --freq "${freq}" \
        --seq_len "${seq_len}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --des "S${SEED}_SEAS_extra2" \
        --seed "${SEED}" \
        --batch_size 32 \
        --train_epochs 50 \
        --patience 20 \
        --learning_rate 0.005 \
        --learning_rate2 0.005 \
        --learning_rate3 0.001 \
        --learning_rate_weight 0.0001 \
        --method SEAS \
        --llm_model BERT \
        --llm_path "${SEAS_LLM_PATH}" \
        --llm_emb_size 768 \
        --llm_layers 6 \
        --text_name fact \
        --text_emb 32 \
        --prompt_weight 0.1 \
        --pool_type avg \
        --n_ts_features 1 \
        --mm_emb_size 16 \
        --mm_hidden_size 16 \
        --text_dropout 0.2 \
        --dropout 0.1 \
        --num_workers 0 \
        --use_fullmodel 0 \
        --down_sampling_layers 2 \
        --down_sampling_method avg \
        --down_sampling_window 2 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --d_model 512 \
        --d_ff 1024 \
        --proj_per_freq \
        --fuse_history \
        --use_product \
        --save_name "${result_file}" \
        > "${log_file}" 2>&1
    done
  done
}

run_timexer() {
  activate_conda_env
  require_dir "${TIMEXER_ROOT}" "TimeXer root" || return 1
  export PYTHONNOUSERSITE=1
  cd "${TIMEXER_ROOT}"
  local out_dir="${OUT_ROOT}/timexer"
  local stdout_dir="${out_dir}/stdout"
  mkdir -p "${out_dir}" "${stdout_dir}"
  for domain in TTC_Climate FNF_AULoad_NSW; do
    if ! should_run_domain "${domain}"; then
      continue
    fi
    local seq_len="${SEQ_LEN_DAILY}"
    local pred_lens="${PRED_LENS_DAILY}"
    local freq="d"
    if [[ "${domain}" == "FNF_AULoad_NSW" ]]; then
      seq_len="${SEQ_LEN_HOURLY}"
      pred_lens="${PRED_LENS_HOURLY}"
      freq="h"
    fi
    for pred_len in ${pred_lens}; do
      local model_id="timexer_freqlookback_${domain}_s${SEED}_sl${seq_len}_p${pred_len}"
      local log_file="${stdout_dir}/${model_id}.log"
      echo "[TimeXer][gpu ${GPU0}] ${domain} sl=${seq_len} pl=${pred_len}"
      CUDA_VISIBLE_DEVICES="${GPU0}" python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "${model_id}" \
        --model TimeXer \
        --data custom \
        --root_path "${DATA_ROOT}" \
        --data_path "${domain}.csv" \
        --features S \
        --target OT \
        --freq "${freq}" \
        --checkpoints "${out_dir}/checkpoints" \
        --seq_len "${seq_len}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --enc_in 1 --dec_in 1 --c_out 1 \
        --d_model 512 --d_ff 2048 --n_heads 8 --e_layers 2 --d_layers 1 \
        --factor 1 --patch_len 8 \
        --des "TIMEXER_EXTRA2" \
        --train_epochs 10 --batch_size 32 --num_workers 0 --patience 3 \
        --learning_rate 0.0001 --itr 1 --seed "${SEED}" \
        > "${log_file}" 2>&1
    done
  done
}

run_gpt4mts() {
  activate_conda_env
  require_dir "${AURORA_ROOT}" "Aurora/VoT root" || return 1
  export PYTHONNOUSERSITE=1
  cd "${AURORA_ROOT}"
  local out_dir="${OUT_ROOT}/gpt4mts"
  local stdout_dir="${out_dir}/stdout"
  mkdir -p "${out_dir}" "${stdout_dir}"
  for domain in TTC_Climate FNF_AULoad_NSW; do
    if ! should_run_domain "${domain}"; then
      continue
    fi
    local seq_len="${SEQ_LEN_DAILY}"
    local pred_lens="${PRED_LENS_DAILY}"
    local freq="d"
    if [[ "${domain}" == "FNF_AULoad_NSW" ]]; then
      seq_len="${SEQ_LEN_HOURLY}"
      pred_lens="${PRED_LENS_HOURLY}"
      freq="h"
    fi
    for pred_len in ${pred_lens}; do
      local model_id="gpt4mts_freqlookback_${domain}_s${SEED}_sl${seq_len}_p${pred_len}"
      local log_file="${stdout_dir}/${model_id}.log"
      local result_file="${out_dir}/result_gpt4mts_${domain}.txt"
      if [[ "${SKIP_EXISTING}" == "1" ]] && result_has_model_id "${result_file}" "${model_id}"; then
        echo "[GPT4MTS][skip] ${domain} sl=${seq_len} pl=${pred_len}"
        continue
      fi
      echo "[GPT4MTS][gpu ${GPU0}] ${domain} sl=${seq_len} pl=${pred_len}"
      CUDA_VISIBLE_DEVICES="${GPU0}" python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${DATA_ROOT}" \
        --data_path "${domain}.csv" \
        --model_id "${model_id}" \
        --model GPT4MTS \
        --data custom \
        --features S \
        --target OT \
        --freq "${freq}" \
        --seq_len "${seq_len}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --text_emb 96 \
        --des "FREQLOOKBACK_GPT4MTS_EXTRA2" \
        --seed "${SEED}" \
        --llm_model GPT2 \
        --llm_path "${EXTRA2_GPT2_PATH}" \
        --llm_layers 6 \
        --d_model 512 --d_ff 2048 --n_heads 8 --e_layers 2 --d_layers 1 \
        --batch_size 16 --train_epochs 10 --num_workers 0 --patience 3 \
        --learning_rate 0.0001 --learning_rate2 0.01 --learning_rate3 0.001 \
        --text_len 4 --pool_type avg \
        --checkpoints "${out_dir}/checkpoints" \
        --save_name "${result_file}" \
        --devices "${GPU0}" > "${log_file}" 2>&1
    done
  done
}

run_mmtsflib() {
  activate_conda_env
  require_dir "${MMTSFLIB_ROOT}" "MM-TSFlib root" || return 1
  export PYTHONNOUSERSITE=1
  export BERT_MODEL_PATH="${SEAS_LLM_PATH}"
  cd "${MMTSFLIB_ROOT}"
  local out_dir="${OUT_ROOT}/mm_tsflib"
  local stdout_dir="${out_dir}/stdout"
  mkdir -p "${out_dir}/results" "${out_dir}/test_results" "${out_dir}/checkpoints" "${stdout_dir}"
  for domain in TTC_Climate FNF_AULoad_NSW; do
    if ! should_run_domain "${domain}"; then
      continue
    fi
    local seq_len="${SEQ_LEN_DAILY}"
    local pred_lens="${PRED_LENS_DAILY}"
    local freq="d"
    if [[ "${domain}" == "FNF_AULoad_NSW" ]]; then
      seq_len="${SEQ_LEN_HOURLY}"
      pred_lens="${PRED_LENS_HOURLY}"
      freq="h"
    fi
    for pred_len in ${pred_lens}; do
      local model_id="mmtsflib_freqlookback_${domain}_s${SEED}_sl${seq_len}_p${pred_len}_iTransformer"
      local log_file="${stdout_dir}/${model_id}.log"
      local result_file="${out_dir}/result_mmtsflib_${domain}.txt"
      if [[ "${SKIP_EXISTING}" == "1" ]] && result_has_model_id "${result_file}" "${model_id}"; then
        echo "[MM-TSFlib][skip] ${domain} sl=${seq_len} pl=${pred_len}"
        continue
      fi
      echo "[MM-TSFlib][gpu ${GPU2}] ${domain} sl=${seq_len} pl=${pred_len}"
      CUDA_VISIBLE_DEVICES="${GPU2}" python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --use_gpu True --gpu 0 \
        --root_path "${DATA_ROOT}" \
        --data_path "${domain}.csv" \
        --model_id "${model_id}" \
        --model iTransformer \
        --data custom \
        --features S \
        --target OT \
        --freq "${freq}" \
        --seq_len "${seq_len}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --enc_in 1 --dec_in 1 --c_out 1 \
        --d_model 512 --d_ff 2048 --n_heads 8 --e_layers 2 --d_layers 1 \
        --des "MMTSFLIB_FREQLOOKBACK_BERT_EXTRA2" \
        --seed "${SEED}" \
        --train_epochs 10 --batch_size 16 --num_workers 0 --patience 3 \
        --learning_rate 0.0001 \
        --llm_model BERT \
        --type_tag "#F#" \
        --text_len 4 \
        --prompt_weight 0.1 \
        --pool_type avg \
        --use_fullmodel 0 \
        --huggingface_token NA \
        --checkpoints "${out_dir}/checkpoints" \
        --save_name "${result_file}" \
        > "${log_file}" 2>&1
    done
  done
}

run_gpt4ts() {
  activate_conda_env
  require_dir "${AURORA_ROOT}" "Aurora/VoT root" || return 1
  export PYTHONNOUSERSITE=1
  cd "${AURORA_ROOT}"
  local out_dir="${OUT_ROOT}/gpt4ts"
  local stdout_dir="${out_dir}/stdout"
  mkdir -p "${out_dir}" "${stdout_dir}"
  for domain in TTC_Climate FNF_AULoad_NSW; do
    if ! should_run_domain "${domain}"; then
      continue
    fi
    local seq_len="${SEQ_LEN_DAILY}"
    local pred_lens="${PRED_LENS_DAILY}"
    local freq="d"
    if [[ "${domain}" == "FNF_AULoad_NSW" ]]; then
      seq_len="${SEQ_LEN_HOURLY}"
      pred_lens="${PRED_LENS_HOURLY}"
      freq="h"
    fi
    for pred_len in ${pred_lens}; do
      local model_id="gpt4ts_freqlookback_${domain}_s${SEED}_sl${seq_len}_p${pred_len}"
      local log_file="${stdout_dir}/${model_id}.log"
      local result_file="${out_dir}/result_gpt4ts_${domain}.txt"
      if [[ "${SKIP_EXISTING}" == "1" ]] && result_has_model_id "${result_file}" "${model_id}"; then
        echo "[GPT4TS][skip] ${domain} sl=${seq_len} pl=${pred_len}"
        continue
      fi
      echo "[GPT4TS][gpu ${GPU2}] ${domain} sl=${seq_len} pl=${pred_len}"
      CUDA_VISIBLE_DEVICES="${GPU2}" python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path "${DATA_ROOT}" \
        --data_path "${domain}.csv" \
        --model_id "${model_id}" \
        --model GPT4TS \
        --data custom \
        --features S \
        --target OT \
        --freq "${freq}" \
        --seq_len "${seq_len}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --text_emb 96 \
        --des "FREQLOOKBACK_GPT4TS_EXTRA2" \
        --seed "${SEED}" \
        --llm_model GPT2 \
        --llm_path "${EXTRA2_GPT2_PATH}" \
        --llm_layers 6 \
        --d_model 768 --d_ff 2048 --n_heads 8 --e_layers 2 --d_layers 1 \
        --batch_size 16 --train_epochs 10 --num_workers 0 --patience 3 \
        --learning_rate 0.0001 --learning_rate2 0.01 --learning_rate3 0.001 \
        --text_len 4 --pool_type avg \
        --checkpoints "${out_dir}/checkpoints" \
        --save_name "${result_file}" \
        --devices "${GPU2}" > "${log_file}" 2>&1
    done
  done
}

run_timellm() {
  activate_conda_env
  require_dir "${TIMELLM_ROOT}" "Time-LLM root" || return 1
  export PYTHONNOUSERSITE=1
  cd "${TIMELLM_ROOT}"
  local out_dir="${OUT_ROOT}/time_llm"
  local stdout_dir="${out_dir}/stdout"
  mkdir -p "${out_dir}" "${stdout_dir}"
  local port_base=29720
  local idx=0
  for domain in TTC_Climate FNF_AULoad_NSW; do
    if ! should_run_domain "${domain}"; then
      continue
    fi
    local seq_len="${SEQ_LEN_DAILY}"
    local pred_lens="${PRED_LENS_DAILY}"
    local freq="d"
    if [[ "${domain}" == "FNF_AULoad_NSW" ]]; then
      seq_len="${SEQ_LEN_HOURLY}"
      pred_lens="${PRED_LENS_HOURLY}"
      freq="h"
    fi
    for pred_len in ${pred_lens}; do
      local model_id="timellm_freqlookback_${domain}_s${SEED}_sl${seq_len}_p${pred_len}"
      local log_file="${stdout_dir}/${model_id}.log"
      local port=$((port_base + idx))
      idx=$((idx + 1))
      echo "[Time-LLM][gpu ${GPU2}] ${domain} sl=${seq_len} pl=${pred_len}"
      CUDA_VISIBLE_DEVICES="${GPU2}" PYTHONNOUSERSITE=1 accelerate launch \
        --num_processes 1 \
        --main_process_port "${port}" \
        run_main.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "${model_id}" \
        --model_comment "FREQLOOKBACK_BERT_EXTRA2" \
        --model TimeLLM \
        --data Traffic \
        --root_path "${DATA_ROOT}" \
        --data_path "${domain}.csv" \
        --features S \
        --target OT \
        --freq "${freq}" \
        --checkpoints "${out_dir}/checkpoints" \
        --seq_len "${seq_len}" \
        --label_len "${LABEL_LEN}" \
        --pred_len "${pred_len}" \
        --enc_in 1 --dec_in 1 --c_out 1 \
        --d_model 16 --d_ff 32 --n_heads 8 --e_layers 2 --d_layers 1 \
        --factor 1 --patch_len 8 --stride 8 \
        --llm_model BERT --llm_dim 768 --llm_layers 6 \
        --train_epochs 10 --batch_size 8 --eval_batch_size 8 --num_workers 0 \
        --patience 3 --learning_rate 0.01 \
        --des "TIME_LLM_FREQLOOKBACK_BERT_EXTRA2" \
        --itr 1 --seed "${SEED}" > "${log_file}" 2>&1
    done
  done
}

echo "Experiment 2 extra datasets"
echo "GPU ${GPU0}: SEAS, spectral baseline, TimeXer, GPT4MTS"
echo "GPU ${GPU2}: MM-TSFlib, GPT4TS, Time-LLM"
echo "Datasets: TTC_Climate, FNF_AULoad_NSW"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUT_ROOT}"

pid0=""
pid2=""
if [[ "${RUN_GROUPS}" == "both" || "${RUN_GROUPS}" == *"gpu0"* ]]; then
  (
    if should_run_model seas; then run_seas; fi
    if should_run_model spectral_baseline; then run_spectral_baseline; fi
    if should_run_model timexer; then run_timexer; fi
    if should_run_model gpt4mts; then run_gpt4mts; fi
  ) &
  pid0=$!
fi

if [[ "${RUN_GROUPS}" == "both" || "${RUN_GROUPS}" == *"gpu2"* ]]; then
  (
    if should_run_model mmtsflib; then run_mmtsflib; fi
    if should_run_model gpt4ts; then run_gpt4ts; fi
    if should_run_model timellm; then run_timellm; fi
  ) &
  pid2=$!
fi

if [[ -n "${pid0}" ]]; then
  wait "${pid0}"
fi
if [[ -n "${pid2}" ]]; then
  wait "${pid2}"
fi

echo "All Experiment 2 runs finished."
echo "Outputs saved to: ${OUT_ROOT}"
