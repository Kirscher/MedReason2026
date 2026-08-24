#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/local_outputs/gpu_sweep_logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}/docker/medreason"
PYTHON="${PYTHON:-python3}"
MEDREASON_DATA_ROOT="${MEDREASON_DATA_ROOT:-${ROOT_DIR}/data}"

run_job() {
  local name="$1"
  shift
  local out_dir="${ROOT_DIR}/local_outputs/${name}"
  mkdir -p "${out_dir}"
  {
    echo "===== ${name} START $(date -Is) ====="
    "$@"
    echo "===== ${name} END $(date -Is) ====="
  } 2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_medreason() {
  local name="$1"
  local input_dir="$2"
  local model_path="$3"
  local retrieval_bank="$4"
  shift 4
  env \
    MEDREASON_INPUT_DIR="${input_dir}" \
    MEDREASON_OUTPUT_DIR="${ROOT_DIR}/local_outputs/${name}" \
    MEDREASON_OUTPUT_FILE="${ROOT_DIR}/local_outputs/${name}/results.json" \
    MEDREASON_SYSTEM=strong_baseline \
    MEDREASON_MODEL_PATH="${model_path}" \
    MEDREASON_RETRIEVAL_BANK="${retrieval_bank}" \
    MEDREASON_MCQ_POLICY=option_aware \
    "$@" \
    "${PYTHON}" "${ROOT_DIR}/docker/medreason/process.py"
  "${PYTHON}" "${ROOT_DIR}/scripts/validate_predictions.py" \
    --cases-json "${input_dir}/cases.json" \
    --results-json "${ROOT_DIR}/local_outputs/${name}/results.json"
  if [[ -f "${input_dir}/ground_truth.json" ]]; then
    "${PYTHON}" "${ROOT_DIR}/scripts/score_predictions.py" \
      --ground-truth "${input_dir}/ground_truth.json" \
      --results-json "${ROOT_DIR}/local_outputs/${name}/results.json"
    "${PYTHON}" "${ROOT_DIR}/scripts/analyze_mcq_disagreements.py" \
      --ground-truth "${input_dir}/ground_truth.json" \
      --results-json "${ROOT_DIR}/local_outputs/${name}/results.json" \
      --output-json "${ROOT_DIR}/local_outputs/${name}/mcq_disagreements.json"
  fi
}

run_job holdout220_qwen3b_k1_option_aware run_medreason \
  holdout220_qwen3b_k1_option_aware \
  "${ROOT_DIR}/local_inputs/holdout_220" \
  "${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-3B-Instruct" \
  "${ROOT_DIR}/artifacts/retrieval_bank_holdout220_excluded.json" \
  MEDREASON_TOP_K_EXAMPLES=1 \
  MEDREASON_OPTION_RETRIEVAL_TOP_K=5 \
  MEDREASON_OPTION_SCORE_MODE=sum5 \
  MEDREASON_MAX_NEW_TOKENS=128

run_job holdout220_qwen3b_k0_option_aware run_medreason \
  holdout220_qwen3b_k0_option_aware \
  "${ROOT_DIR}/local_inputs/holdout_220" \
  "${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-3B-Instruct" \
  "${ROOT_DIR}/artifacts/retrieval_bank_holdout220_excluded.json" \
  MEDREASON_TOP_K_EXAMPLES=0 \
  MEDREASON_OPTION_RETRIEVAL_TOP_K=5 \
  MEDREASON_OPTION_SCORE_MODE=sum5 \
  MEDREASON_MAX_NEW_TOKENS=128

run_job holdout220_qwen7b_4bit_k1_option_aware run_medreason \
  holdout220_qwen7b_4bit_k1_option_aware \
  "${ROOT_DIR}/local_inputs/holdout_220" \
  "${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-7B-Instruct" \
  "${ROOT_DIR}/artifacts/retrieval_bank_holdout220_excluded.json" \
  MEDREASON_TOP_K_EXAMPLES=1 \
  MEDREASON_OPTION_RETRIEVAL_TOP_K=5 \
  MEDREASON_OPTION_SCORE_MODE=sum5 \
  MEDREASON_LOAD_IN_4BIT=1 \
  MEDREASON_MAX_NEW_TOKENS=128
