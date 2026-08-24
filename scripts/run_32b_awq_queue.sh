#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/local_outputs/gpu_sweep_logs"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${ROOT_DIR}/docker/medreason"
PYTHON="${PYTHON:-python3}"
MEDREASON_DATA_ROOT="${MEDREASON_DATA_ROOT:-${ROOT_DIR}/data}"
MODEL_PATH="${MODEL_PATH:-${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-32B-Instruct-AWQ}"

setup_awq_cuda() {
  local cuda_home="${MEDREASON_CUDA_HOME:-${CUDA_HOME:-}}"
  if [[ -d "${cuda_home}" ]]; then
    export CUDA_HOME="${cuda_home}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
    if [[ -f "${CUDA_HOME}/lib/libcudart.so.13" ]]; then
      ln -sfn libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"
    fi
  fi
}

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
  local retrieval_bank="$3"
  shift 3
  env \
    MEDREASON_INPUT_DIR="${input_dir}" \
    MEDREASON_OUTPUT_DIR="${ROOT_DIR}/local_outputs/${name}" \
    MEDREASON_OUTPUT_FILE="${ROOT_DIR}/local_outputs/${name}/results.json" \
    MEDREASON_SYSTEM=strong_baseline \
    MEDREASON_MODEL_PATH="${MODEL_PATH}" \
    MEDREASON_RETRIEVAL_BANK="${retrieval_bank}" \
    MEDREASON_MCQ_POLICY=option_aware \
    MEDREASON_DTYPE=float16 \
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

setup_awq_cuda

if [[ "${MEDREASON_SKIP_HOLDOUT25:-0}" != "1" ]]; then
  run_job holdout25_qwen32b_awq_k1_option_aware run_medreason \
    holdout25_qwen32b_awq_k1_option_aware \
    "${ROOT_DIR}/local_inputs/holdout_25" \
    "${ROOT_DIR}/artifacts/retrieval_bank_holdout25_excluded.json" \
    MEDREASON_TOP_K_EXAMPLES=1 \
    MEDREASON_OPTION_RETRIEVAL_TOP_K=5 \
    MEDREASON_OPTION_SCORE_MODE=sum5 \
    MEDREASON_MAX_NEW_TOKENS=160
fi

run_job holdout220_qwen32b_awq_k1_option_aware run_medreason \
  holdout220_qwen32b_awq_k1_option_aware \
  "${ROOT_DIR}/local_inputs/holdout_220" \
  "${ROOT_DIR}/artifacts/retrieval_bank_holdout220_excluded.json" \
  MEDREASON_TOP_K_EXAMPLES=1 \
  MEDREASON_OPTION_RETRIEVAL_TOP_K=5 \
  MEDREASON_OPTION_SCORE_MODE=sum5 \
  MEDREASON_MAX_NEW_TOKENS=160
