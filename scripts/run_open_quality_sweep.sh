#!/usr/bin/env bash
# Autonomous open-quality sweep.
# Focuses on improving open-ended performance while keeping the current best MCQ adapter.
# Trains new open adapters with modality_guard and evidence_first+visual_answer styles,
# also tests inference-only variants (no new training). Packages the winner Docker image.
#
# Usage:
#   nohup bash scripts/run_open_quality_sweep.sh \
#     > artifacts/sweeps/open_quality/master.nohup.log 2>&1 &
#   echo $! > artifacts/sweeps/open_quality/master.pid
#
# Or with custom run dir:
#   MEDREASON_SWEEP_RUN_ID=open_quality_YYYYMMDD_HHMMSS bash scripts/run_open_quality_sweep.sh
set -uo pipefail
trap 'rc=$?; printf "[open_quality_sweep exit] rc=%s line=%s cmd=%q\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2' EXIT
if [ "${MEDREASON_SWEEP_DEBUG:-0}" = "1" ]; then
  set -x
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
MEDREASON_DATA_ROOT="${MEDREASON_DATA_ROOT:-${ROOT_DIR}/data}"

RUN_ID="${MEDREASON_SWEEP_RUN_ID:-open_quality_$(date +%Y%m%d_%H%M%S)}"
BASE_SWEEP_DIR="${MEDREASON_SWEEP_BASE_DIR:-${ROOT_DIR}/artifacts/sweeps}"
RUN_DIR="${MEDREASON_SWEEP_RUN_DIR:-${BASE_SWEEP_DIR}/${RUN_ID}}"
LOG_DIR="${RUN_DIR}/logs"
DATA_DIR="${RUN_DIR}/datasets"
ADAPTER_DIR="${RUN_DIR}/adapters"
EVAL_DIR="${RUN_DIR}/evals"
REPORT_DIR="${RUN_DIR}/reports"
DOCKER_OUT_DIR="${RUN_DIR}/docker"
STATUS_TSV="${RUN_DIR}/status.tsv"
REGISTRY_JSON="${RUN_DIR}/candidate_registry.json"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:-${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
TRAIN_JSON="${TRAIN_JSON:-${MEDREASON_DATA_ROOT}/train/medreason_train_selection.json}"
TRAIN_IMG_DIR="${TRAIN_IMG_DIR:-${MEDREASON_DATA_ROOT}/train/imgs}"

# Current best MCQ and open adapters from the June 2 full solution sweep.
CURRENT_MCQ_LORA="${CURRENT_MCQ_LORA:-${ROOT_DIR}/artifacts/finetune/mcq2048x250_lr1e4_r16_s43}"
CURRENT_OPEN_LORA="${CURRENT_OPEN_LORA:-${ROOT_DIR}/artifacts/finetune/open_ctx1_1024x200_lr1e4_r16_s61}"
CURRENT_THRESHOLD="${CURRENT_THRESHOLD:-0.239}"
MAX_NEW_TOKENS_MCQ="${MAX_NEW_TOKENS_MCQ:-192}"
MAX_NEW_TOKENS_OPEN="${MAX_NEW_TOKENS_OPEN:-384}"
EVAL_HOLDOUTS="holdout_60 holdout_220"

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${ADAPTER_DIR}" "${EVAL_DIR}" "${REPORT_DIR}" "${DOCKER_OUT_DIR}" "${BASE_SWEEP_DIR}"
touch "${STATUS_TSV}"
[ -s "${STATUS_TSV}" ] || printf "timestamp\tstage\tname\tstatus\tdetails\n" > "${STATUS_TSV}"
[ -s "${REGISTRY_JSON}" ] || printf '{"candidates":[]}\n' > "${REGISTRY_JSON}"

exec > >(tee -a "${LOG_DIR}/master.log") 2>&1

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

status() {
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$(date --iso-8601=seconds)" "$1" "$2" "$3" "${4:-}" \
    | tee -a "${STATUS_TSV}"
}

run_logged() {
  local stage="$1" name="$2"; shift 2
  local log="${LOG_DIR}/${stage}_${name}.log"
  status "${stage}" "${name}" START "$*"
  "$@" >"${log}" 2>&1; local rc=$?
  [ "${rc}" -eq 0 ] \
    && status "${stage}" "${name}" OK "log=${log}" \
    || status "${stage}" "${name}" FAIL "rc=${rc} log=${log}"
  return "${rc}"
}

record_candidate() {
  # name role mcq_lora open_lora threshold open_prompt_style open_max_new_tokens
  "${SYSTEM_PYTHON}" - "${REGISTRY_JSON}" "$@" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
name, role, mcq_lora, open_lora, threshold, open_prompt_style, open_max_tokens = sys.argv[2:]
payload = json.loads(path.read_text())
candidates = [c for c in payload.get("candidates", []) if c.get("name") != name]
candidates.append({
    "name": name, "role": role,
    "mcq_lora": mcq_lora, "open_lora": open_lora,
    "threshold": float(threshold),
    "open_prompt_style": open_prompt_style,
    "open_max_new_tokens": int(open_max_tokens),
})
payload["candidates"] = candidates
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

holdout_retrieval_bank() {
  case "$1" in
    holdout_25)  printf "%s/artifacts/retrieval_bank_holdout25_excluded.json"  "${ROOT_DIR}" ;;
    holdout_60)  printf "%s/artifacts/retrieval_bank_holdout60_excluded.json"  "${ROOT_DIR}" ;;
    holdout_220) printf "%s/artifacts/retrieval_bank_holdout220_excluded.json" "${ROOT_DIR}" ;;
    *)           printf "%s/artifacts/retrieval_bank.json"                     "${ROOT_DIR}" ;;
  esac
}

# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

build_combined_exclude() {
  local output="${RUN_DIR}/holdout_exclude_cases.json"
  [ -s "${output}" ] && { printf "%s" "${output}"; return 0; }
  "${SYSTEM_PYTHON}" - "${output}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
seen, cases = set(), []
for p in ["local_inputs/holdout_25/cases.json",
          "local_inputs/holdout_60/cases.json",
          "local_inputs/holdout_220/cases.json"]:
    for item in json.loads(Path(p).read_text()).get("cases", []):
        cid = str(item["case_id"])
        if cid not in seen:
            seen.add(cid); cases.append({"case_id": cid})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"cases": cases}, indent=2) + "\n")
print(out)
PY
}

build_dataset() {
  local name="$1"; shift
  local output="${DATA_DIR}/${name}.jsonl"
  if [ -s "${output}" ]; then
    status dataset "${name}" SKIP "exists=${output}"; return 0
  fi
  run_logged dataset "${name}" \
    "${PYTHON_BIN}" finetune/build_sft_dataset.py \
      --train-json "${TRAIN_JSON}" \
      --train-img-dir "${TRAIN_IMG_DIR}" \
      --exclude-case-files \
        local_inputs/holdout_25/cases.json \
        local_inputs/holdout_60/cases.json \
        local_inputs/holdout_220/cases.json \
      --output-jsonl "${output}" "$@"
}

# ---------------------------------------------------------------------------
# Adapter training
# ---------------------------------------------------------------------------

train_adapter() {
  # name dataset max_examples max_steps lr seed rank
  local name="$1" dataset="$2" max_examples="$3" max_steps="$4" lr="$5" seed="$6" rank="$7"
  local output_dir="${ADAPTER_DIR}/${name}"
  if [ -s "${output_dir}/adapter_model.safetensors" ]; then
    status train "${name}" SKIP "exists=${output_dir}"; return 0
  fi
  run_logged train "${name}" \
    "${PYTHON_BIN}" finetune/train_lora_qwen25vl.py \
      --model-path "${MODEL_PATH}" \
      --train-jsonl "${DATA_DIR}/${dataset}.jsonl" \
      --output-dir "${output_dir}" \
      --max-examples "${max_examples}" \
      --max-steps "${max_steps}" \
      --gradient-accumulation-steps 4 \
      --learning-rate "${lr}" \
      --lora-r "${rank}" \
      --lora-alpha "$((rank * 2))" \
      --lora-dropout 0.05 \
      --seed "${seed}" \
      --load-in-4bit
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

evaluate_candidate() {
  # name role mcq_lora open_lora threshold open_prompt_style open_max_new_tokens holdout...
  local name="$1" role="$2" mcq_lora="$3" open_lora="$4" threshold="$5"
  local open_prompt_style="$6" open_max_new_tokens="$7"
  shift 7
  record_candidate "${name}" "${role}" "${mcq_lora}" "${open_lora}" "${threshold}" \
    "${open_prompt_style}" "${open_max_new_tokens}"
  for holdout in "$@"; do
    local input_dir="${ROOT_DIR}/local_inputs/${holdout}"
    local retrieval_bank; retrieval_bank="$(holdout_retrieval_bank "${holdout}")"
    local out_dir="${EVAL_DIR}/${name}_${holdout}"
    mkdir -p "${out_dir}"
    if [ -s "${out_dir}/results.json" ]; then
      status eval "${name}_${holdout}" SKIP "exists=${out_dir}/results.json"
    else
      run_logged eval "${name}_${holdout}" \
        env PYTHONPATH="${ROOT_DIR}/docker/medreason" \
          MEDREASON_SYSTEM=strong_baseline \
          MEDREASON_INPUT_DIR="${input_dir}" \
          MEDREASON_OUTPUT_DIR="${out_dir}" \
          MEDREASON_OUTPUT_FILE="${out_dir}/results.json" \
          MEDREASON_MODEL_PATH="${MODEL_PATH}" \
          MEDREASON_LORA_PATH="${mcq_lora}" \
          MEDREASON_OPEN_LORA_PATH="${open_lora}" \
          MEDREASON_RETRIEVAL_BANK="${retrieval_bank}" \
          MEDREASON_MCQ_POLICY=hybrid_low_confidence \
          MEDREASON_OPTION_SCORE_MODE=sum5 \
          MEDREASON_VLM_OVERRIDE_CONFIDENCE_THRESHOLD="${threshold}" \
          MEDREASON_TOP_K_EXAMPLES=1 \
          MEDREASON_OPEN_TOP_K_EXAMPLES=1 \
          MEDREASON_OPEN_PROMPT_STYLE="${open_prompt_style}" \
          MEDREASON_MAX_NEW_TOKENS="${open_max_new_tokens}" \
          MEDREASON_LOG_EVERY=25 \
          "${PYTHON_BIN}" docker/medreason/process.py
    fi
    if [ -s "${out_dir}/results.json" ]; then
      run_logged validate "${name}_${holdout}" \
        "${SYSTEM_PYTHON}" scripts/validate_predictions.py \
          --cases-json "${input_dir}/cases.json" \
          --results-json "${out_dir}/results.json" || true
      run_logged score "${name}_${holdout}" \
        "${SYSTEM_PYTHON}" scripts/score_predictions.py \
          --ground-truth "${input_dir}/ground_truth.json" \
          --results-json "${out_dir}/results.json" || true
      run_logged open_analysis "${name}_${holdout}" \
        "${SYSTEM_PYTHON}" scripts/analyze_open_predictions.py \
          --ground-truth "${input_dir}/ground_truth.json" \
          --cases-json "${input_dir}/cases.json" \
          --results-json "${out_dir}/results.json" \
          --retrieval-bank "${retrieval_bank}" \
          --output-json "${out_dir}/open_analysis.json" || true
      run_logged calibration "${name}_${holdout}" \
        "${SYSTEM_PYTHON}" scripts/calibrate_mcq_threshold.py \
          --ground-truth "${input_dir}/ground_truth.json" \
          --results-json "${out_dir}/results.json" \
          --output-json "${out_dir}/calibration.json" \
          --start 0.15 --stop 0.35 --step 0.001 || true
    fi
  done
}

summarize() {
  run_logged report summarize \
    "${SYSTEM_PYTHON}" scripts/summarize_solution_sweep.py \
      --run-dir "${RUN_DIR}" \
      --repo-root "${ROOT_DIR}" || true
}

# ---------------------------------------------------------------------------
# Docker packaging
# ---------------------------------------------------------------------------

prepare_contract_input() {
  local source_name="$1" output_dir="$2" image_root="$3"
  rm -rf "${output_dir}"; mkdir -p "${output_dir}/imgs"
  "${SYSTEM_PYTHON}" - "${ROOT_DIR}/local_inputs/${source_name}/cases.json" \
    "${output_dir}" "${image_root}" <<'PY'
import json, shutil, sys
from pathlib import Path
cases = json.loads(Path(sys.argv[1]).read_text()).get("cases", [])
out_dir, img_root = Path(sys.argv[2]), Path(sys.argv[3])
for c in cases:
    name = Path(c["image_path"]).name
    shutil.copy2(img_root / name, out_dir / "imgs" / name)
(out_dir / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
print(f"prepared {len(cases)} cases at {out_dir}")
PY
}

package_winner() {
  local winner_json="${RUN_DIR}/winner.json"
  [ -s "${winner_json}" ] || { status package winner SKIP "missing=${winner_json}"; return 0; }

  local fallback
  fallback="$("${SYSTEM_PYTHON}" - "${winner_json}" <<'PY'
import json, sys
print(str(json.load(open(sys.argv[1])).get("fallback_to_current_final", True)).lower())
PY
)"
  if [ "${fallback}" = "true" ]; then
    status package winner SKIP "no candidate beat acceptance rules; falling back to current submission"
    return 0
  fi

  local mcq_lora open_lora threshold open_prompt_style open_max_tokens
  read -r mcq_lora open_lora threshold open_prompt_style open_max_tokens < <(
    "${SYSTEM_PYTHON}" - "${winner_json}" <<'PY'
import json, sys
m = (json.load(open(sys.argv[1])).get("winner") or {}).get("meta") or {}
print(
    m.get("mcq_lora", ""), m.get("open_lora", ""),
    m.get("threshold", 0.239), m.get("open_prompt_style", "baseline"),
    m.get("open_max_new_tokens", 192),
)
PY
  )

  [ -d "${mcq_lora}" ] || { status package winner FAIL "mcq_lora not found: ${mcq_lora}"; return 1; }
  [ -d "${open_lora}" ] || { status package winner FAIL "open_lora not found: ${open_lora}"; return 1; }

  run_logged package stage_loras bash -c "
    rm -rf docker/medreason/lora/sweep_winner_mcq docker/medreason/lora/sweep_winner_open
    mkdir -p docker/medreason/lora
    cp -a '${mcq_lora}' docker/medreason/lora/sweep_winner_mcq
    cp -a '${open_lora}' docker/medreason/lora/sweep_winner_open
  "

  local image_tag="medreason-submission:open-quality-${RUN_ID}"
  local tar_path="${DOCKER_OUT_DIR}/medreason-submission-open-quality-winner.tar"

  run_logged package build_winner bash -c "
    cd docker/medreason && ./build_large.sh '${image_tag}' '${tar_path}' \
      --build-arg INSTALL_VLM_DEPS=1 \
      --build-arg DEFAULT_LORA_PATH=/opt/app/lora/sweep_winner_mcq \
      --build-arg DEFAULT_OPEN_LORA_PATH=/opt/app/lora/sweep_winner_open \
      --build-arg DEFAULT_VLM_OVERRIDE_CONFIDENCE_THRESHOLD='${threshold}' \
      --build-arg DEFAULT_MAX_NEW_TOKENS='${open_max_tokens}' \
      --build-arg DEFAULT_OPEN_PROMPT_STYLE='${open_prompt_style}'
  "

  if [ "${MEDREASON_LOAD_WINNER_IMAGE:-1}" = "1" ]; then
    run_logged package docker_load docker load -i "${tar_path}"

    local smoke_in="${RUN_DIR}/contract_inputs/smoke"
    local smoke_out="${RUN_DIR}/contract_outputs/smoke"
    local hold25_in="${RUN_DIR}/contract_inputs/holdout25"
    local hold25_out="${RUN_DIR}/contract_outputs/holdout25"
    prepare_contract_input validation_2 "${smoke_in}" "${MEDREASON_DATA_ROOT}/validation/imgs"
    prepare_contract_input holdout_25   "${hold25_in}" "${TRAIN_IMG_DIR}"
    mkdir -p "${smoke_out}" "${hold25_out}"

    run_logged package docker_smoke \
      docker run --rm --gpus all \
        -v "${smoke_in}:/input:ro" -v "${smoke_out}:/output" "${image_tag}"
    run_logged package docker_holdout25 \
      docker run --rm --gpus all \
        -v "${hold25_in}:/input:ro" -v "${hold25_out}:/output" "${image_tag}"
    run_logged package validate_smoke \
      "${SYSTEM_PYTHON}" \
        external/MedReason-Challenge-Docker/MedReason-Evaluation/validate_submission.py \
        "${smoke_out}/results.json" --cases-json "${smoke_in}/cases.json"
    run_logged package validate_holdout25 \
      "${SYSTEM_PYTHON}" \
        external/MedReason-Challenge-Docker/MedReason-Evaluation/validate_submission.py \
        "${hold25_out}/results.json" --cases-json "${hold25_in}/cases.json"
  fi

  run_logged package pigz pigz -k -f "${tar_path}"
  run_logged package gzip_test gzip -t "${tar_path}.gz"
  status package winner OK "tar=${tar_path}.gz  prompt=${open_prompt_style}  max_tokens=${open_max_tokens}"
}

# ===========================================================================
# SWEEP BODY
# ===========================================================================

printf "run_id=%s\nrun_dir=%s\nstarted_at=%s\n" "${RUN_ID}" "${RUN_DIR}" "$(date --iso-8601=seconds)"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
git status --short --branch > "${RUN_DIR}/git_status.txt" 2>/dev/null || true
nvidia-smi > "${RUN_DIR}/nvidia_smi_start.txt" 2>/dev/null || true
df -h / "${BASE_SWEEP_DIR}" > "${RUN_DIR}/disk_start.txt" 2>/dev/null || true
status init sweep OK "run_dir=${RUN_DIR}"

# ---------------------------------------------------------------------------
# Phase 0: baseline reference (no new training)
# ---------------------------------------------------------------------------

status phase 0 START "baseline reference — current best MCQ + current best open"

# Baseline: current winner as-is (baseline prompt, 192 tokens)
evaluate_candidate \
  baseline_current combined \
  "${CURRENT_MCQ_LORA}" "${CURRENT_OPEN_LORA}" "${CURRENT_THRESHOLD}" \
  baseline "${MAX_NEW_TOKENS_MCQ}" \
  ${EVAL_HOLDOUTS}

# Inference-only variant: match train prompt (evidence_first) at inference
evaluate_candidate \
  infer_ef combined \
  "${CURRENT_MCQ_LORA}" "${CURRENT_OPEN_LORA}" "${CURRENT_THRESHOLD}" \
  evidence_first "${MAX_NEW_TOKENS_MCQ}" \
  ${EVAL_HOLDOUTS}

# Inference-only variant: modality_guard at inference + higher token budget
evaluate_candidate \
  infer_mg_384 combined \
  "${CURRENT_MCQ_LORA}" "${CURRENT_OPEN_LORA}" "${CURRENT_THRESHOLD}" \
  modality_guard "${MAX_NEW_TOKENS_OPEN}" \
  ${EVAL_HOLDOUTS}

# Inference-only variant: evidence_first + higher token budget
evaluate_candidate \
  infer_ef_384 combined \
  "${CURRENT_MCQ_LORA}" "${CURRENT_OPEN_LORA}" "${CURRENT_THRESHOLD}" \
  evidence_first "${MAX_NEW_TOKENS_OPEN}" \
  ${EVAL_HOLDOUTS}

summarize
status phase 0 OK "inference-only variants done"

# ---------------------------------------------------------------------------
# Phase 1: Build retrieval bank and SFT datasets
# ---------------------------------------------------------------------------

status phase 1 START "datasets"

EXCLUDE_JSON="$(build_combined_exclude)"
RETRIEVAL_BANK="${RUN_DIR}/retrieval_bank_all_holdouts_excluded.json"
if [ -s "${RETRIEVAL_BANK}" ]; then
  status prep retrieval_bank SKIP "exists=${RETRIEVAL_BANK}"
else
  run_logged prep retrieval_bank \
    "${SYSTEM_PYTHON}" scripts/build_retrieval_bank.py \
      --train-json "${TRAIN_JSON}" \
      --exclude-case-ids "${EXCLUDE_JSON}" \
      --output "${RETRIEVAL_BANK}"
fi

# evidence_first prompt + visual_answer response (uses real visual_description as reasoning trace target)
build_dataset open_ef_visual_1024_s71 \
  --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 \
  --shuffle --seed 71 --open-only --max-examples 1024 \
  --prompt-style evidence_first --response-style visual_answer

# modality_guard prompt + visual_answer response
build_dataset open_mg_visual_1024_s72 \
  --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 \
  --shuffle --seed 72 --open-only --max-examples 1024 \
  --prompt-style modality_guard --response-style visual_answer

# modality_guard + visual_answer + more data
build_dataset open_mg_visual_2048_s73 \
  --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 \
  --shuffle --seed 73 --open-only --max-examples 2048 \
  --prompt-style modality_guard --response-style visual_answer

# modality_guard + visual_answer + no retrieval context (clean modality signal)
build_dataset open_mg_noctx_1024_s74 \
  --top-k-examples 0 \
  --shuffle --seed 74 --open-only --max-examples 1024 \
  --prompt-style modality_guard --response-style visual_answer

status phase 1 OK "datasets done"

# ---------------------------------------------------------------------------
# Phase 2: Train open adapters
# ---------------------------------------------------------------------------

status phase 2 START "training open adapters"

#                  name                           dataset               ex   steps  lr    seed rank
train_adapter open_ef_visual_1024x200_lr1e4_r16_s71  open_ef_visual_1024_s71  1024  200  1e-4  71  16
train_adapter open_mg_visual_1024x200_lr1e4_r16_s72  open_mg_visual_1024_s72  1024  200  1e-4  72  16
train_adapter open_mg_visual_2048x250_lr1e4_r16_s73  open_mg_visual_2048_s73  2048  250  1e-4  73  16
train_adapter open_mg_noctx_1024x200_lr1e4_r16_s74   open_mg_noctx_1024_s74   1024  200  1e-4  74  16

status phase 2 OK "training done"

# ---------------------------------------------------------------------------
# Phase 3: Evaluate trained adapters (h220 + h60 for acceptance criteria)
# ---------------------------------------------------------------------------

status phase 3 START "evaluating trained adapters"

for adapter_spec in \
  "open_ef_visual_1024x200_lr1e4_r16_s71:evidence_first" \
  "open_mg_visual_1024x200_lr1e4_r16_s72:modality_guard" \
  "open_mg_visual_2048x250_lr1e4_r16_s73:modality_guard" \
  "open_mg_noctx_1024x200_lr1e4_r16_s74:modality_guard" \
; do
  adapter_name="${adapter_spec%%:*}"
  prompt_style="${adapter_spec##*:}"
  adapter_path="${ADAPTER_DIR}/${adapter_name}"
  if [ ! -s "${adapter_path}/adapter_model.safetensors" ]; then
    status eval "${adapter_name}" SKIP "adapter missing"
    continue
  fi
  # Primary eval: matching prompt style, extended token budget
  evaluate_candidate \
    "bestmcq__${adapter_name}" combined \
    "${CURRENT_MCQ_LORA}" "${adapter_path}" "${CURRENT_THRESHOLD}" \
    "${prompt_style}" "${MAX_NEW_TOKENS_OPEN}" \
    ${EVAL_HOLDOUTS}
  # Secondary eval: evidence_first prompt for cross-comparison on same adapter
  if [ "${prompt_style}" = "modality_guard" ]; then
    evaluate_candidate \
      "bestmcq__${adapter_name}__ef" combined \
      "${CURRENT_MCQ_LORA}" "${adapter_path}" "${CURRENT_THRESHOLD}" \
      evidence_first "${MAX_NEW_TOKENS_OPEN}" \
      holdout_220
  fi
  summarize
done

status phase 3 OK "evaluations done"

# ---------------------------------------------------------------------------
# Phase 4: Select winner and package Docker
# ---------------------------------------------------------------------------

status phase 4 START "selecting winner and packaging Docker"
summarize
package_winner || true

nvidia-smi > "${RUN_DIR}/nvidia_smi_end.txt" 2>/dev/null || true
df -h / "${BASE_SWEEP_DIR}" > "${RUN_DIR}/disk_end.txt" 2>/dev/null || true
status done sweep OK "finished_at=$(date --iso-8601=seconds)"
printf "finished_at=%s\nrun_dir=%s\n" "$(date --iso-8601=seconds)" "${RUN_DIR}"
