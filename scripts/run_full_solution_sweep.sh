#!/usr/bin/env bash
set -uo pipefail
trap 'rc=$?; printf "[MedReason sweep exit] rc=%s line=%s cmd=%q\n" "$rc" "$LINENO" "$BASH_COMMAND" >&2' EXIT
if [ "${MEDREASON_SWEEP_DEBUG:-0}" = "1" ]; then
  set -x
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
MEDREASON_DATA_ROOT="${MEDREASON_DATA_ROOT:-${ROOT_DIR}/data}"

RUN_ID="${MEDREASON_SWEEP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
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
LATEST_LINK="${BASE_SWEEP_DIR}/latest"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:-${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
TRAIN_JSON="${TRAIN_JSON:-${MEDREASON_DATA_ROOT}/train/medreason_train_selection.json}"
TRAIN_IMG_DIR="${TRAIN_IMG_DIR:-${MEDREASON_DATA_ROOT}/train/imgs}"
CURRENT_MCQ_LORA="${CURRENT_MCQ_LORA:-${ROOT_DIR}/artifacts/finetune/qwen25vl3b-lora-mcq1024x150-s41}"
CURRENT_OPEN_LORA="${CURRENT_OPEN_LORA:-${ROOT_DIR}/artifacts/finetune/open_ctx1_1024x150_lr1e4_s57}"
CURRENT_THRESHOLD="${CURRENT_THRESHOLD:-0.239}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
PROMOTE_HOLDOUT60_MCQ="${PROMOTE_HOLDOUT60_MCQ:-40}"
EVAL_HOLDOUTS="${EVAL_HOLDOUTS:-holdout_25 holdout_60 holdout_220}"

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${ADAPTER_DIR}" "${EVAL_DIR}" "${REPORT_DIR}" "${DOCKER_OUT_DIR}" "${BASE_SWEEP_DIR}"
rm -f "${LATEST_LINK}"
ln -s "$(realpath "${RUN_DIR}")" "${LATEST_LINK}"
touch "${STATUS_TSV}"
if [ ! -s "${STATUS_TSV}" ]; then
  printf "timestamp\tstage\tname\tstatus\tdetails\n" > "${STATUS_TSV}"
fi
if [ ! -s "${REGISTRY_JSON}" ]; then
  printf '{"candidates":[]}\n' > "${REGISTRY_JSON}"
fi

if [ "${MEDREASON_SWEEP_INTERNAL_TEE:-0}" = "1" ]; then
  exec > >(tee -a "${LOG_DIR}/master.log") 2>&1
fi

status() {
  local stage="$1"
  local name="$2"
  local state="$3"
  local details="${4:-}"
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date --iso-8601=seconds)" "${stage}" "${name}" "${state}" "${details}" \
    | tee -a "${STATUS_TSV}"
}

run_logged() {
  local stage="$1"
  local name="$2"
  shift 2
  local log="${LOG_DIR}/${stage}_${name}.log"
  status "${stage}" "${name}" START "$*"
  "$@" >"${log}" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    status "${stage}" "${name}" OK "log=${log}"
  else
    status "${stage}" "${name}" FAIL "rc=${rc} log=${log}"
  fi
  return "${rc}"
}

record_candidate() {
  local name="$1"
  local role="$2"
  local mcq_lora="$3"
  local open_lora="$4"
  local threshold="$5"
  "${SYSTEM_PYTHON}" - "${REGISTRY_JSON}" "${name}" "${role}" "${mcq_lora}" "${open_lora}" "${threshold}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
name, role, mcq_lora, open_lora, threshold = sys.argv[2:]
payload = json.loads(path.read_text(encoding="utf-8"))
candidates = [item for item in payload.get("candidates", []) if item.get("name") != name]
candidates.append(
    {
        "name": name,
        "role": role,
        "mcq_lora": mcq_lora,
        "open_lora": open_lora,
        "threshold": float(threshold),
    }
)
payload["candidates"] = candidates
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

holdout_retrieval_bank() {
  case "$1" in
    holdout_25) printf "%s/artifacts/retrieval_bank_holdout25_excluded.json" "${ROOT_DIR}" ;;
    holdout_60) printf "%s/artifacts/retrieval_bank_holdout60_excluded.json" "${ROOT_DIR}" ;;
    holdout_220) printf "%s/artifacts/retrieval_bank_holdout220_excluded.json" "${ROOT_DIR}" ;;
    *) printf "%s/artifacts/retrieval_bank.json" "${ROOT_DIR}" ;;
  esac
}

mcq_correct_for_results() {
  local holdout="$1"
  local results="$2"
  "${SYSTEM_PYTHON}" - "${ROOT_DIR}/local_inputs/${holdout}/ground_truth.json" "${results}" <<'PY'
import json
import sys
from pathlib import Path

truth_payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results_payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
truth = {
    str(item["case_id"]): str(item.get("answer", "")).strip().upper()[:1]
    for item in truth_payload.get("answers", truth_payload)
    if str(item.get("task_type", "")).lower() == "mcq"
}
correct = 0
total = 0
vlm_correct = 0
for item in results_payload.get("answers", results_payload):
    if str(item.get("task_type", "")).lower() != "mcq":
        continue
    cid = str(item.get("case_id"))
    if cid not in truth:
        continue
    total += 1
    correct += str(item.get("answer", "")).strip().upper()[:1] == truth[cid]
    metadata = item.get("metadata") or {}
    vlm_correct += str(metadata.get("vlm_answer", "")).strip().upper()[:1] == truth[cid]
print(f"{correct}\t{total}\t{vlm_correct}")
PY
}

build_combined_exclude() {
  local output="${RUN_DIR}/holdout_exclude_cases.json"
  if [ -s "${output}" ]; then
    printf "%s" "${output}"
    return 0
  fi
  "${SYSTEM_PYTHON}" - "${output}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
paths = [
    Path("local_inputs/holdout_25/cases.json"),
    Path("local_inputs/holdout_60/cases.json"),
    Path("local_inputs/holdout_220/cases.json"),
]
seen = set()
cases = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("cases", payload):
        cid = str(item["case_id"])
        if cid not in seen:
            seen.add(cid)
            cases.append({"case_id": cid})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
print(out)
PY
}

build_dataset() {
  local name="$1"
  shift
  local output="${DATA_DIR}/${name}.jsonl"
  if [ -s "${output}" ]; then
    status dataset "${name}" SKIP "exists=${output}"
    return 0
  fi
  run_logged dataset "${name}" \
    "${PYTHON_BIN}" finetune/build_sft_dataset.py \
      --train-json "${TRAIN_JSON}" \
      --train-img-dir "${TRAIN_IMG_DIR}" \
      --exclude-case-files local_inputs/holdout_25/cases.json local_inputs/holdout_60/cases.json local_inputs/holdout_220/cases.json \
      --output-jsonl "${output}" \
      "$@"
}

train_adapter() {
  local name="$1"
  local dataset="$2"
  local max_examples="$3"
  local max_steps="$4"
  local lr="$5"
  local seed="$6"
  local rank="$7"
  local output_dir="${ADAPTER_DIR}/${name}"
  local alpha=$((rank * 2))
  if [ -s "${output_dir}/adapter_model.safetensors" ]; then
    status train "${name}" SKIP "exists=${output_dir}"
    return 0
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
      --lora-alpha "${alpha}" \
      --lora-dropout 0.05 \
      --seed "${seed}" \
      --load-in-4bit
}

evaluate_candidate() {
  local name="$1"
  local role="$2"
  local mcq_lora="$3"
  local open_lora="$4"
  local threshold="$5"
  shift 5
  local holdouts=("$@")
  record_candidate "${name}" "${role}" "${mcq_lora}" "${open_lora}" "${threshold}"
  for holdout in "${holdouts[@]}"; do
    local input_dir="${ROOT_DIR}/local_inputs/${holdout}"
    local retrieval_bank
    retrieval_bank="$(holdout_retrieval_bank "${holdout}")"
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
          MEDREASON_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
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

select_best_mcq_lora() {
  "${SYSTEM_PYTHON}" - "${RUN_DIR}/summary.json" "${CURRENT_MCQ_LORA}" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
fallback = sys.argv[2]
if not summary.exists():
    print(fallback)
    raise SystemExit
payload = json.loads(summary.read_text(encoding="utf-8"))
rows = []
for item in payload.get("candidates", []):
    meta = item.get("meta") or {}
    if meta.get("role") != "mcq":
        continue
    h220 = (item.get("holdouts") or {}).get("holdout_220") or {}
    if not h220:
        continue
    rows.append(
        (
            h220.get("cv_correct") or -1,
            h220.get("mcq_correct") or -1,
            item["name"],
            meta.get("mcq_lora") or fallback,
        )
    )
rows.sort(reverse=True)
print(rows[0][3] if rows else fallback)
PY
}

select_top_open_loras() {
  "${SYSTEM_PYTHON}" - "${RUN_DIR}/summary.json" "${CURRENT_OPEN_LORA}" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
fallback = sys.argv[2]
if not summary.exists():
    print(fallback)
    raise SystemExit
payload = json.loads(summary.read_text(encoding="utf-8"))
rows = []
for item in payload.get("candidates", []):
    meta = item.get("meta") or {}
    if meta.get("role") != "open":
        continue
    h220 = (item.get("holdouts") or {}).get("holdout_220") or {}
    if not h220:
        continue
    if (h220.get("mcq_correct") or 0) < 164:
        continue
    rows.append(
        (
            h220.get("open_answer_token_f1") or 0.0,
            h220.get("open_trace_token_f1") or 0.0,
            item["name"],
            meta.get("open_lora") or fallback,
        )
    )
rows.sort(reverse=True)
selected = [row[3] for row in rows[:2]]
if fallback not in selected:
    selected.append(fallback)
print("\n".join(selected[:3]))
PY
}

prepare_contract_input() {
  local source_name="$1"
  local output_dir="$2"
  local image_root="$3"
  rm -rf "${output_dir}"
  mkdir -p "${output_dir}/imgs"
  "${SYSTEM_PYTHON}" - "${ROOT_DIR}/local_inputs/${source_name}/cases.json" "${output_dir}" "${image_root}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

cases_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
image_root = Path(sys.argv[3])
payload = json.loads(cases_path.read_text(encoding="utf-8"))
cases = payload.get("cases", payload)
for case in cases:
    name = Path(case["image_path"]).name
    shutil.copy2(image_root / name, out_dir / "imgs" / name)
(out_dir / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
print(f"prepared {len(cases)} cases at {out_dir}")
PY
}

package_winner_if_any() {
  local winner_json="${RUN_DIR}/winner.json"
  if [ ! -s "${winner_json}" ]; then
    status package winner SKIP "missing=${winner_json}"
    return 0
  fi
  local fallback
  fallback="$("${SYSTEM_PYTHON}" - "${winner_json}" <<'PY'
import json, sys
print(str(json.load(open(sys.argv[1])).get("fallback_to_current_final", True)).lower())
PY
)"
  if [ "${fallback}" = "true" ]; then
    status package winner SKIP "summarizer selected current final fallback"
    return 0
  fi

  local mcq_lora open_lora threshold
  read -r mcq_lora open_lora threshold < <("${SYSTEM_PYTHON}" - "${winner_json}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
meta = ((payload.get("winner") or {}).get("meta") or {})
print(meta.get("mcq_lora", ""), meta.get("open_lora", ""), meta.get("threshold", 0.239))
PY
)
  if [ ! -d "${mcq_lora}" ] || [ ! -d "${open_lora}" ]; then
    status package winner FAIL "missing winner lora paths mcq=${mcq_lora} open=${open_lora}"
    return 1
  fi

  run_logged package stage_loras \
    bash -c "rm -rf docker/medreason/lora/sweep_winner_mcq docker/medreason/lora/sweep_winner_open && mkdir -p docker/medreason/lora && cp -a '${mcq_lora}' docker/medreason/lora/sweep_winner_mcq && cp -a '${open_lora}' docker/medreason/lora/sweep_winner_open"

  local image_tag="medreason-submission:sweep-winner-${RUN_ID}"
  local tar_path="${DOCKER_OUT_DIR}/medreason-submission-sweep-winner.tar"
  run_logged package build_winner \
    bash -c "cd docker/medreason && ./build_large.sh '${image_tag}' '${tar_path}' --build-arg INSTALL_VLM_DEPS=1 --build-arg DEFAULT_LORA_PATH=/opt/app/lora/sweep_winner_mcq --build-arg DEFAULT_OPEN_LORA_PATH=/opt/app/lora/sweep_winner_open --build-arg DEFAULT_VLM_OVERRIDE_CONFIDENCE_THRESHOLD='${threshold}' --build-arg DEFAULT_MAX_NEW_TOKENS='${MAX_NEW_TOKENS}'"

  if [ "${MEDREASON_LOAD_WINNER_IMAGE:-1}" = "1" ]; then
    run_logged package docker_load docker load -i "${tar_path}"
    local smoke_in="${RUN_DIR}/contract_inputs/smoke"
    local smoke_out="${RUN_DIR}/contract_outputs/smoke"
    local hold25_in="${RUN_DIR}/contract_inputs/holdout25"
    local hold25_out="${RUN_DIR}/contract_outputs/holdout25"
    prepare_contract_input validation_2 "${smoke_in}" "${MEDREASON_DATA_ROOT}/validation/imgs"
    prepare_contract_input holdout_25 "${hold25_in}" "${TRAIN_IMG_DIR}"
    mkdir -p "${smoke_out}" "${hold25_out}"
    run_logged package docker_smoke \
      docker run --rm --gpus all -v "${smoke_in}:/input:ro" -v "${smoke_out}:/output" "${image_tag}"
    run_logged package docker_holdout25 \
      docker run --rm --gpus all -v "${hold25_in}:/input:ro" -v "${hold25_out}:/output" "${image_tag}"
    run_logged package validate_smoke \
      "${SYSTEM_PYTHON}" external/MedReason-Challenge-Docker/MedReason-Evaluation/validate_submission.py "${smoke_out}/results.json" --cases-json "${smoke_in}/cases.json"
    run_logged package validate_holdout25 \
      "${SYSTEM_PYTHON}" external/MedReason-Challenge-Docker/MedReason-Evaluation/validate_submission.py "${hold25_out}/results.json" --cases-json "${hold25_in}/cases.json"
  fi

  run_logged package pigz pigz -k -f "${tar_path}"
  run_logged package gzip_test gzip -t "${tar_path}.gz"
}

printf "run_id=%s\nrun_dir=%s\nstarted_at=%s\n" "${RUN_ID}" "${RUN_DIR}" "$(date --iso-8601=seconds)"
git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
git status --short --branch > "${RUN_DIR}/git_status.txt" 2>/dev/null || true
nvidia-smi > "${RUN_DIR}/nvidia_smi_start.txt" 2>/dev/null || true
df -h / "${BASE_SWEEP_DIR}" > "${RUN_DIR}/disk_start.txt" 2>/dev/null || true
status init sweep OK "run_dir=${RUN_DIR}"

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

status baseline current START "evaluating current final candidate"
evaluate_candidate baseline_current baseline "${CURRENT_MCQ_LORA}" "${CURRENT_OPEN_LORA}" "${CURRENT_THRESHOLD}" ${EVAL_HOLDOUTS}
summarize

for seed in 42 43 44 45; do
  build_dataset "mcq_1024_s${seed}" --top-k-examples 0 --shuffle --seed "${seed}" --mcq-only --max-examples 1024
done
for seed in 42 43; do
  build_dataset "mcq_2048_s${seed}" --top-k-examples 0 --shuffle --seed "${seed}" --mcq-only --max-examples 2048
done

declare -a MCQ_JOBS=(
  "mcq1024x150_lr2e4_r16_s42 mcq_1024_s42 1024 150 2e-4 42 16"
  "mcq1024x150_lr2e4_r16_s43 mcq_1024_s43 1024 150 2e-4 43 16"
  "mcq1024x150_lr2e4_r16_s44 mcq_1024_s44 1024 150 2e-4 44 16"
  "mcq1024x150_lr2e4_r16_s45 mcq_1024_s45 1024 150 2e-4 45 16"
  "mcq1024x200_lr1e4_r16_s42 mcq_1024_s42 1024 200 1e-4 42 16"
  "mcq1024x200_lr1e4_r16_s43 mcq_1024_s43 1024 200 1e-4 43 16"
  "mcq1024x150_lr2e4_r32_s42 mcq_1024_s42 1024 150 2e-4 42 32"
  "mcq1024x150_lr2e4_r32_s43 mcq_1024_s43 1024 150 2e-4 43 32"
  "mcq2048x250_lr1e4_r16_s42 mcq_2048_s42 2048 250 1e-4 42 16"
  "mcq2048x250_lr1e4_r16_s43 mcq_2048_s43 2048 250 1e-4 43 16"
)

for job in "${MCQ_JOBS[@]}"; do
  read -r name dataset max_examples max_steps lr seed rank <<<"${job}"
  train_adapter "${name}" "${dataset}" "${max_examples}" "${max_steps}" "${lr}" "${seed}" "${rank}" || true
  adapter_path="${ADAPTER_DIR}/${name}"
  if [ -s "${adapter_path}/adapter_model.safetensors" ]; then
    evaluate_candidate "${name}" mcq "${adapter_path}" "${CURRENT_OPEN_LORA}" "${CURRENT_THRESHOLD}" holdout_60
    read -r h60_correct h60_total h60_vlm_correct < <(mcq_correct_for_results holdout_60 "${EVAL_DIR}/${name}_holdout_60/results.json")
    status promote "${name}" INFO "holdout60=${h60_correct}/${h60_total} vlm=${h60_vlm_correct}/${h60_total}"
    if [ "${h60_correct}" -ge "${PROMOTE_HOLDOUT60_MCQ}" ] || [ "${h60_vlm_correct}" -ge "${PROMOTE_HOLDOUT60_MCQ}" ]; then
      evaluate_candidate "${name}" mcq "${adapter_path}" "${CURRENT_OPEN_LORA}" "${CURRENT_THRESHOLD}" holdout_25 holdout_220
    else
      status promote "${name}" SKIP "below threshold ${PROMOTE_HOLDOUT60_MCQ}/50"
    fi
    summarize
  fi
done

BEST_MCQ_LORA="$(select_best_mcq_lora)"
status select best_mcq OK "${BEST_MCQ_LORA}"

build_dataset open_ctx1_1024_s61 --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 --shuffle --seed 61 --open-only --max-examples 1024 --prompt-style evidence_first --response-style concise_open
build_dataset open_ctx1_2048_s62 --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 --shuffle --seed 62 --open-only --max-examples 2048 --prompt-style evidence_first --response-style concise_open
build_dataset open_ctx2_1024_s63 --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 2 --shuffle --seed 63 --open-only --max-examples 1024 --prompt-style evidence_first --response-style concise_open
build_dataset open_noctx_2048_s64 --top-k-examples 0 --shuffle --seed 64 --open-only --max-examples 2048 --prompt-style evidence_first --response-style concise_open
build_dataset mixed_openheavy_ctx1_2048_s65 --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 --shuffle --seed 65 --max-open 1536 --max-mcq 512 --max-examples 2048 --prompt-style evidence_first --response-style concise_open

declare -a OPEN_JOBS=(
  "open_ctx1_1024x200_lr1e4_r16_s61 open_ctx1_1024_s61 1024 200 1e-4 61 16"
  "open_ctx1_2048x250_lr1e4_r16_s62 open_ctx1_2048_s62 2048 250 1e-4 62 16"
  "open_ctx1_2048x400_lr5e5_r16_s62 open_ctx1_2048_s62 2048 400 5e-5 62 16"
  "open_ctx2_1024x200_lr1e4_r16_s63 open_ctx2_1024_s63 1024 200 1e-4 63 16"
  "open_noctx_2048x250_lr1e4_r16_s64 open_noctx_2048_s64 2048 250 1e-4 64 16"
  "mixed_openheavy_ctx1_2048x250_lr1e4_r16_s65 mixed_openheavy_ctx1_2048_s65 2048 250 1e-4 65 16"
)

for job in "${OPEN_JOBS[@]}"; do
  read -r name dataset max_examples max_steps lr seed rank <<<"${job}"
  train_adapter "${name}" "${dataset}" "${max_examples}" "${max_steps}" "${lr}" "${seed}" "${rank}" || true
  adapter_path="${ADAPTER_DIR}/${name}"
  if [ -s "${adapter_path}/adapter_model.safetensors" ]; then
    evaluate_candidate "currentmcq__${name}" open "${CURRENT_MCQ_LORA}" "${adapter_path}" "${CURRENT_THRESHOLD}" ${EVAL_HOLDOUTS}
    summarize
  fi
done

mapfile -t TOP_OPEN_LORAS < <(select_top_open_loras)
idx=0
for open_lora in "${TOP_OPEN_LORAS[@]}"; do
  idx=$((idx + 1))
  evaluate_candidate "bestmcq__open${idx}" combined "${BEST_MCQ_LORA}" "${open_lora}" "${CURRENT_THRESHOLD}" ${EVAL_HOLDOUTS}
  summarize
done

summarize
package_winner_if_any || true

nvidia-smi > "${RUN_DIR}/nvidia_smi_end.txt" 2>/dev/null || true
df -h / "${BASE_SWEEP_DIR}" > "${RUN_DIR}/disk_end.txt" 2>/dev/null || true
status done sweep OK "finished_at=$(date --iso-8601=seconds)"
printf "finished_at=%s\n" "$(date --iso-8601=seconds)"
