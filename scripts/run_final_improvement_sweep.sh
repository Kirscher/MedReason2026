#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
MEDREASON_DATA_ROOT="${MEDREASON_DATA_ROOT:-${ROOT_DIR}/data}"
RUN_DIR="${MEDREASON_FINAL_SWEEP_DIR:-${ROOT_DIR}/artifacts/sweeps/final_improvements}"
DATA_DIR="${RUN_DIR}/datasets"
ADAPTER_DIR="${RUN_DIR}/adapters"
EVAL_DIR="${RUN_DIR}/evals"
LOG_DIR="${RUN_DIR}/logs"
REPORT_DIR="${RUN_DIR}/reports"
STATUS_FILE="${RUN_DIR}/status.tsv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:-${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
OPEN_LORA="${OPEN_LORA:-${ROOT_DIR}/docker/medreason/lora/sweep_winner_open}"
EXISTING_TRAIN_PID="${MEDREASON_EXISTING_TRAIN_PID:-}"
mkdir -p "${DATA_DIR}" "${ADAPTER_DIR}" "${EVAL_DIR}" "${LOG_DIR}" "${REPORT_DIR}"
if [ ! -s "${STATUS_FILE}" ]; then
  printf 'timestamp\tstage\tname\tstatus\tdetails\n' > "${STATUS_FILE}"
fi

status() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" "$3" "${4:-}" | tee -a "${STATUS_FILE}"
}

run_logged() {
  local stage="$1" name="$2"
  shift 2
  local log="${LOG_DIR}/${stage}_${name}.log"
  status "${stage}" "${name}" START "$*"
  "$@" > "${log}" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    status "${stage}" "${name}" OK "log=${log}"
  else
    status "${stage}" "${name}" FAIL "rc=${rc} log=${log}"
  fi
  return "${rc}"
}

train_candidate() {
  local name="$1" dataset="$2" scheduler="$3" warmup="$4"
  local output="${ADAPTER_DIR}/${name}"
  if [ -s "${output}/adapter_model.safetensors" ]; then
    status train "${name}" SKIP "adapter exists"
    return 0
  fi
  run_logged train "${name}" \
    "${PYTHON_BIN}" finetune/train_lora_qwen25vl.py \
      --model-path "${MODEL_PATH}" \
      --train-jsonl "${DATA_DIR}/${dataset}.jsonl" \
      --output-dir "${output}" \
      --max-examples 4096 --max-steps 250 \
      --gradient-accumulation-steps 4 --learning-rate 1e-4 \
      --lr-scheduler-type "${scheduler}" --warmup-steps "${warmup}" \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
      --seed 81 --load-in-4bit
}

retrieval_bank_for() {
  case "$1" in
    holdout_60) printf '%s/artifacts/retrieval_bank_holdout60_excluded.json' "${ROOT_DIR}" ;;
    holdout_220) printf '%s/artifacts/retrieval_bank_holdout220_excluded.json' "${ROOT_DIR}" ;;
    *) return 1 ;;
  esac
}

evaluate_candidate() {
  local name="$1" holdout="$2"
  local adapter="${ADAPTER_DIR}/${name}"
  local input="${ROOT_DIR}/local_inputs/${holdout}"
  local output="${EVAL_DIR}/${name}_${holdout}"
  local bank
  bank="$(retrieval_bank_for "${holdout}")"
  mkdir -p "${output}"
  if [ ! -s "${output}/results.json" ]; then
    run_logged eval "${name}_${holdout}" \
      env PYTHONPATH="${ROOT_DIR}/docker/medreason" \
        MEDREASON_SYSTEM=strong_baseline \
        MEDREASON_INPUT_DIR="${input}" \
        MEDREASON_OUTPUT_DIR="${output}" \
        MEDREASON_OUTPUT_FILE="${output}/results.json" \
        MEDREASON_MODEL_PATH="${MODEL_PATH}" \
        MEDREASON_LORA_PATH="${adapter}" \
        MEDREASON_OPEN_LORA_PATH="${OPEN_LORA}" \
        MEDREASON_RETRIEVAL_BANK="${bank}" \
        MEDREASON_MCQ_POLICY=hybrid_low_confidence \
        MEDREASON_OPTION_SCORE_MODE=sum5 \
        MEDREASON_VLM_OVERRIDE_CONFIDENCE_THRESHOLD=0.239 \
        MEDREASON_TOP_K_EXAMPLES=1 MEDREASON_OPEN_TOP_K_EXAMPLES=1 \
        MEDREASON_OPEN_PROMPT_STYLE=modality_guard \
        MEDREASON_MAX_NEW_TOKENS=192 MEDREASON_LOG_EVERY=25 \
        "${PYTHON_BIN}" docker/medreason/process.py || return 1
  else
    status eval "${name}_${holdout}" SKIP "results exist"
  fi
  run_logged validate "${name}_${holdout}" \
    "${SYSTEM_PYTHON}" scripts/validate_predictions.py \
      --cases-json "${input}/cases.json" --results-json "${output}/results.json" || return 1
  run_logged score "${name}_${holdout}" \
    "${SYSTEM_PYTHON}" scripts/score_predictions.py \
      --ground-truth "${input}/ground_truth.json" --results-json "${output}/results.json" --open-fuzzy || true
  run_logged calibration "${name}_${holdout}" \
    "${SYSTEM_PYTHON}" scripts/calibrate_mcq_threshold.py \
      --ground-truth "${input}/ground_truth.json" --results-json "${output}/results.json" \
      --output-json "${output}/calibration.json" --folds 5 --start 0.15 --stop 0.35 --step 0.001 || true
  run_logged open_analysis "${name}_${holdout}" \
    "${SYSTEM_PYTHON}" scripts/analyze_open_predictions.py \
      --ground-truth "${input}/ground_truth.json" --cases-json "${input}/cases.json" \
      --results-json "${output}/results.json" --retrieval-bank "${bank}" \
      --output-json "${output}/open_analysis.json" || true
}

summarize() {
  "${SYSTEM_PYTHON}" - "${RUN_DIR}" <<'PYSUMMARY'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
truth_cache = {}
rows = []
for result in sorted((run / "evals").glob("*/results.json")):
    dirname = result.parent.name
    if dirname.endswith("_holdout_220"):
        holdout = "holdout_220"
    elif dirname.endswith("_holdout_60"):
        holdout = "holdout_60"
    else:
        continue
    name = dirname[: -len("_" + holdout)]
    truth_path = Path("local_inputs") / holdout / "ground_truth.json"
    if holdout not in truth_cache:
        payload = json.loads(truth_path.read_text())
        truth_cache[holdout] = {
            str(x["case_id"]): str(x.get("answer", "")).strip().upper()
            for x in payload.get("answers", payload)
            if str(x.get("task_type", "")).lower() == "mcq"
        }
    truth = truth_cache[holdout]
    answers = json.loads(result.read_text()).get("answers", [])
    correct = sum(
        str(x.get("answer", "")).strip().upper() == truth.get(str(x.get("case_id")))
        for x in answers if str(x.get("case_id")) in truth
    )
    cal_path = result.parent / "calibration.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    open_path = result.parent / "open_analysis.json"
    open_data = json.loads(open_path.read_text()) if open_path.exists() else {}
    rows.append({
        "candidate": name, "holdout": holdout, "mcq_correct": correct, "mcq_total": len(truth),
        "full_fit_threshold": (cal.get("full_fit") or {}).get("threshold"),
        "full_fit_correct": (cal.get("full_fit") or {}).get("correct"),
        "cv_correct": cal.get("cross_validated_correct"),
        "open_answer_token_f1": open_data.get("answer_token_f1_mean"),
    })
by_name = {}
for row in rows:
    by_name.setdefault(row["candidate"], {})[row["holdout"]] = row
ranked = []
for name, holdouts in by_name.items():
    h220 = holdouts.get("holdout_220", {})
    h60 = holdouts.get("holdout_60", {})
    ranked.append({
        "candidate": name, "holdouts": holdouts,
        "rank_key": [h220.get("cv_correct", -1), h220.get("mcq_correct", -1), h60.get("mcq_correct", -1)],
    })
ranked.sort(key=lambda x: x["rank_key"], reverse=True)
baseline = {"mcq_correct": 171, "cv_correct": 169, "holdout_60_mcq": 44, "threshold": 0.239}
winner = ranked[0] if ranked else None
promote = bool(winner and winner["rank_key"] > [baseline["cv_correct"], baseline["mcq_correct"], baseline["holdout_60_mcq"]])
payload = {"baseline": baseline, "candidates": ranked, "winner": winner, "promote_over_current": promote}
(run / "reports" / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
lines = ["# Final Improvement Sweep", "", f"- promote_over_current: `{promote}`", "", "| Candidate | H220 MCQ | H220 CV | H60 MCQ |", "| --- | ---: | ---: | ---: |"]
for row in ranked:
    h220 = row["holdouts"].get("holdout_220", {})
    h60 = row["holdouts"].get("holdout_60", {})
    lines.append(f"| {row['candidate']} | {h220.get('mcq_correct', '-')}/200 | {h220.get('cv_correct', '-')}/200 | {h60.get('mcq_correct', '-')}/50 |")
(run / "reports" / "summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(payload, indent=2))
PYSUMMARY
}

package_winner() {
  local summary="${REPORT_DIR}/summary.json"
  [ -s "${summary}" ] || { status package winner SKIP "missing summary"; return 0; }
  local promote winner threshold
  read -r promote winner threshold < <("${SYSTEM_PYTHON}" - "${summary}" "${EVAL_DIR}" <<'PYPACKAGE'
import json
import statistics
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
winner = (summary.get("winner") or {}).get("candidate", "")
promote = str(bool(summary.get("promote_over_current"))).lower()
threshold = 0.239
calibration = Path(sys.argv[2]) / f"{winner}_holdout_220" / "calibration.json"
if calibration.exists():
    payload = json.loads(calibration.read_text())
    values = [float(f["train_selected_threshold"]) for f in payload.get("folds", [])]
    if values:
        threshold = statistics.median(values)
print(promote, winner, threshold)
PYPACKAGE
)
  if [ "${promote}" != "true" ] || [ -z "${winner}" ]; then
    status package winner SKIP "current open-quality-fixed remains selected"
    return 0
  fi
  local mcq_lora="${ADAPTER_DIR}/${winner}"
  [ -s "${mcq_lora}/adapter_model.safetensors" ] || { status package winner FAIL "missing ${mcq_lora}"; return 1; }
  run_logged package stage_loras \
    bash -c "mkdir -p docker/medreason/lora/final_improved_mcq docker/medreason/lora/final_improved_open && rsync -a --delete '${mcq_lora}/' docker/medreason/lora/final_improved_mcq/ && rsync -a --delete '${OPEN_LORA}/' docker/medreason/lora/final_improved_open/" || return 1
  local docker_dir="${RUN_DIR}/docker"
  local tar_path="${docker_dir}/medreason-submission-final-improved.tar"
  mkdir -p "${docker_dir}" "${RUN_DIR}/contract_outputs/smoke"
  run_logged package build_final \
    bash -c "cd docker/medreason && ./build_large.sh medreason-submission:final-improved '${tar_path}' --build-arg INSTALL_VLM_DEPS=1 --build-arg DEFAULT_LORA_PATH=/opt/app/lora/final_improved_mcq --build-arg DEFAULT_OPEN_LORA_PATH=/opt/app/lora/final_improved_open --build-arg DEFAULT_VLM_OVERRIDE_CONFIDENCE_THRESHOLD='${threshold}' --build-arg DEFAULT_MAX_NEW_TOKENS=384 --build-arg DEFAULT_OPEN_PROMPT_STYLE=modality_guard --build-arg DEFAULT_STRICT_RESOURCES=1" || return 1
  run_logged package docker_load docker load -i "${tar_path}" || return 1
  run_logged package smoke \
    docker run --rm --gpus all \
      -v "${ROOT_DIR}/docker/medreason/test:/input:ro" \
      -v "${RUN_DIR}/contract_outputs/smoke:/output" \
      medreason-submission:final-improved || return 1
  run_logged package validate_smoke \
    "${SYSTEM_PYTHON}" docker/medreason/tools/validate_output.py \
      "${RUN_DIR}/contract_outputs/smoke/results.json" \
      --input-json docker/medreason/test/cases.json || return 1
  if command -v pigz >/dev/null 2>&1; then
    run_logged package compress pigz -k -f "${tar_path}" || return 1
  else
    run_logged package compress gzip -k -f "${tar_path}" || return 1
  fi
  run_logged package gzip_test gzip -t "${tar_path}.gz" || return 1
  status package winner OK "candidate=${winner} threshold=${threshold} archive=${tar_path}.gz"
}

status sweep final_improvements START "run_dir=${RUN_DIR}"
FIRST="mcq_rtctx_perm_af_const_4096x250_s81"
if [ -n "${EXISTING_TRAIN_PID}" ] && kill -0 "${EXISTING_TRAIN_PID}" 2>/dev/null; then
  status train "${FIRST}" WAIT "existing_pid=${EXISTING_TRAIN_PID}"
  while kill -0 "${EXISTING_TRAIN_PID}" 2>/dev/null; do sleep 30; done
fi
train_candidate "${FIRST}" runtime_ctx1_perm1_answer_first_s81 constant 0 || true
train_candidate mcq_rtctx_perm_af_cos_4096x250_s81 runtime_ctx1_perm1_answer_first_s81 cosine 20 || true
train_candidate mcq_rtctx_perm_opt_cos_4096x250_s81 runtime_ctx1_perm1_option_first_s81 cosine 20 || true
for name in "${FIRST}" mcq_rtctx_perm_af_cos_4096x250_s81 mcq_rtctx_perm_opt_cos_4096x250_s81; do
  if [ ! -s "${ADAPTER_DIR}/${name}/adapter_model.safetensors" ]; then
    status eval "${name}" SKIP "missing adapter"
    continue
  fi
  evaluate_candidate "${name}" holdout_60 || true
  evaluate_candidate "${name}" holdout_220 || true
done
summarize > "${REPORT_DIR}/summary_stdout.json" 2>&1 || true
package_winner || true
status sweep final_improvements OK "summary=${REPORT_DIR}/summary.md"
