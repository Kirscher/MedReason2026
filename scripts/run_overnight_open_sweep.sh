#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
MEDREASON_DATA_ROOT="${MEDREASON_DATA_ROOT:-${ROOT_DIR}/data}"

RUN_ID="${MEDREASON_SWEEP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${MEDREASON_SWEEP_RUN_DIR:-local_outputs/overnight_open_sweep_${RUN_ID}}"
LOG_DIR="${RUN_DIR}/logs"
DATA_DIR="${RUN_DIR}/datasets"
EVAL_DIR="${RUN_DIR}/evals"
STATUS_TSV="${RUN_DIR}/status.tsv"
SUMMARY_JSON="${RUN_DIR}/summary.json"
SUMMARY_MD="${RUN_DIR}/summary.md"
LATEST_LINK="local_outputs/overnight_open_sweep_latest"

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-${MEDREASON_DATA_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
TRAIN_JSON="${TRAIN_JSON:-${MEDREASON_DATA_ROOT}/train/medreason_train_selection.json}"
TRAIN_IMG_DIR="${TRAIN_IMG_DIR:-${MEDREASON_DATA_ROOT}/train/imgs}"
MCQ_LORA_PATH="${MCQ_LORA_PATH:-artifacts/finetune/qwen25vl3b-lora-mcq1024x150-s41}"
MCQ_THRESHOLD="${MCQ_THRESHOLD:-0.239}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
EVAL_HOLDOUTS="${EVAL_HOLDOUTS:-holdout_25 holdout_220}"

mkdir -p "${LOG_DIR}" "${DATA_DIR}" "${EVAL_DIR}" "$(dirname "${STATUS_TSV}")"
rm -f "${LATEST_LINK}"
ln -s "$(realpath "${RUN_DIR}")" "${LATEST_LINK}"

exec > >(tee -a "${LOG_DIR}/master.log") 2>&1

echo "run_id=${RUN_ID}"
echo "run_dir=${RUN_DIR}"
echo "started_at=$(date --iso-8601=seconds)"
echo "python=${PYTHON_BIN}"
echo "model_path=${MODEL_PATH}"
echo "mcq_lora_path=${MCQ_LORA_PATH}"
echo "mcq_threshold=${MCQ_THRESHOLD}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "eval_holdouts=${EVAL_HOLDOUTS}"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true

printf "timestamp\tstage\tname\tstatus\tdetails\n" > "${STATUS_TSV}"

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
  status "${stage}" "${name}" "START" "$*"
  "$@" >"${log}" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    status "${stage}" "${name}" "OK" "log=${log}"
  else
    status "${stage}" "${name}" "FAIL" "rc=${rc} log=${log}"
  fi
  return "${rc}"
}

build_combined_exclude() {
  local output="${RUN_DIR}/holdout_exclude_cases.json"
  python3 - "$output" <<'PY'
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

EXCLUDE_JSON="$(build_combined_exclude)"
RETRIEVAL_BANK="${RUN_DIR}/retrieval_bank_all_holdouts_excluded.json"

if [ ! -s "${RETRIEVAL_BANK}" ]; then
  run_logged prep retrieval_bank \
    python3 scripts/build_retrieval_bank.py \
      --train-json "${TRAIN_JSON}" \
      --exclude-case-ids "${EXCLUDE_JSON}" \
      --output "${RETRIEVAL_BANK}" || true
else
  status prep retrieval_bank SKIP "exists=${RETRIEVAL_BANK}"
fi

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
      "$@" || true
}

build_dataset open_noctx_512_s51 --top-k-examples 0 --shuffle --seed 51 --open-only --max-examples 512
build_dataset open_noctx_1024_s52 --top-k-examples 0 --shuffle --seed 52 --open-only --max-examples 1024
build_dataset open_ctx1_512_s53 --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 --shuffle --seed 53 --open-only --max-examples 512
build_dataset open_ctx1_1024_s54 --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 --shuffle --seed 54 --open-only --max-examples 1024
build_dataset mixed_openheavy_noctx_1024_s55 --top-k-examples 0 --shuffle --seed 55 --max-open 768 --max-mcq 256 --max-examples 1024
build_dataset mixed_openheavy_ctx1_1024_s56 --retrieval-bank "${RETRIEVAL_BANK}" --top-k-examples 1 --shuffle --seed 56 --max-open 768 --max-mcq 256 --max-examples 1024

train_adapter() {
  local name="$1"
  local dataset="$2"
  local max_examples="$3"
  local max_steps="$4"
  local lr="$5"
  local seed="$6"
  local save_every="${7:-0}"
  local output_dir="artifacts/finetune/${name}"
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
      --lora-r 16 \
      --lora-alpha 32 \
      --lora-dropout 0.05 \
      --seed "${seed}" \
      --load-in-4bit \
      --save-every "${save_every}" || true
}

evaluate_adapter() {
  local name="$1"
  local open_lora_path="$2"
  for holdout in ${EVAL_HOLDOUTS}; do
    local input_dir="local_inputs/${holdout}"
    local retrieval_bank="artifacts/retrieval_bank_${holdout#holdout_}_excluded.json"
    if [ "${holdout}" = "holdout_25" ]; then
      retrieval_bank="artifacts/retrieval_bank_holdout25_excluded.json"
    elif [ "${holdout}" = "holdout_60" ]; then
      retrieval_bank="artifacts/retrieval_bank_holdout60_excluded.json"
    elif [ "${holdout}" = "holdout_220" ]; then
      retrieval_bank="artifacts/retrieval_bank_holdout220_excluded.json"
    fi
    local out_dir="${EVAL_DIR}/${name}_${holdout}"
    mkdir -p "${out_dir}"
    if [ -s "${out_dir}/results.json" ]; then
      status eval "${name}_${holdout}" SKIP "exists=${out_dir}/results.json"
    else
      run_logged eval "${name}_${holdout}" \
        env PYTHONPATH=docker/medreason \
          MEDREASON_SYSTEM=strong_baseline \
          MEDREASON_INPUT_DIR="${input_dir}" \
          MEDREASON_OUTPUT_FILE="${out_dir}/results.json" \
          MEDREASON_MODEL_PATH="${MODEL_PATH}" \
          MEDREASON_LORA_PATH="${MCQ_LORA_PATH}" \
          MEDREASON_OPEN_LORA_PATH="${open_lora_path}" \
          MEDREASON_RETRIEVAL_BANK="${retrieval_bank}" \
          MEDREASON_MCQ_POLICY=hybrid_low_confidence \
          MEDREASON_OPTION_SCORE_MODE=sum5 \
          MEDREASON_VLM_OVERRIDE_CONFIDENCE_THRESHOLD="${MCQ_THRESHOLD}" \
          MEDREASON_TOP_K_EXAMPLES=1 \
          MEDREASON_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
          MEDREASON_LOG_EVERY=25 \
          "${PYTHON_BIN}" docker/medreason/process.py || true
    fi
    if [ -s "${out_dir}/results.json" ]; then
      run_logged validate "${name}_${holdout}" \
        python3 scripts/validate_predictions.py \
          --cases-json "${input_dir}/cases.json" \
          --results-json "${out_dir}/results.json" || true
      run_logged open_analysis "${name}_${holdout}" \
        python3 scripts/analyze_open_predictions.py \
          --ground-truth "${input_dir}/ground_truth.json" \
          --cases-json "${input_dir}/cases.json" \
          --results-json "${out_dir}/results.json" \
          --retrieval-bank "${retrieval_bank}" \
          --output-json "${out_dir}/open_analysis.json" || true
      run_logged calibration "${name}_${holdout}" \
        python3 scripts/calibrate_mcq_threshold.py \
          --ground-truth "${input_dir}/ground_truth.json" \
          --results-json "${out_dir}/results.json" \
          --output-json "${out_dir}/calibration.json" \
          --start 0.15 --stop 0.35 --step 0.001 || true
    fi
  done
}

train_adapter open_only_512x100_lr1e4_s51 open_noctx_512_s51 512 100 1e-4 51 0
evaluate_adapter open_only_512x100_lr1e4_s51 artifacts/finetune/open_only_512x100_lr1e4_s51

train_adapter open_only_512x200_lr1e4_s52 open_noctx_512_s51 512 200 1e-4 52 100
evaluate_adapter open_only_512x200_lr1e4_s52 artifacts/finetune/open_only_512x200_lr1e4_s52

train_adapter open_only_1024x150_lr1e4_s53 open_noctx_1024_s52 1024 150 1e-4 53 0
evaluate_adapter open_only_1024x150_lr1e4_s53 artifacts/finetune/open_only_1024x150_lr1e4_s53

train_adapter open_only_1024x150_lr2e4_s54 open_noctx_1024_s52 1024 150 2e-4 54 0
evaluate_adapter open_only_1024x150_lr2e4_s54 artifacts/finetune/open_only_1024x150_lr2e4_s54

train_adapter mixed_openheavy_1024x150_lr1e4_s55 mixed_openheavy_noctx_1024_s55 1024 150 1e-4 55 0
evaluate_adapter mixed_openheavy_1024x150_lr1e4_s55 artifacts/finetune/mixed_openheavy_1024x150_lr1e4_s55

train_adapter open_ctx1_512x100_lr1e4_s56 open_ctx1_512_s53 512 100 1e-4 56 0
evaluate_adapter open_ctx1_512x100_lr1e4_s56 artifacts/finetune/open_ctx1_512x100_lr1e4_s56

train_adapter open_ctx1_1024x150_lr1e4_s57 open_ctx1_1024_s54 1024 150 1e-4 57 0
evaluate_adapter open_ctx1_1024x150_lr1e4_s57 artifacts/finetune/open_ctx1_1024x150_lr1e4_s57

evaluate_adapter baseline_open_base base
evaluate_adapter baseline_open_512 artifacts/finetune/qwen25vl3b-lora-512x100

python3 - "${RUN_DIR}" "${SUMMARY_JSON}" "${SUMMARY_MD}" <<'PY'
import json
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
eval_dir = run_dir / "evals"

def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def mcq_score(results_path, truth_path):
    truth = {
        str(item["case_id"]): str(item.get("answer", "")).strip().upper()[:1]
        for item in load_json(truth_path)["answers"]
        if str(item.get("task_type", "")).lower() == "mcq"
    }
    correct = total = 0
    for item in load_json(results_path)["answers"]:
        if str(item.get("task_type", "")).lower() != "mcq":
            continue
        cid = str(item.get("case_id"))
        if cid not in truth:
            continue
        total += 1
        correct += str(item.get("answer", "")).strip().upper()[:1] == truth[cid]
    return correct, total, correct / total if total else None

rows = []
for out_dir in sorted(eval_dir.glob("*_holdout_*")):
    match = re.match(r"(.+)_(holdout_\d+)$", out_dir.name)
    if not match:
        continue
    name, holdout = match.groups()
    results = out_dir / "results.json"
    open_analysis = out_dir / "open_analysis.json"
    truth = Path("local_inputs") / holdout / "ground_truth.json"
    if not results.exists() or not open_analysis.exists() or not truth.exists():
        continue
    mcq_correct, mcq_total, mcq_acc = mcq_score(results, truth)
    open_payload = load_json(open_analysis)
    rows.append(
        {
            "name": name,
            "holdout": holdout,
            "mcq_correct": mcq_correct,
            "mcq_total": mcq_total,
            "mcq_accuracy": mcq_acc,
            "open_answer_token_f1": open_payload.get("answer_token_f1_mean"),
            "open_trace_token_f1": open_payload.get("trace_token_f1_mean"),
            "open_answer_nonzero": open_payload.get("answer_nonzero"),
            "retrieval_best_token_f1": open_payload.get("retrieval_best_token_f1_mean"),
            "results_json": str(results),
        }
    )

rows.sort(
    key=lambda row: (
        row["holdout"] != "holdout_220",
        -(row["open_answer_token_f1"] or 0),
        -(row["open_trace_token_f1"] or 0),
        row["name"],
    )
)
summary = {"run_dir": str(run_dir), "rows": rows}
summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Overnight Open Sweep Summary",
    "",
    f"- run_dir: `{run_dir}`",
    f"- rows: {len(rows)}",
    "",
    "| Adapter | Holdout | MCQ | Open answer F1 | Open trace F1 | Nonzero |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]
for row in rows:
    mcq = f"{row['mcq_correct']}/{row['mcq_total']}" if row["mcq_total"] else "-"
    lines.append(
        "| {name} | {holdout} | {mcq} | {af1:.4f} | {tf1:.4f} | {nz} |".format(
            name=row["name"],
            holdout=row["holdout"],
            mcq=mcq,
            af1=row["open_answer_token_f1"] or 0,
            tf1=row["open_trace_token_f1"] or 0,
            nz=row["open_answer_nonzero"],
        )
    )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

status report summary OK "json=${SUMMARY_JSON} md=${SUMMARY_MD}"
echo "finished_at=$(date --iso-8601=seconds)"
