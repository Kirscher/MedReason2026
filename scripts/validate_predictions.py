from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload)
    if not isinstance(cases, list):
        raise ValueError("cases JSON must be a list or contain a cases list")
    return cases


def load_answers(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answers = payload.get("answers", payload)
    if not isinstance(answers, list):
        raise ValueError("results JSON must be a list or contain an answers list")
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MedReason prediction coverage and field quality.")
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    args = parser.parse_args()

    cases = load_cases(args.cases_json)
    answers = load_answers(args.results_json)
    case_by_id = {str(case["case_id"]): case for case in cases}
    answer_by_id = {str(answer.get("case_id")): answer for answer in answers}

    missing = sorted(set(case_by_id) - set(answer_by_id))
    extra = sorted(set(answer_by_id) - set(case_by_id))
    invalid_mcq = []
    empty_open_trace = []
    empty_answer = []
    task_mismatch = []

    for case_id, case in case_by_id.items():
        answer = answer_by_id.get(case_id)
        if not answer:
            continue
        if not str(answer.get("answer", "")).strip():
            empty_answer.append(case_id)
        task = str(case.get("task_type", "")).strip().lower()
        pred_task = str(answer.get("task_type", "")).strip().lower()
        if task != pred_task:
            task_mismatch.append(case_id)
            print(f"task mismatch {case_id}: expected={task} got={pred_task}")
        if task == "mcq":
            labels = {str(opt.get("label", "")).strip() for opt in case.get("options", [])}
            if str(answer.get("answer", "")).strip() not in labels:
                invalid_mcq.append(case_id)
        if task == "open" and not str(answer.get("reasoning_trace", "")).strip():
            empty_open_trace.append(case_id)

    counts = Counter(str(answer.get("task_type", "")) for answer in answers)
    print(f"cases={len(cases)} answers={len(answers)} counts={dict(counts)}")
    print(f"missing={len(missing)} extra={len(extra)} invalid_mcq={len(invalid_mcq)}")
    print(f"empty_answer={len(empty_answer)} empty_open_trace={len(empty_open_trace)}")
    if missing or extra or task_mismatch or invalid_mcq or empty_answer or empty_open_trace:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
