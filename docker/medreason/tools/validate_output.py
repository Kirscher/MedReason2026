#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repository root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medreason_docker.io import load_cases, normalize_task_type  # noqa: E402


def fail(message: str) -> None:
    raise SystemExit(f"[validate_output][ERROR] {message}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_answers(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("answers"), list):
        answers = payload["answers"]
    elif isinstance(payload, list):
        answers = payload
    else:
        fail("Submission must be a JSON object with an 'answers' list, or a raw list of answers.")
    if not all(isinstance(item, dict) for item in answers):
        fail("Every item in answers must be an object.")
    return answers


def validate_basic(answers: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for i, item in enumerate(answers):
        case_id = str(item.get("case_id", "")).strip()
        if not case_id:
            fail(f"answers[{i}].case_id is required")
        if case_id in seen:
            fail(f"Duplicate case_id in answers: {case_id}")
        seen.add(case_id)

        try:
            task_type = normalize_task_type(str(item.get("task_type", "")))
        except Exception as exc:  # noqa: BLE001
            fail(f"answers[{i}].task_type is invalid: {exc}")

        answer = item.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            fail(f"answers[{i}].answer must be a non-empty string")

        reasoning = item.get("reasoning_trace", "")
        if task_type == "open" and (not isinstance(reasoning, str) or not reasoning.strip()):
            fail(f"answers[{i}].reasoning_trace is required and non-empty for open-ended cases")
        if reasoning is not None and not isinstance(reasoning, str):
            fail(f"answers[{i}].reasoning_trace must be a string when provided")


def validate_against_input(answers: list[dict[str, Any]], input_json: Path) -> None:
    cases = load_cases(input_json, input_dir=input_json.parent)
    case_by_id = {case.case_id: case for case in cases}
    answer_by_id = {str(item["case_id"]): item for item in answers}

    missing = sorted(set(case_by_id) - set(answer_by_id))
    extra = sorted(set(answer_by_id) - set(case_by_id))
    if missing:
        fail(f"Missing predictions for case_id(s): {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        fail(f"Submission contains unknown case_id(s): {extra[:10]}{'...' if len(extra) > 10 else ''}")

    for case_id, case in case_by_id.items():
        item = answer_by_id[case_id]
        pred_task = normalize_task_type(str(item.get("task_type", "")))
        if pred_task != case.task_type:
            fail(f"case_id={case_id} task_type mismatch: expected {case.task_type}, got {pred_task}")
        if case.task_type == "mcq":
            answer = str(item.get("answer", "")).strip()
            if answer not in case.option_labels:
                fail(f"case_id={case_id} answer must be one of {sorted(case.option_labels)}, got {answer!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a MedReason results.json file.")
    parser.add_argument("submission", type=Path, help="Path to results.json")
    parser.add_argument("--input-json", type=Path, default=None, help="Optional cases.json for strict case-level validation")
    args = parser.parse_args()

    payload = load_json(args.submission)
    answers = get_answers(payload)
    validate_basic(answers)
    if args.input_json is not None:
        validate_against_input(answers, args.input_json)
    print(f"[validate_output] OK: {len(answers)} predictions validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
