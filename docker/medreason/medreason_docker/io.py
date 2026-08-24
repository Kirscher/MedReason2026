from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import MedReasonCase, MedReasonOption, MedReasonPrediction, TaskType

_TASK_ALIASES: dict[str, TaskType] = {
    "mcq": "mcq",
    "closed": "mcq",
    "closed_ended": "mcq",
    "closed-ended": "mcq",
    "multiple_choice": "mcq",
    "multiple-choice": "mcq",
    "open": "open",
    "open_ended": "open",
    "open-ended": "open",
    "free_text": "open",
    "free-text": "open",
}


def normalize_task_type(value: str) -> TaskType:
    key = str(value).strip().lower()
    if key not in _TASK_ALIASES:
        raise ValueError(f"Unsupported task_type: {value!r}. Expected one of {sorted(_TASK_ALIASES)}")
    return _TASK_ALIASES[key]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_image_paths(raw_case: dict[str, Any], input_dir: Path) -> tuple[Path, ...]:
    raw_paths: list[str] = []
    if raw_case.get("image_path"):
        raw_paths.append(str(raw_case["image_path"]))
    if raw_case.get("image_paths"):
        if not isinstance(raw_case["image_paths"], list):
            raise ValueError(f"case_id={raw_case.get('case_id')} image_paths must be a list")
        raw_paths.extend(str(p) for p in raw_case["image_paths"])

    resolved: list[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        if not p.is_absolute():
            p = input_dir / p
        resolved.append(p)
    return tuple(resolved)


def _parse_options(raw_options: Any, case_id: str) -> tuple[MedReasonOption, ...]:
    if raw_options is None:
        return ()
    if not isinstance(raw_options, list):
        raise ValueError(f"case_id={case_id} options must be a list")

    options: list[MedReasonOption] = []
    for idx, opt in enumerate(raw_options):
        if isinstance(opt, dict):
            label = str(opt.get("label", "")).strip()
            text = str(opt.get("text", "")).strip()
        else:
            label = chr(ord("A") + idx)
            text = str(opt).strip()
        if not label or not text:
            raise ValueError(f"case_id={case_id} option[{idx}] must contain non-empty label and text")
        options.append(MedReasonOption(label=label, text=text))
    return tuple(options)


def load_cases(cases_file: Path, input_dir: Path | None = None) -> list[MedReasonCase]:
    input_dir = input_dir or cases_file.parent
    payload = _read_json(cases_file)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise ValueError("cases.json must contain either a list or an object with a 'cases' list")

    cases: list[MedReasonCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("Each case must be a JSON object")
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Each case must contain a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id in input: {case_id}")
        seen.add(case_id)

        task_type = normalize_task_type(str(raw.get("task_type", "")))
        question = str(raw.get("question", "")).strip()
        if not question:
            raise ValueError(f"case_id={case_id} question must be non-empty")
        image_paths = _resolve_image_paths(raw, input_dir=input_dir)
        if not image_paths:
            raise ValueError(f"case_id={case_id} must contain image_path or image_paths")
        options = _parse_options(raw.get("options"), case_id=case_id)
        if task_type == "mcq" and not options:
            raise ValueError(f"case_id={case_id} MCQ cases must contain options")
        cases.append(
            MedReasonCase(
                case_id=case_id,
                task_type=task_type,
                question=question,
                image_paths=image_paths,
                options=options,
                metadata=dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), dict) else {},
            )
        )
    return cases


def prediction_to_dict(prediction: MedReasonPrediction) -> dict[str, Any]:
    item: dict[str, Any] = {
        "case_id": prediction.case_id,
        "task_type": prediction.task_type,
        "answer": prediction.answer,
        "reasoning_trace": prediction.reasoning_trace,
    }
    if prediction.confidence is not None:
        item["confidence"] = prediction.confidence
    if prediction.metadata:
        item["metadata"] = prediction.metadata
    return item


def write_predictions(
    predictions: list[MedReasonPrediction],
    output_file: Path,
    submission_name: str = "MedReason system submission",
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": submission_name,
        "type": "Medical visual reasoning",
        "answers": [prediction_to_dict(p) for p in predictions],
        "version": {"major": 1, "minor": 0},
    }
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
