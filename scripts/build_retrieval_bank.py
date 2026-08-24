from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OPTION_LABELS = ("A", "B", "C", "D", "E")


def task_type(case: dict[str, Any]) -> str:
    raw = str(case.get("task_type") or case.get("question type") or "").strip().lower()
    return "open" if raw in {"open", "open-ended", "open_ended"} else "mcq"


def option_text(case: dict[str, Any], label: str) -> str:
    text = str(case.get(label) or "").strip()
    prefix = f"{label})"
    return text[len(prefix) :].strip() if text.startswith(prefix) else text


def convert_case(case: dict[str, Any]) -> dict[str, Any]:
    options = [
        {"label": label, "text": option_text(case, label)}
        for label in OPTION_LABELS
        if case.get(label)
    ]
    return {
        "case_id": str(case["case_id"]),
        "task_type": task_type(case),
        "question": str(case.get("question") or ""),
        "answer": str(case.get("answer") or "").strip(),
        "options": options,
        "visual_description": str(case.get("visual_description") or ""),
        "metadata": {
            "modality": case.get("modality") or case.get("primary_modality"),
            "organ_system": case.get("organ system"),
            "task": case.get("task"),
            "subtype": case.get("subtype"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact train retrieval bank for MedReason.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_bank.json"))
    parser.add_argument("--max-examples", type=int, default=0, help="0 keeps all answered train examples.")
    parser.add_argument(
        "--exclude-case-ids",
        type=Path,
        default=None,
        help="Optional JSON file containing cases/answers whose case_id values should be excluded.",
    )
    args = parser.parse_args()

    payload = json.loads(args.train_json.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    if args.exclude_case_ids:
        exclude_payload = json.loads(args.exclude_case_ids.read_text(encoding="utf-8"))
        exclude_items = exclude_payload.get("cases") or exclude_payload.get("answers") or exclude_payload
        if not isinstance(exclude_items, list):
            raise ValueError("--exclude-case-ids must be a list or contain cases/answers")
        excluded = {str(item["case_id"]) for item in exclude_items}

    cases = payload["cases"]
    examples = [
        convert_case(case)
        for case in cases
        if str(case.get("answer") or "").strip() and str(case.get("case_id")) not in excluded
    ]
    if args.max_examples > 0:
        examples = examples[: args.max_examples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "name": "MedReason train retrieval bank",
                "source": str(args.train_json),
                "num_examples": len(examples),
                "examples": examples,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output} examples={len(examples)}")


if __name__ == "__main__":
    main()
