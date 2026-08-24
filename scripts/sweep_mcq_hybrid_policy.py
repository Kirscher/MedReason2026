from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_items(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get(key, payload)
    if not isinstance(items, list):
        raise ValueError(f"{path} must be a list or contain {key}")
    return items


def norm(answer: Any) -> str:
    return str(answer or "").strip().upper()[:1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep confidence-gated VLM overrides for MCQ predictions."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--start", type=float, default=0.15)
    parser.add_argument("--stop", type=float, default=0.35)
    parser.add_argument("--step", type=float, default=0.001)
    args = parser.parse_args()

    truth = {
        str(item["case_id"]): norm(item.get("answer"))
        for item in load_items(args.ground_truth, "answers")
        if str(item.get("task_type", "")).lower() == "mcq"
    }
    predictions = [
        item
        for item in load_items(args.results_json, "answers")
        if str(item.get("task_type", "")).lower() == "mcq" and str(item.get("case_id")) in truth
    ]

    rows: list[dict[str, Any]] = []
    threshold = args.start
    while threshold <= args.stop + 1e-12:
        correct = 0
        overrides = 0
        override_correct = 0
        for item in predictions:
            metadata = item.get("metadata") or {}
            retrieval_answer = norm(metadata.get("retrieval_answer") or item.get("answer"))
            vlm_answer = norm(metadata.get("vlm_answer"))
            retrieval_confidence = float(metadata.get("retrieval_confidence") or 0.0)
            answer = retrieval_answer
            if vlm_answer in {"A", "B", "C", "D", "E"} and retrieval_confidence < threshold:
                answer = vlm_answer
                overrides += 1
            case_correct = answer == truth[str(item["case_id"])]
            correct += int(case_correct)
            override_correct += int(
                case_correct
                and answer == vlm_answer
                and retrieval_answer != vlm_answer
                and retrieval_confidence < threshold
            )
        total = len(predictions)
        rows.append(
            {
                "threshold": round(threshold, 4),
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else None,
                "overrides": overrides,
                "override_correct": override_correct,
            }
        )
        threshold += args.step

    rows.sort(key=lambda row: (row["accuracy"], row["correct"], -row["overrides"]), reverse=True)
    for row in rows[:20]:
        print(json.dumps(row, sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"results": rows}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
