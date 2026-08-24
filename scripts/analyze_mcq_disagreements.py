from __future__ import annotations

import argparse
import json
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Analyze MCQ VLM/retrieval disagreement metadata.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    truth = {str(item["case_id"]): norm(item.get("answer")) for item in load_items(args.ground_truth, "answers")}
    predictions = load_items(args.results_json, "answers")

    counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for item in predictions:
        if str(item.get("task_type", "")).lower() != "mcq" or str(item.get("case_id")) not in truth:
            continue
        case_id = str(item["case_id"])
        metadata = item.get("metadata") or {}
        final_answer = norm(item.get("answer"))
        vlm_answer = norm(metadata.get("vlm_answer"))
        retrieval_answer = norm(metadata.get("retrieval_answer"))
        gt = truth[case_id]
        final_correct = final_answer == gt
        vlm_correct = vlm_answer == gt if vlm_answer else None
        retrieval_correct = retrieval_answer == gt if retrieval_answer else None
        disagree = bool(vlm_answer and retrieval_answer and vlm_answer != retrieval_answer)

        counts["mcq_total"] += 1
        counts["final_correct"] += int(final_correct)
        if vlm_correct is not None:
            counts["vlm_available"] += 1
            counts["vlm_correct"] += int(vlm_correct)
        if retrieval_correct is not None:
            counts["retrieval_available"] += 1
            counts["retrieval_correct"] += int(retrieval_correct)
        if disagree:
            counts["disagree"] += 1
            counts["disagree_final_correct"] += int(final_correct)
            counts["disagree_vlm_correct"] += int(bool(vlm_correct))
            counts["disagree_retrieval_correct"] += int(bool(retrieval_correct))
            cases.append(
                {
                    "case_id": case_id,
                    "ground_truth": gt,
                    "final_answer": final_answer,
                    "vlm_answer": vlm_answer,
                    "retrieval_answer": retrieval_answer,
                    "retrieval_confidence": metadata.get("retrieval_confidence"),
                    "final_correct": final_correct,
                    "vlm_correct": vlm_correct,
                    "retrieval_correct": retrieval_correct,
                }
            )

    summary = {
        "counts": dict(counts),
        "accuracy": counts["final_correct"] / counts["mcq_total"] if counts["mcq_total"] else None,
        "vlm_accuracy": counts["vlm_correct"] / counts["vlm_available"]
        if counts["vlm_available"]
        else None,
        "retrieval_accuracy": counts["retrieval_correct"] / counts["retrieval_available"]
        if counts["retrieval_available"]
        else None,
        "disagreement_cases": cases,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "disagreement_cases"}, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
