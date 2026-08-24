from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


VALID_OPTIONS = {"A", "B", "C", "D", "E"}
LABEL_RE = r"[A-E]"


def load_items(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get(key, payload)
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a list or {key}")
    return items


def norm(value: Any) -> str:
    text = str(value or "").strip()
    upper = text.upper()
    if upper in VALID_OPTIONS:
        return upper
    patterns = [
        rf"^\(?({LABEL_RE})\)?[\s\.:\-)]*$",
        rf"^\(?({LABEL_RE})\)?[\.:\-)]\s*.+$",
        rf"^(?:OPTION|ANSWER|FINAL ANSWER|FINAL)\s*[:\-]?\s*\(?({LABEL_RE})\)?(?:\s|[\.\:\-\)]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return upper


def fold_for_case(case_id: str, folds: int, seed: str) -> int:
    digest = hashlib.sha1(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    values = []
    value = start
    while value <= stop + 1e-12:
        values.append(round(value, 4))
        value += step
    return values


def apply_policy(row: dict[str, Any], threshold: float) -> str:
    metadata = row["metadata"]
    retrieval_answer = norm(metadata.get("retrieval_answer") or row.get("answer"))
    vlm_answer = norm(metadata.get("vlm_answer"))
    retrieval_confidence = float(metadata.get("retrieval_confidence") or 0.0)
    if vlm_answer in VALID_OPTIONS and retrieval_confidence < threshold:
        return vlm_answer
    return retrieval_answer


def score(rows: list[dict[str, Any]], threshold: float) -> tuple[int, int]:
    correct = sum(apply_policy(row, threshold) == row["truth"] for row in rows)
    return correct, len(rows)


def best_threshold(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        correct, total = score(rows, threshold)
        result = {
            "threshold": threshold,
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else None,
        }
        if best is None or (result["correct"], -result["threshold"]) > (
            best["correct"],
            -best["threshold"],
        ):
            best = result
    if best is None:
        raise RuntimeError("No thresholds evaluated")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate confidence-gated MCQ VLM overrides with hash k-fold validation."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", default="medreason")
    parser.add_argument("--start", type=float, default=0.15)
    parser.add_argument("--stop", type=float, default=0.35)
    parser.add_argument("--step", type=float, default=0.001)
    args = parser.parse_args()

    truth = {
        str(item["case_id"]): norm(item.get("answer"))
        for item in load_items(args.ground_truth, "answers")
        if str(item.get("task_type", "")).lower() == "mcq"
    }
    rows = []
    for item in load_items(args.results_json, "answers"):
        case_id = str(item.get("case_id"))
        if str(item.get("task_type", "")).lower() != "mcq" or case_id not in truth:
            continue
        metadata = item.get("metadata") or {}
        if not metadata.get("retrieval_answer") or not metadata.get("vlm_answer"):
            continue
        rows.append(
            {
                "case_id": case_id,
                "truth": truth[case_id],
                "answer": item.get("answer"),
                "metadata": metadata,
                "fold": fold_for_case(case_id, args.folds, args.seed),
            }
        )

    thresholds = threshold_grid(args.start, args.stop, args.step)
    full = best_threshold(rows, thresholds)
    fold_results = []
    total_correct = 0
    total = 0
    for fold in range(args.folds):
        train = [row for row in rows if row["fold"] != fold]
        valid = [row for row in rows if row["fold"] == fold]
        selected = best_threshold(train, thresholds)
        valid_correct, valid_total = score(valid, float(selected["threshold"]))
        total_correct += valid_correct
        total += valid_total
        fold_results.append(
            {
                "fold": fold,
                "train_selected_threshold": selected["threshold"],
                "train_accuracy": selected["accuracy"],
                "valid_correct": valid_correct,
                "valid_total": valid_total,
                "valid_accuracy": valid_correct / valid_total if valid_total else None,
            }
        )

    summary = {
        "num_rows": len(rows),
        "full_fit": full,
        "folds": fold_results,
        "cross_validated_correct": total_correct,
        "cross_validated_total": total,
        "cross_validated_accuracy": total_correct / total if total else None,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
