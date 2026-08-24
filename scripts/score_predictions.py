from __future__ import annotations

import argparse
import json
from collections import Counter
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def load_answers(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answers = payload.get("answers", payload)
    if not isinstance(answers, list):
        raise ValueError(f"{path} must contain an answers list")
    return answers


def normalize_mcq(answer: str) -> str:
    return str(answer or "").strip().upper()[:1]


def normalize_open(text: str) -> str:
    s = str(text or "")
    s = s.strip().lower()
    # remove punctuation
    s = re.sub(r"[^\w\s]|[\-_\n\r]", " ", s)
    # simple remove of common leading articles (en/fr)
    s = re.sub(r"^(the |a |an |le |la |les |un |une |des )", "", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def open_matches(pred: str, gt: str, fuzzy_thresh: float = 0.85) -> bool:
    p = normalize_open(pred)
    g = normalize_open(gt)
    if not g and not p:
        return True
    if not g or not p:
        return False
    if p == g:
        return True
    # substring match either way
    if g in p or p in g:
        return True
    # fuzzy ratio
    ratio = SequenceMatcher(None, p, g).ratio()
    return ratio >= fuzzy_thresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MedReason predictions when local labels are available.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument(
        "--open-fuzzy",
        action="store_true",
        help="Use normalized substring/fuzzy matching for open-ended answers instead of strict exact match.",
    )
    args = parser.parse_args()

    truth = {str(item["case_id"]): item for item in load_answers(args.ground_truth)}
    pred = {str(item["case_id"]): item for item in load_answers(args.results_json)}
    missing = sorted(set(truth) - set(pred))
    extra = sorted(set(pred) - set(truth))
    counts = Counter(str(item.get("task_type", "")) for item in truth.values())

    mcq_total = 0
    mcq_correct = 0
    open_exact_total = 0
    open_exact_correct = 0
    for case_id, gt in truth.items():
        item = pred.get(case_id)
        if not item:
            continue
        task = str(gt.get("task_type"))
        if task == "mcq":
            mcq_total += 1
            mcq_correct += normalize_mcq(item.get("answer", "")) == normalize_mcq(gt.get("answer", ""))
        elif task == "open":
            open_exact_total += 1
            if args.open_fuzzy:
                open_exact_correct += int(open_matches(item.get("answer", ""), gt.get("answer", "")))
            else:
                open_exact_correct += str(item.get("answer", "")).strip().lower() == str(
                    gt.get("answer", "")
                ).strip().lower()

    metrics = {
        "num_ground_truth": len(truth),
        "num_predictions": len(pred),
        "missing": len(missing),
        "extra": len(extra),
        "task_counts": dict(counts),
        "mcq_accuracy": mcq_correct / mcq_total if mcq_total else None,
        "mcq_correct": mcq_correct,
        "mcq_total": mcq_total,
        "open_exact_match": open_exact_correct / open_exact_total if open_exact_total else None,
        "open_exact_correct": open_exact_correct,
        "open_exact_total": open_exact_total,
        "open_scoring": "fuzzy" if args.open_fuzzy else "strict",
    }
    print(json.dumps(metrics, indent=2))
    if missing or extra:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
