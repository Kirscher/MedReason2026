from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "what",
    "which",
}


def load_items(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get(key, payload)
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a list or {key}")
    return items


def tokens(text: Any) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if token not in STOPWORDS
    ]


def token_f1(prediction: Any, reference: Any) -> float:
    pred = tokens(prediction)
    ref = tokens(reference)
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute lightweight lexical diagnostics for open-ended MedReason predictions."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path, default=None)
    parser.add_argument("--retrieval-bank", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    truth = {
        str(item["case_id"]): item
        for item in load_items(args.ground_truth, "answers")
        if str(item.get("task_type", "")).lower() == "open"
    }
    predictions = {
        str(item["case_id"]): item
        for item in load_items(args.results_json, "answers")
        if str(item.get("case_id")) in truth
    }
    cases = {}
    if args.cases_json:
        cases = {str(item["case_id"]): item for item in load_items(args.cases_json, "cases")}

    bank = None
    if args.retrieval_bank:
        import sys

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "docker" / "medreason"))
        from medreason_docker.retrieval import RetrievalBank

        bank = RetrievalBank.from_json(args.retrieval_bank)

    rows = []
    for case_id, gt in truth.items():
        pred = predictions.get(case_id, {})
        row: dict[str, Any] = {
            "case_id": case_id,
            "ground_truth": gt.get("answer", ""),
            "answer": pred.get("answer", ""),
            "reasoning_trace": pred.get("reasoning_trace", ""),
            "answer_token_f1": token_f1(pred.get("answer", ""), gt.get("answer", "")),
            "trace_token_f1": token_f1(pred.get("reasoning_trace", ""), gt.get("answer", "")),
        }
        if bank and case_id in cases:
            case = cases[case_id]
            metadata = case.get("metadata") or {}
            query = " ".join(
                [
                    str(case.get("question") or ""),
                    " ".join(str(value) for value in metadata.values() if value),
                ]
            )
            hits = bank.search(query, task_type="open", top_k=args.top_k)
            row["retrieval_candidates"] = [
                {
                    "rank": rank,
                    "score": hit.score,
                    "case_id": hit.example.get("case_id"),
                    "answer": hit.example.get("answer", ""),
                    "answer_token_f1": token_f1(hit.example.get("answer", ""), gt.get("answer", "")),
                }
                for rank, hit in enumerate(hits, start=1)
            ]
        rows.append(row)

    answer_scores = [row["answer_token_f1"] for row in rows]
    trace_scores = [row["trace_token_f1"] for row in rows]
    retrieval_best = [
        max((cand["answer_token_f1"] for cand in row.get("retrieval_candidates", [])), default=0.0)
        for row in rows
    ]
    summary = {
        "num_open": len(rows),
        "answer_token_f1_mean": sum(answer_scores) / len(answer_scores) if rows else None,
        "trace_token_f1_mean": sum(trace_scores) / len(trace_scores) if rows else None,
        "answer_nonzero": sum(score > 0 for score in answer_scores),
        "retrieval_best_token_f1_mean": sum(retrieval_best) / len(retrieval_best)
        if rows and any("retrieval_candidates" in row for row in rows)
        else None,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
