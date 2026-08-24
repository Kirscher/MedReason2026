from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply an experimental retrieval-gap fallback to open-ended predictions."
    )
    parser.add_argument("--input-results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, default=None)
    parser.add_argument("--min-score-gap", type=float, default=0.04)
    parser.add_argument("--min-top-score", type=float, default=0.0)
    args = parser.parse_args()

    payload = json.loads(args.input_results.read_text(encoding="utf-8"))
    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise ValueError(f"{args.input_results} must contain an answers list")

    analysis_candidates: dict[str, list[dict[str, Any]]] = {}
    if args.analysis_json:
        analysis = json.loads(args.analysis_json.read_text(encoding="utf-8"))
        for row in analysis.get("rows") or []:
            analysis_candidates[str(row.get("case_id"))] = row.get("retrieval_candidates") or []

    overrides = 0
    for item in answers:
        if str(item.get("task_type", "")).lower() != "open":
            continue
        metadata: dict[str, Any] = item.setdefault("metadata", {})
        candidates = metadata.get("open_retrieval_candidates") or analysis_candidates.get(str(item.get("case_id"))) or []
        if not candidates:
            continue
        top = candidates[0]
        second_score = float(candidates[1].get("score") or 0.0) if len(candidates) > 1 else 0.0
        top_score = float(top.get("score") or 0.0)
        gap = top_score - second_score
        top_answer = str(top.get("answer") or "").strip()
        if top_answer and top_score >= args.min_top_score and gap >= args.min_score_gap:
            metadata["open_override"] = "retrieval_score_gap"
            metadata["open_override_previous_answer"] = item.get("answer", "")
            metadata["open_override_score_gap"] = round(gap, 4)
            item["answer"] = top_answer
            if not str(item.get("reasoning_trace") or "").strip():
                item["reasoning_trace"] = f"Retrieved a high-confidence analogous open-ended case: {top_answer}"
            overrides += 1

    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_results} open_overrides={overrides}")


if __name__ == "__main__":
    main()
