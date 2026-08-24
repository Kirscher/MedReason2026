from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VALID_OPTIONS = {"A", "B", "C", "D", "E"}
LABEL_RE = r"[A-E]"


def norm_answer(value: Any) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a confidence-gated VLM override to an existing MedReason results.json."
    )
    parser.add_argument("--input-results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input_results.read_text(encoding="utf-8"))
    answers = payload.get("answers")
    if not isinstance(answers, list):
        raise ValueError(f"{args.input_results} must contain an answers list")

    overrides = 0
    for item in answers:
        if str(item.get("task_type", "")).lower() != "mcq":
            continue
        metadata = item.setdefault("metadata", {})
        retrieval_answer = norm_answer(metadata.get("retrieval_answer") or item.get("answer"))
        vlm_answer = norm_answer(metadata.get("vlm_answer"))
        retrieval_confidence = float(metadata.get("retrieval_confidence") or 0.0)

        final_answer = retrieval_answer
        metadata.pop("mcq_override", None)
        if vlm_answer in VALID_OPTIONS and retrieval_confidence < args.threshold:
            final_answer = vlm_answer
            overrides += 1
            if final_answer != retrieval_answer:
                metadata["mcq_override"] = "vlm_low_retrieval_confidence"
        item["answer"] = final_answer
        metadata["mcq_policy"] = "hybrid_low_confidence"
        metadata["vlm_override_confidence_threshold"] = args.threshold

    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    args.output_results.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_results} overrides={overrides} threshold={args.threshold}")


if __name__ == "__main__":
    main()
