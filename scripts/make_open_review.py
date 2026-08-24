from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Render open-ended diagnostics as a review markdown.")
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--sort-by", choices=["answer_token_f1", "trace_token_f1"], default="answer_token_f1")
    args = parser.parse_args()

    payload = json.loads(args.analysis_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = payload.get("rows") or []
    rows.sort(key=lambda row: (row.get(args.sort_by) or 0.0, row.get("case_id", "")))

    lines = [
        "# Open-Ended Review",
        "",
        f"- source: `{args.analysis_json}`",
        f"- num_open: {payload.get('num_open')}",
        f"- answer_token_f1_mean: {payload.get('answer_token_f1_mean')}",
        f"- trace_token_f1_mean: {payload.get('trace_token_f1_mean')}",
        f"- retrieval_best_token_f1_mean: {payload.get('retrieval_best_token_f1_mean')}",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('case_id')}",
                "",
                f"- answer_token_f1: `{row.get('answer_token_f1'):.4f}`",
                f"- trace_token_f1: `{row.get('trace_token_f1'):.4f}`",
                "",
                "**Ground truth**",
                "",
                str(row.get("ground_truth", "")).strip(),
                "",
                "**Prediction**",
                "",
                str(row.get("answer", "")).strip(),
                "",
                "**Reasoning trace**",
                "",
                str(row.get("reasoning_trace", "")).strip(),
                "",
                "**Retrieved open candidates**",
                "",
            ]
        )
        candidates = row.get("retrieval_candidates") or []
        if not candidates:
            lines.append("- none")
        for cand in candidates:
            lines.append(
                "- "
                f"rank `{cand.get('rank')}`, score `{cand.get('score'):.4f}`, "
                f"token_f1 `{cand.get('answer_token_f1'):.4f}`, "
                f"case `{cand.get('case_id')}`: {str(cand.get('answer', '')).strip()}"
            )
        lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
