from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BASELINE_MCQ_FULL = 166
BASELINE_MCQ_CV = 162
BASELINE_OPEN_F1 = 0.136


def load_items(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get(key, payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"{path} must contain a list or {key}")
    return items


def norm_mcq(value: Any) -> str:
    return str(value or "").strip().upper()[:1]


def validate_predictions(cases_path: Path, results_path: Path) -> dict[str, Any]:
    cases = load_items(cases_path, "cases")
    answers = load_items(results_path, "answers")
    case_by_id = {str(case["case_id"]): case for case in cases}
    answer_by_id = {str(answer.get("case_id")): answer for answer in answers}
    missing = sorted(set(case_by_id) - set(answer_by_id))
    extra = sorted(set(answer_by_id) - set(case_by_id))
    invalid_mcq = []
    empty_answer = []
    empty_open_trace = []
    for case_id, case in case_by_id.items():
        answer = answer_by_id.get(case_id)
        if answer is None:
            continue
        if not str(answer.get("answer", "")).strip():
            empty_answer.append(case_id)
        task = str(case.get("task_type", "")).strip().lower()
        pred_task = str(answer.get("task_type", "")).strip().lower()
        if task == "mcq":
            labels = {str(opt.get("label", "")).strip() for opt in case.get("options", [])}
            if labels and str(answer.get("answer", "")).strip() not in labels:
                invalid_mcq.append(case_id)
        if task == "open" and not str(answer.get("reasoning_trace", "")).strip():
            empty_open_trace.append(case_id)
        if pred_task != task:
            extra.append(f"task_mismatch:{case_id}")
    valid = not (missing or extra or invalid_mcq or empty_answer or empty_open_trace)
    return {
        "valid": valid,
        "num_cases": len(cases),
        "num_answers": len(answers),
        "missing": len(missing),
        "extra": len(extra),
        "invalid_mcq": len(invalid_mcq),
        "empty_answer": len(empty_answer),
        "empty_open_trace": len(empty_open_trace),
        "answer_counts": dict(Counter(str(answer.get("task_type", "")) for answer in answers)),
    }


def score_mcq(ground_truth_path: Path, results_path: Path) -> dict[str, Any]:
    truth = {
        str(item["case_id"]): norm_mcq(item.get("answer"))
        for item in load_items(ground_truth_path, "answers")
        if str(item.get("task_type", "")).lower() == "mcq"
    }
    correct = 0
    total = 0
    for item in load_items(results_path, "answers"):
        if str(item.get("task_type", "")).lower() != "mcq":
            continue
        case_id = str(item.get("case_id"))
        if case_id not in truth:
            continue
        total += 1
        correct += norm_mcq(item.get("answer")) == truth[case_id]
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_and_holdout(eval_dir: Path) -> tuple[str, str] | None:
    for suffix in ("_holdout_220", "_holdout_60", "_holdout_25"):
        if eval_dir.name.endswith(suffix):
            return eval_dir.name[: -len(suffix)], suffix[1:]
    return None


def summarize_eval(eval_dir: Path, root: Path) -> dict[str, Any] | None:
    parsed = candidate_and_holdout(eval_dir)
    if parsed is None:
        return None
    candidate, holdout = parsed
    results_path = eval_dir / "results.json"
    if not results_path.exists():
        return None
    input_dir = root / "local_inputs" / holdout
    cases_path = input_dir / "cases.json"
    ground_truth_path = input_dir / "ground_truth.json"
    if not cases_path.exists() or not ground_truth_path.exists():
        return None
    validation = validate_predictions(cases_path, results_path)
    mcq = score_mcq(ground_truth_path, results_path)
    open_analysis = load_json_if_exists(eval_dir / "open_analysis.json")
    calibration = load_json_if_exists(eval_dir / "calibration.json")
    return {
        "candidate": candidate,
        "holdout": holdout,
        "results_json": str(results_path),
        "valid": validation["valid"],
        "validation": validation,
        "mcq_correct": mcq["correct"],
        "mcq_total": mcq["total"],
        "mcq_accuracy": mcq["accuracy"],
        "cv_correct": calibration.get("cross_validated_correct"),
        "cv_total": calibration.get("cross_validated_total"),
        "cv_accuracy": calibration.get("cross_validated_accuracy"),
        "full_fit_threshold": (calibration.get("full_fit") or {}).get("threshold"),
        "full_fit_correct": (calibration.get("full_fit") or {}).get("correct"),
        "open_answer_token_f1": open_analysis.get("answer_token_f1_mean"),
        "open_trace_token_f1": open_analysis.get("trace_token_f1_mean"),
        "open_answer_nonzero": open_analysis.get("answer_nonzero"),
        "retrieval_best_token_f1": open_analysis.get("retrieval_best_token_f1_mean"),
    }


def accepted(group: dict[str, dict[str, Any]]) -> bool:
    h220 = group.get("holdout_220")
    h60 = group.get("holdout_60")
    if h220 is None or h60 is None:
        return False
    if not h220["valid"] or not h60["valid"]:
        return False
    cv_correct = h220.get("cv_correct")
    if cv_correct is None or cv_correct < BASELINE_MCQ_CV:
        return False
    mcq_correct = h220.get("mcq_correct") or 0
    open_f1 = h220.get("open_answer_token_f1") or 0.0
    if not (mcq_correct > BASELINE_MCQ_FULL or (mcq_correct >= 164 and open_f1 >= 0.166)):
        return False
    if (h60.get("mcq_correct") or 0) < 38:
        return False
    return True


def rank_key(group: dict[str, dict[str, Any]]) -> tuple[Any, ...]:
    h220 = group.get("holdout_220") or {}
    h60 = group.get("holdout_60") or {}
    return (
        1 if accepted(group) else 0,
        h220.get("cv_correct") or -1,
        h220.get("mcq_correct") or -1,
        h60.get("mcq_correct") or -1,
        h220.get("open_answer_token_f1") or 0.0,
        h220.get("open_trace_token_f1") or 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and select winners from a MedReason solution sweep.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    run_dir = args.run_dir
    repo_root = args.repo_root.resolve()
    eval_root = run_dir / "evals"
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if eval_root.exists():
        for eval_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
            row = summarize_eval(eval_dir, repo_root)
            if row is not None:
                rows.append(row)

    registry_path = run_dir / "candidate_registry.json"
    registry = load_json_if_exists(registry_path)
    candidates_meta = {item["name"]: item for item in registry.get("candidates", [])}

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["candidate"], {})[row["holdout"]] = row

    candidate_rows = []
    for name, group in grouped.items():
        h220 = group.get("holdout_220", {})
        h60 = group.get("holdout_60", {})
        candidate_rows.append(
            {
                "name": name,
                "accepted": accepted(group),
                "rank_key": rank_key(group),
                "meta": candidates_meta.get(name, {}),
                "holdouts": group,
                "holdout_220_mcq": h220.get("mcq_correct"),
                "holdout_220_cv": h220.get("cv_correct"),
                "holdout_220_open_f1": h220.get("open_answer_token_f1"),
                "holdout_60_mcq": h60.get("mcq_correct"),
            }
        )
    candidate_rows.sort(key=lambda row: row["rank_key"], reverse=True)

    winner = candidate_rows[0] if candidate_rows and candidate_rows[0]["accepted"] else None
    winner_payload = {
        "fallback_to_current_final": winner is None,
        "reason": "no candidate met acceptance rules" if winner is None else "candidate met acceptance rules",
        "winner": winner,
        "baseline": {
            "holdout_220_mcq_full": BASELINE_MCQ_FULL,
            "holdout_220_mcq_cv": BASELINE_MCQ_CV,
            "holdout_220_open_answer_token_f1": BASELINE_OPEN_F1,
        },
    }
    summary = {
        "run_dir": str(run_dir),
        "num_evals": len(rows),
        "num_candidates": len(candidate_rows),
        "rows": rows,
        "candidates": candidate_rows,
        "winner": winner_payload,
    }

    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (reports_dir / "winner.json").write_text(json.dumps(winner_payload, indent=2) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "winner.json").write_text(json.dumps(winner_payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# MedReason Full Solution Sweep Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- candidates: {len(candidate_rows)}",
        f"- winner: `{winner['name']}`" if winner else "- winner: fallback to current final",
        "",
        "| Candidate | Accepted | H220 MCQ | H220 CV | H220 open F1 | H60 MCQ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidate_rows[:40]:
        h220 = row["holdouts"].get("holdout_220", {})
        h60 = row["holdouts"].get("holdout_60", {})
        lines.append(
            "| {name} | {accepted} | {h220_mcq}/{h220_total} | {h220_cv}/{h220_cv_total} | {open_f1:.4f} | {h60_mcq}/{h60_total} |".format(
                name=row["name"],
                accepted="yes" if row["accepted"] else "no",
                h220_mcq=h220.get("mcq_correct", "-"),
                h220_total=h220.get("mcq_total", "-"),
                h220_cv=h220.get("cv_correct", "-"),
                h220_cv_total=h220.get("cv_total", "-"),
                open_f1=h220.get("open_answer_token_f1") or 0.0,
                h60_mcq=h60.get("mcq_correct", "-"),
                h60_total=h60.get("mcq_total", "-"),
            )
        )
    markdown = "\n".join(lines) + "\n"
    (reports_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(winner_payload, indent=2))


if __name__ == "__main__":
    main()
