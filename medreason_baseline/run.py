from __future__ import annotations

import argparse
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .retrieval import TfidfIndex, tokenize

OPTION_LABELS = ("A", "B", "C", "D", "E")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def task_type(case: dict[str, Any]) -> str:
    raw = str(case.get("task_type") or case.get("question type") or "").lower()
    if raw in {"open", "open-ended", "open_ended"}:
        return "open"
    return "mcq"


def option_text(case: dict[str, Any], label: str) -> str:
    text = str(case.get(label) or "")
    prefix = f"{label})"
    if text.startswith(prefix):
        return text[len(prefix) :].strip()
    return text.strip()


def case_text(case: dict[str, Any], *, include_answer_option: bool = False) -> str:
    parts = [
        str(case.get("question") or ""),
        str(case.get("visual_description") or ""),
        str(case.get("modality") or ""),
        str(case.get("primary_modality") or ""),
        str(case.get("organ system") or ""),
        str(case.get("task") or ""),
        str(case.get("subtype") or ""),
    ]
    if include_answer_option:
        answer = str(case.get("answer") or "").strip().upper()
        if answer in OPTION_LABELS:
            parts.append(option_text(case, answer))
    else:
        parts.extend(option_text(case, label) for label in OPTION_LABELS)
    return " ".join(part for part in parts if part)


def load_splits(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = load_json(data_root / "train" / "medreason_train_selection.json")["cases"]
    validation = load_json(
        data_root / "validation" / "medreason_validation_participant_facing.json"
    )["cases"]
    return train, validation


def to_official_case(case: dict[str, Any]) -> dict[str, Any]:
    item: dict[str, Any] = {
        "case_id": case["case_id"],
        "task_type": task_type(case),
        "image_path": str(case.get("image_path") or ""),
        "question": str(case.get("question") or ""),
    }
    options = [
        {"label": label, "text": option_text(case, label)}
        for label in OPTION_LABELS
        if case.get(label)
    ]
    if options:
        item["options"] = options
    return item


class RetrievalBaseline:
    def __init__(self, train_cases: list[dict[str, Any]]) -> None:
        self.mcq_cases = [
            case
            for case in train_cases
            if task_type(case) == "mcq" and str(case.get("answer") or "").upper() in OPTION_LABELS
        ]
        self.open_cases = [
            case for case in train_cases if task_type(case) == "open" and case.get("answer")
        ]

        self.mcq_option_index = TfidfIndex(
            case_text(case, include_answer_option=True) for case in self.mcq_cases
        )
        self.open_index = TfidfIndex(case_text(case) for case in self.open_cases)
        self.majority_answer = Counter(
            str(case.get("answer") or "").strip().upper() for case in self.mcq_cases
        ).most_common(1)[0][0]
        self.correct_centroid = self._build_correct_centroid()

    def _build_correct_centroid(self) -> dict[str, float]:
        terms: Counter[str] = Counter()
        for case in self.mcq_cases:
            terms.update(tokenize(case_text(case, include_answer_option=True)))
        weights = {
            term: (1.0 + math.log(tf)) * self.mcq_option_index.idf.get(term, 1.0)
            for term, tf in terms.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
        return {term: weight / norm for term, weight in weights.items()}

    def _centroid_score(self, text: str) -> float:
        terms = Counter(tokenize(text))
        if not terms:
            return 0.0
        weights = {
            term: (1.0 + math.log(tf)) * self.mcq_option_index.idf.get(term, 1.0)
            for term, tf in terms.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
        return sum(
            (weight / norm) * self.correct_centroid.get(term, 0.0)
            for term, weight in weights.items()
        )

    def predict_mcq(self, case: dict[str, Any]) -> dict[str, Any]:
        option_scores: dict[str, float] = {}
        for label in OPTION_LABELS:
            if not case.get(label):
                continue
            query = f"{case.get('question', '')} {option_text(case, label)}"
            option_scores[label] = self._centroid_score(query)

        answer = max(option_scores, key=option_scores.get) if option_scores else self.majority_answer
        confidence = max(option_scores.values()) if option_scores else 0.0
        return {
            "case_id": case["case_id"],
            "task_type": "mcq",
            "answer": answer,
            "reasoning_trace": "Text-retrieval baseline selected the option most similar to answered training cases.",
            "confidence": round(float(confidence), 4),
        }

    def predict_open(self, case: dict[str, Any]) -> dict[str, Any]:
        hits = self.open_index.search(case_text(case), top_k=1)
        if hits:
            neighbor = self.open_cases[hits[0].doc_id]
            answer = str(neighbor.get("answer") or "").strip()
            visual = str(neighbor.get("visual_description") or "").strip()
            reasoning = (
                "Nearest-neighbor baseline used a visually described training case with "
                f"similar question wording. Retrieved visual cue: {visual}"
                if visual
                else "Nearest-neighbor baseline used the most similar answered training case."
            )
        else:
            answer = "The abnormality cannot be determined confidently from this baseline."
            reasoning = "No sufficiently similar open-ended training case was retrieved."

        return {
            "case_id": case["case_id"],
            "task_type": "open",
            "reasoning_trace": reasoning,
            "answer": answer,
        }

    def predict(self, case: dict[str, Any]) -> dict[str, Any]:
        return self.predict_open(case) if task_type(case) == "open" else self.predict_mcq(case)


def write_submission_zip(results_path: Path) -> Path:
    zip_path = results_path.with_name("validation_submission.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(results_path, arcname="results.json")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight MedReason baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["validation"], default="validation")
    parser.add_argument("--output", type=Path, default=Path("outputs/validation_results.json"))
    args = parser.parse_args()

    train_cases, validation_cases = load_splits(args.data_root)
    baseline = RetrievalBaseline(train_cases)
    answers = [baseline.predict(case) for case in validation_cases]

    payload = {
        "name": "MedReason retrieval baseline",
        "type": "Medical visual reasoning",
        "answers": answers,
        "version": {"major": 0, "minor": 1},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    cases_path = args.output.with_name("validation_cases_official.json")
    with cases_path.open("w", encoding="utf-8") as f:
        json.dump({"cases": [to_official_case(case) for case in validation_cases]}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    zip_path = write_submission_zip(args.output)
    counts = Counter(answer["task_type"] for answer in answers)
    print(f"wrote {args.output}")
    print(f"wrote {cases_path}")
    print(f"wrote {zip_path}")
    print(f"answers={len(answers)} mcq={counts['mcq']} open={counts['open']}")


if __name__ == "__main__":
    main()
