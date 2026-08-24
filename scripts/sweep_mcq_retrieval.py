from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

OPTION_LABELS = ("A", "B", "C", "D", "E")


def load_items(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get(key, payload)
    if not isinstance(items, list):
        raise ValueError(f"{path} must be a list or contain {key}")
    return items


def tokenize(text: str) -> list[str]:
    import re

    stopwords = {
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
        "given",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "to",
        "what",
        "which",
        "with",
    }
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if tok not in stopwords]


def option_text(item: dict[str, Any], label: str) -> str:
    label = label.strip().upper()
    for option in item.get("options") or []:
        if isinstance(option, dict) and str(option.get("label", "")).strip().upper() == label:
            return str(option.get("text", "")).strip()
    return ""


def correct_answer_text(item: dict[str, Any]) -> str:
    return option_text(item, str(item.get("answer", "")))


def metadata_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return " ".join(str(value) for value in metadata.values() if value)


def all_option_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(option.get("text", "")) for option in item.get("options") or [] if isinstance(option, dict)
    )


def text_for_index(item: dict[str, Any], mode: str) -> str:
    if mode == "answer_only":
        return correct_answer_text(item)
    if mode == "answer_metadata":
        return " ".join([correct_answer_text(item), metadata_text(item)])
    if mode == "question_answer_metadata":
        return " ".join([str(item.get("question", "")), correct_answer_text(item), metadata_text(item)])
    if mode == "question_visual_answer_metadata":
        return " ".join(
            [
                str(item.get("question", "")),
                str(item.get("visual_description", "")),
                correct_answer_text(item),
                metadata_text(item),
            ]
        )
    if mode == "full_all_options":
        return " ".join(
            [
                str(item.get("question", "")),
                str(item.get("visual_description", "")),
                all_option_text(item),
                metadata_text(item),
            ]
        )
    raise ValueError(f"unknown index mode: {mode}")


def text_for_query(case: dict[str, Any], label: str, mode: str) -> str:
    text = option_text(case, label)
    if mode == "option_only":
        return text
    if mode == "option_metadata":
        return " ".join([text, metadata_text(case)])
    if mode == "question_option":
        return " ".join([str(case.get("question", "")), text])
    if mode == "question_option_metadata":
        return " ".join([str(case.get("question", "")), text, metadata_text(case)])
    if mode == "question_alloptions_metadata":
        return " ".join([str(case.get("question", "")), all_option_text(case), metadata_text(case)])
    raise ValueError(f"unknown query mode: {mode}")


@dataclass(frozen=True)
class Hit:
    item: dict[str, Any]
    score: float


class TfidfIndex:
    def __init__(self, examples: list[dict[str, Any]], text_fn: Callable[[dict[str, Any]], str]) -> None:
        self.examples = examples
        self.doc_terms: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for example in examples:
            terms = Counter(tokenize(text_fn(example)))
            self.doc_terms.append(terms)
            df.update(terms.keys())
        self.idf = {
            term: math.log((1 + len(self.doc_terms)) / (1 + freq)) + 1.0 for term, freq in df.items()
        }
        self.inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.norms: list[float] = []
        for doc_id, terms in enumerate(self.doc_terms):
            norm_sq = 0.0
            for term, tf in terms.items():
                weight = (1.0 + math.log(tf)) * self.idf[term]
                self.inverted[term].append((doc_id, weight))
                norm_sq += weight * weight
            self.norms.append(math.sqrt(norm_sq) or 1.0)

    def search(self, query: str, top_k: int) -> list[Hit]:
        q_terms = Counter(tokenize(query))
        q_weights: dict[str, float] = {}
        q_norm_sq = 0.0
        for term, tf in q_terms.items():
            if term not in self.idf:
                continue
            weight = (1.0 + math.log(tf)) * self.idf[term]
            q_weights[term] = weight
            q_norm_sq += weight * weight
        q_norm = math.sqrt(q_norm_sq) or 1.0
        scores: dict[int, float] = defaultdict(float)
        for term, q_weight in q_weights.items():
            for doc_id, doc_weight in self.inverted.get(term, ()):
                scores[doc_id] += q_weight * doc_weight
        ranked = [
            Hit(item=self.examples[doc_id], score=score / (q_norm * self.norms[doc_id]))
            for doc_id, score in scores.items()
        ]
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:top_k]


def score_hits(hits: list[Hit], mode: str) -> float:
    if not hits:
        return 0.0
    scores = [max(hit.score, 0.0) for hit in hits]
    if mode == "top1":
        return scores[0]
    if mode == "sum3":
        return sum(scores[:3])
    if mode == "sum5":
        return sum(scores[:5])
    if mode == "top1_plus_sum5":
        return scores[0] + sum(scores[:5])
    if mode == "ranked5":
        return sum(score / (rank + 1) for rank, score in enumerate(scores[:5]))
    raise ValueError(f"unknown score mode: {mode}")


def evaluate_policy(
    *,
    index: TfidfIndex,
    cases: list[dict[str, Any]],
    truth: dict[str, str],
    query_mode: str,
    score_mode: str,
    top_k: int,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for case in cases:
        if str(case.get("task_type", "")).lower() != "mcq":
            continue
        scores: dict[str, float] = {}
        for option in case.get("options") or []:
            label = str(option.get("label", "")).strip().upper()
            if not label:
                continue
            hits = index.search(text_for_query(case, label, query_mode), top_k=top_k)
            scores[label] = score_hits(hits, score_mode)
        if not scores:
            continue
        prediction = max(scores, key=scores.get)
        correct += prediction == truth[str(case["case_id"])]
        total += 1
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep MCQ retrieval policies on labeled holdouts.")
    parser.add_argument("--retrieval-bank", type=Path, required=True)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--index-modes", nargs="*", default=None)
    parser.add_argument("--query-modes", nargs="*", default=None)
    parser.add_argument("--score-modes", nargs="*", default=None)
    parser.add_argument("--top-ks", nargs="*", type=int, default=None)
    args = parser.parse_args()

    examples = [
        example
        for example in load_items(args.retrieval_bank, "examples")
        if example.get("task_type") == "mcq" and correct_answer_text(example)
    ]
    cases = load_items(args.cases_json, "cases")
    truth = {
        str(item["case_id"]): str(item.get("answer", "")).strip().upper()
        for item in load_items(args.ground_truth, "answers")
    }

    index_modes = args.index_modes or [
        "answer_only",
        "answer_metadata",
        "question_answer_metadata",
        "question_visual_answer_metadata",
        "full_all_options",
    ]
    query_modes = args.query_modes or [
        "option_only",
        "option_metadata",
        "question_option",
        "question_option_metadata",
        "question_alloptions_metadata",
    ]
    score_modes = args.score_modes or ["top1", "sum3", "sum5", "top1_plus_sum5", "ranked5"]
    top_ks = args.top_ks or [1, 3, 5, 8, 12, 20]

    results: list[dict[str, Any]] = []
    max_top_k = max(top_ks)
    mcq_cases = [case for case in cases if str(case.get("task_type", "")).lower() == "mcq"]

    for index_mode in index_modes:
        index = TfidfIndex(examples, lambda item, mode=index_mode: text_for_index(item, mode))
        for query_mode in query_modes:
            cached_hits: dict[tuple[str, str], list[Hit]] = {}
            for case in mcq_cases:
                for option in case.get("options") or []:
                    label = str(option.get("label", "")).strip().upper()
                    if label:
                        cached_hits[(str(case["case_id"]), label)] = index.search(
                            text_for_query(case, label, query_mode), top_k=max_top_k
                        )

            for score_mode in score_modes:
                for top_k in top_ks:
                    correct = 0
                    total = 0
                    for case in mcq_cases:
                        scores: dict[str, float] = {}
                        for option in case.get("options") or []:
                            label = str(option.get("label", "")).strip().upper()
                            if not label:
                                continue
                            hits = cached_hits.get((str(case["case_id"]), label), [])[:top_k]
                            scores[label] = score_hits(hits, score_mode)
                        if not scores:
                            continue
                        prediction = max(scores, key=scores.get)
                        correct += prediction == truth[str(case["case_id"])]
                        total += 1
                    results.append(
                        {
                            "index_mode": index_mode,
                            "query_mode": query_mode,
                            "score_mode": score_mode,
                            "top_k": top_k,
                            "correct": correct,
                            "total": total,
                            "accuracy": correct / total if total else None,
                        }
                    )

    results.sort(key=lambda item: (item["accuracy"], item["correct"]), reverse=True)
    for row in results[: args.topn]:
        print(json.dumps(row, sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
