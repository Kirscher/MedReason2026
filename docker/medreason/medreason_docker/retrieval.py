from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")
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


def tokenize(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS]


def text_for_example(example: dict[str, Any]) -> str:
    options = example.get("options") or []
    option_text = " ".join(str(opt.get("text", "")) for opt in options if isinstance(opt, dict))
    metadata = example.get("metadata") or {}
    metadata_text = " ".join(str(v) for v in metadata.values())
    return " ".join(
        [
            str(example.get("question", "")),
            str(example.get("visual_description", "")),
            option_text,
            metadata_text,
        ]
    )


def option_text(example: dict[str, Any], label: str) -> str:
    label = str(label).strip().upper()
    for option in example.get("options") or []:
        if not isinstance(option, dict):
            continue
        if str(option.get("label", "")).strip().upper() == label:
            return str(option.get("text", "")).strip()
    return ""


def correct_answer_text_for_example(example: dict[str, Any]) -> str:
    return option_text(example, str(example.get("answer", "")))


def text_for_correct_answer_example(example: dict[str, Any]) -> str:
    metadata = example.get("metadata") or {}
    metadata_text = " ".join(str(v) for v in metadata.values())
    return " ".join(
        [
            str(example.get("question", "")),
            str(example.get("visual_description", "")),
            correct_answer_text_for_example(example),
            metadata_text,
        ]
    )


@dataclass(frozen=True)
class SearchHit:
    example: dict[str, Any]
    score: float


class _TfidfIndex:
    def __init__(self, examples: Iterable[dict[str, Any]], text_fn) -> None:
        self.examples = list(examples)
        self.doc_terms: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for example in self.examples:
            terms = Counter(tokenize(text_fn(example)))
            self.doc_terms.append(terms)
            df.update(terms.keys())

        self.n_docs = len(self.doc_terms)
        self.idf = {
            term: math.log((1 + self.n_docs) / (1 + freq)) + 1.0
            for term, freq in df.items()
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

    def search(self, query: str, task_type: str | None = None, top_k: int = 3) -> list[SearchHit]:
        if not self.examples:
            return []
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
                example = self.examples[doc_id]
                if task_type and example.get("task_type") != task_type:
                    continue
                scores[doc_id] += q_weight * doc_weight

        ranked = [
            SearchHit(example=self.examples[doc_id], score=score / (q_norm * self.norms[doc_id]))
            for doc_id, score in scores.items()
        ]
        ranked.sort(key=lambda hit: hit.score, reverse=True)
        return ranked[:top_k]


class RetrievalBank:
    def __init__(self, examples: Iterable[dict[str, Any]]) -> None:
        self.examples = list(examples)
        self.default_index = _TfidfIndex(self.examples, text_for_example)
        self.correct_answer_examples = [
            example
            for example in self.examples
            if example.get("task_type") == "mcq" and correct_answer_text_for_example(example)
        ]
        self.correct_answer_index = _TfidfIndex(
            self.correct_answer_examples, text_for_correct_answer_example
        )

    @classmethod
    def from_json(cls, path: Path) -> "RetrievalBank":
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = payload.get("examples", payload)
        if not isinstance(examples, list):
            raise ValueError(f"Retrieval bank must be a list or contain an examples list: {path}")
        return cls(examples)

    def search(self, query: str, task_type: str | None = None, top_k: int = 3) -> list[SearchHit]:
        return self.default_index.search(query, task_type=task_type, top_k=top_k)

    def search_correct_answer_examples(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self.correct_answer_index.search(query, task_type="mcq", top_k=top_k)
