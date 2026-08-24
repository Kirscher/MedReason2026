from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


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


@dataclass(frozen=True)
class SearchResult:
    doc_id: int
    score: float


class TfidfIndex:
    def __init__(self, docs: Iterable[str]) -> None:
        self.doc_terms: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for doc in docs:
            terms = Counter(tokenize(doc))
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

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        q_terms = Counter(tokenize(query))
        if not q_terms:
            return []

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
            SearchResult(doc_id=doc_id, score=score / (q_norm * self.norms[doc_id]))
            for doc_id, score in scores.items()
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
