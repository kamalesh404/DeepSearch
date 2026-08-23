"""Ranking algorithms: BM25, TF-IDF, neural reranking hooks, hybrid scoring."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RankStats:
    """Corpus-level statistics shared by scoring functions."""

    avg_doc_len: float = 1.0
    doc_freq: Counter = field(default_factory=Counter)
    total_docs: int = 0


def bm25_score(query_terms: list[str], doc_tokens: list[str],
               stats: RankStats, k1: float = 1.5, b: float = 0.75) -> float:
    """Okapi BM25 relevance score for a single document."""
    tf = Counter(doc_tokens)
    score, doc_len = 0.0, max(len(doc_tokens), 1)
    for term in query_terms:
        df = stats.doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1 + (stats.total_docs - df + 0.5) / (df + 0.5))
        freq = tf.get(term, 0)
        denom = freq + k1 * (1 - b + b * doc_len / stats.avg_doc_len)
        score += idf * (freq * (k1 + 1)) / denom
    return score


def tfidf_score(query_terms: list[str], doc_tokens: list[str], stats: RankStats) -> float:
    """Classic TF-IDF cosine-style linear score."""
    tf = Counter(doc_tokens)
    score = 0.0
    for term in query_terms:
        df = stats.doc_freq.get(term, 0)
        idf = math.log((stats.total_docs + 1) / (df + 1)) + 1
        score += (tf.get(term, 0) / max(len(doc_tokens), 1)) * idf
    return score


class NeuralReranker:
    """Optional cross-encoder hook; falls back to lexical score when unavailable."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model = None
        self.model_name = model_name

    def _load(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError:
            return False
        if self._model is None:
            self._model = CrossEncoder(self.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
        return True

    def rerank(self, query: str, docs: list[dict]) -> list[tuple[dict, float]]:
        if not docs or not self._load():
            return [(d, d.get("score", 0.0)) for d in docs]
        pairs = [(query, d["text"]) for d in docs]
        scores = self._model.predict(pairs)
        return list(zip(docs, map(float, scores)))


class Ranker:
    """Hybrid ranker blending BM25 and TF-IDF with an optional neural pass."""

    def __init__(self, bm25_weight: float = 0.7, use_neural: bool = True) -> None:
        self.bm25_weight = bm25_weight
        self.stats = RankStats()
        self.neural = NeuralReranker() if use_neural else None

    def update_stats(self, documents: list[list[str]]) -> None:
        """Refresh corpus statistics from tokenized documents."""
        self.stats = RankStats(total_docs=len(documents))
        lengths = []
        for tokens in documents:
            lengths.append(len(tokens))
            self.stats.doc_freq.update(set(tokens))
        self.stats.avg_doc_len = sum(lengths) / max(len(lengths), 1)

    def score(self, query: str, candidates: list[dict]) -> list[tuple[dict, float]]:
        """Score, optionally rerank, and sort candidates descending."""
        terms = query.lower().split()
        blended: list[tuple[dict, float]] = []
        for cand in candidates:
            tokens = cand["text"].lower().split()
            s_bm25 = bm25_score(terms, tokens, self.stats)
            s_tfidf = tfidf_score(terms, tokens, self.stats)
            cand["score"] = self.bm25_weight * s_bm25 + (1 - self.bm25_weight) * s_tfidf
            blended.append(cand)
        if self.neural is not None:
            reranked = self.neural.rerank(query, blended)
        else:
            reranked = [(c, c["score"]) for c in blended]
        return sorted(reranked, key=lambda pair: pair[1], reverse=True)
