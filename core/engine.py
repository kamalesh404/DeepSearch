"""Search engine orchestrator: query parsing, retrieval, ranking, response assembly."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from core.config import get_config
from core.ranking import Ranker
from retriever.indexer import IndexManager

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single ranked result returned to the API layer."""

    doc_id: str
    url: str
    title: str
    snippet: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResponse:
    """Envelope for a full query execution."""

    query: str
    results: list[SearchResult]
    total_ms: float
    total_hits: int


class QueryParser:
    """Minimal query parser supporting quoted phrases, exclusions, and field filters."""

    def parse(self, raw: str) -> dict:
        """Split a raw query into positive terms, phrases, exclusions, and filters."""
        tokens, phrases, excluded, filters = [], [], [], {}
        remaining, i = raw.split(), 0
        while i < len(remaining):
            tok = remaining[i]
            if tok.startswith("-") and len(tok) > 1:
                excluded.append(tok[1:].lower())
            elif tok.startswith("site:"):
                filters["site"] = tok[5:].lower()
            elif '"' in tok and (tok.count('"') == 1):
                phrase = tok.strip('"') + " " + remaining[i + 1].strip('"')
                phrases.append(phrase.lower())
                i += 1
            else:
                tokens.extend(w.lower() for w in tok.split('"') if w)
            i += 1
        return {"terms": tokens, "phrases": phrases,
                "excluded": excluded, "filters": filters}


class SearchEngine:
    """Facade coordinating parsing, candidate retrieval, ranking, and snippets."""

    def __init__(self) -> None:
        self.config = get_config()
        self.parser = QueryParser()
        self.index = IndexManager(
            vector_dim=int(self.config.get("index.vector_dim", 384))
        )
        self.ranker = Ranker(use_neural=bool(self.config.get("search.rerank", True)))

    def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Execute the end-to-end pipeline for one user query."""
        started = time.perf_counter()
        parsed = self.parser.parse(query)
        top_k = top_k or int(self.config.get("search.top_k", 10))

        candidates = self.index.retrieve(parsed["terms"], limit=top_k * 4)
        candidates = [c for c in candidates if not _is_excluded(c["text"], parsed["excluded"])]
        ranked = self.ranker.score(query, candidates)

        min_score = float(self.config.get("search.min_score", 0.05))
        results = []
        for doc, score in ranked[:top_k]:
            if score < min_score:
                break
            results.append(SearchResult(
                doc_id=doc["id"], url=doc.get("url", ""),
                title=doc.get("title", ""), snippet=_snippet(doc["text"], parsed["terms"]),
                score=round(score, 4), metadata=doc.get("metadata", {}),
            ))
        elapsed = (time.perf_counter() - started) * 1000.0
        logger.info("query=%r hits=%d ms=%.1f", query, len(results), elapsed)
        return SearchResponse(query=query, results=results,
                              total_ms=round(elapsed, 2), total_hits=len(results))


def _is_excluded(text: str, excluded: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in excluded)


def _snippet(text: str, terms: list[str], window: int = 220) -> str:
    """Return a short snippet centered on the first matched term."""
    lowered = text.lower()
    cut = next((lowered.find(t) for t in terms if t in lowered), 0)
    start = max(cut - window // 3, 0)
    return ("…" if start > 0 else "") + text[start:start + window].strip()
