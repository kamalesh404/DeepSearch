"""Document indexing: inverted index, vector index, and metadata store."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from retriever.chunker import Chunk, Chunker

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Document:
    """A unit of content entering the index."""

    id: str
    text: str
    url: str | None = None
    title: str | None = None
    metadata: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable content hash used for deduplication."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


class InvertedIndex:
    """Term -> posting-list map with document frequencies."""

    def __init__(self) -> None:
        self.postings: dict[str, set[str]] = defaultdict(set)
        self.docs: dict[str, list[str]] = {}

    def add(self, doc_id: str, text: str) -> None:
        tokens = TOKEN_RE.findall(text.lower())
        self.docs[doc_id] = tokens
        self.postings.update({t: self.postings[t] for t in set(tokens)})
        for token in set(tokens):
            self.postings[token].add(doc_id)

    def lookup(self, terms: list[str]) -> dict[str, int]:
        """Score candidates by the number of query terms they contain."""
        hits: dict[str, int] = defaultdict(int)
        for term in terms:
            for doc_id in self.postings.get(term, ()):  # union scoring
                hits[doc_id] += 1
        return hits


class VectorIndex:
    """Flat cosine-similarity vector store; swap for FAISS at scale."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.matrix: np.ndarray | None = None

    @staticmethod
    def _embed(text: str, dim: int) -> np.ndarray:
        """Deterministic hashing embedding placeholder for a real encoder."""
        vec = np.zeros(dim, dtype=np.float32)
        for token in TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % dim
            vec[bucket] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def add(self, doc_id: str, text: str) -> None:
        row = self._embed(text, self.dim).reshape(1, -1)
        self.matrix = row if self.matrix is None else np.vstack([self.matrix, row])
        self.ids.append(doc_id)

    def search(self, text: str, top_k: int = 20) -> dict[str, float]:
        """Return cosine similarities above zero for the nearest rows."""
        if self.matrix is None or not self.ids:
            return {}
        query = self._embed(text, self.dim)
        sims = self.matrix @ query
        order = np.argsort(sims)[::-1][:top_k]
        return {self.ids[i]: float(sims[i]) for i in order if sims[i] > 0}


@dataclass
class IndexManager:
    """Facade combining lexical, vector, and metadata stores over chunks."""

    vector_dim: int = 384
    chunk_size: int = 512

    def __post_init__(self) -> None:
        self.lexical = InvertedIndex()
        self.vectors = VectorIndex(dim=self.vector_dim)
        self.chunk_store: dict[str, dict] = {}
        self.seen_hashes: set[str] = set()
        self.last_updated: datetime | None = None

    def add(self, documents: list[Document]) -> int:
        """Chunk, dedupe, and index each document; returns chunks added."""
        chunker = Chunker(size=self.chunk_size)
        added = 0
        for doc in documents:
            fp = doc.fingerprint()
            if fp in self.seen_hashes:
                continue
            self.seen_hashes.add(fp)
            for chunk in chunker.chunk(doc.id, doc.text):
                key = f"{doc.id}#{chunk.chunk_id}"
                self.lexical.add(key, chunk.text)
                self.vectors.add(key, chunk.text)
                self.chunk_store[key] = {
                    "id": key, "text": chunk.text,
                    "url": doc.url or "", "title": doc.title or "",
                    "metadata": {**doc.metadata, "doc_id": doc.id},
                }
                added += 1
        self.last_updated = datetime.now(timezone.utc)
        return added

    def retrieve(self, terms: list[str], limit: int = 40) -> list[dict]:
        """Hybrid retrieval merging lexical overlap with vector similarity."""
        lex = self.lexical.lookup(terms)
        vec = self.vectors.search(" ".join(terms), top_k=limit)
        fused: dict[str, float] = defaultdict(float)
        for key, count in lex.items():
            fused[key] += 0.5 * count
        for key, sim in vec.items():
            fused[key] += 0.5 * sim
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [self.chunk_store[key] | {"score": score}
                for key, score in ranked if key in self.chunk_store]

    def stats(self) -> dict:
        """Summary snapshot consumed by /status and dashboards."""
        return {
            "documents": len({v["metadata"].get("doc_id") for v in self.chunk_store.values()}),
            "chunks": len(self.chunk_store),
            "vector_ready": self.vectors.matrix is not None,
            "last_updated": self.last_updated,
        }
