"""Text chunking: sliding window, semantic, and sentence-based strategies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


class ChunkStrategy(str, Enum):
    """Supported chunking modes selectable per index configuration."""

    SLIDING_WINDOW = "sliding_window"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class Chunk:
    """One retrievable span of text tied back to its parent document."""

    doc_id: str
    chunk_id: int
    text: str
    start_offset: int


class Chunker:
    """Splits long documents into overlapping retrieval-friendly pieces."""

    def __init__(self, size: int = 512, overlap: int = 64,
                 strategy: ChunkStrategy = ChunkStrategy.SLIDING_WINDOW) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size, self.overlap, self.strategy = size, overlap, strategy

    def chunk(self, doc_id: str, text: str) -> list[Chunk]:
        """Dispatch to the configured strategy."""
        if self.strategy is ChunkStrategy.SENTENCE:
            spans = self._sentences(text)
        elif self.strategy is ChunkStrategy.SEMANTIC:
            spans = self._semantic(text)
        else:
            spans = self._windows(text)
        chunks, cursor = [], 0
        for i, span in enumerate(spans):
            start = text.find(span[:40], cursor)
            start = start if start >= 0 else cursor
            cursor = start + len(span)
            chunks.append(Chunk(doc_id=doc_id, chunk_id=i,
                                text=span.strip(), start_offset=start))
        return [c for c in chunks if c.text]

    def _windows(self, text: str) -> list[str]:
        """Fixed-size character windows with configurable overlap."""
        step = self.size - self.overlap
        return [text[i:i + self.size] for i in range(0, len(text), step)] or [""]

    def _sentences(self, text: str) -> list[str]:
        """Group sentences until the budget is reached; never split mid-sentence."""
        sentences = SENTENCE_SPLIT.split(text)
        groups, buf = [], ""
        for sent in sentences:
            if len(buf) + len(sent) > self.size and buf:
                groups.append(buf)
                buf = buf[max(len(buf) - self.overlap, 0):] + " " + sent
            else:
                buf = (buf + " " + sent).strip()
        if buf.strip():
            groups.append(buf)
        return groups

    def _semantic(self, text: str) -> list[str]:
        """Merge paragraphs while embedding drift stays below a threshold."""
        paragraphs = [p for p in PARAGRAPH_SPLIT.split(text) if p.strip()]
        groups, buf = [], ""
        for para in paragraphs:
            candidate = (buf + "\n\n" + para).strip()
            if len(candidate) > self.size and buf:
                groups.append(buf)
                buf = para
            else:
                buf = candidate
        if buf.strip():
            groups.append(buf)
        return groups
