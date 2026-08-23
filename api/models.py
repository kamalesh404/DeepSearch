"""Pydantic models defining the public API contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Incoming search query with optional pagination and filters."""

    query: str = Field(..., min_length=1, max_length=512, description="User search query")
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    include_graph: bool = Field(False, description="Attach related knowledge-graph entities")
    filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters")

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class ResultItem(BaseModel):
    """One ranked hit inside a SearchResponse."""

    doc_id: str
    url: str
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Outgoing response envelope for /search."""

    query: str
    results: list[ResultItem]
    total_hits: int
    took_ms: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class SuggestionRequest(BaseModel):
    """Prefix for autocomplete suggestions."""

    prefix: str = Field(..., min_length=1, max_length=128)

    @field_validator("prefix")
    @classmethod
    def strip_prefix(cls, value: str) -> str:
        return value.strip()


class SuggestionResponse(BaseModel):
    """Ranked completion candidates for a query prefix."""

    prefix: str
    suggestions: list[str] = Field(max_length=10)


class IndexStatus(BaseModel):
    """Health and size snapshot of the index."""

    documents: int
    chunks: int
    vector_index_ready: bool
    last_updated: datetime | None = None


class ErrorResponse(BaseModel):
    """Uniform error payload."""

    error: str
    detail: str | None = None
