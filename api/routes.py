"""FastAPI routes exposing search, suggestions, indexing, and status endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.models import (
    ErrorResponse,
    IndexStatus,
    SearchRequest,
    SuggestionRequest,
    SuggestionResponse,
)
from core.engine import SearchEngine, SearchResponse
from retriever.indexer import Document, IndexManager

logger = logging.getLogger(__name__)
router = APIRouter()

_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    """Lazily build and cache the process-wide search engine."""
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine


def get_index(engine: SearchEngine = Depends(get_engine)) -> IndexManager:
    """Expose the engine's index manager to write endpoints."""
    return engine.index


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["search"],
    summary="Run a ranked search",
)
async def search(payload: SearchRequest, engine: SearchEngine = Depends(get_engine)) -> SearchResponse:
    try:
        return engine.search(payload.query, top_k=payload.top_k)
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.exception("search failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post("/suggest", response_model=SuggestionResponse, tags=["search"])
async def suggest(payload: SuggestionRequest) -> SuggestionResponse:
    """Return autocomplete completions from recent popular queries."""
    corpus = ["deep learning tutorial", "deep search algorithms", "deployment guide",
              "knowledge graph basics", "bm25 ranking", "vector embeddings"]
    lowered = payload.prefix.lower()
    matches = sorted(q for q in corpus if q.startswith(lowered))[:10]
    return SuggestionResponse(prefix=payload.prefix, suggestions=matches)


@router.post("/index", status_code=status.HTTP_202_ACCEPTED, tags=["admin"])
async def add_document(doc: dict, index: IndexManager = Depends(get_index)) -> dict:
    """Ingest one document into the index (fire-and-forget style)."""
    required = {"id", "text"}
    if not required.issubset(doc):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"missing fields: {required - set(doc)}")
    document = Document(id=str(doc["id"]), text=doc["text"],
                        url=doc.get("url"), title=doc.get("title"),
                        metadata=doc.get("metadata", {}))
    index.add([document])
    return {"queued": True, "doc_id": document.id}


@router.get("/status", response_model=IndexStatus, tags=["admin"])
async def status_endpoint(index: IndexManager = Depends(get_index)) -> IndexStatus:
    """Report index health for load balancers and dashboards."""
    stats = index.stats()
    return IndexStatus(documents=stats["documents"], chunks=stats["chunks"],
                       vector_index_ready=bool(stats["vector_ready"]),
                       last_updated=stats.get("last_updated"))
