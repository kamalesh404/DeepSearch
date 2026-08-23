from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime

app = FastAPI(title="DeepSearch API", version="1.0.0", description="AI-Powered Search Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    use_graph: bool = True

class IndexURLRequest(BaseModel):
    url: str
    depth: int = 1

class IndexDocumentRequest(BaseModel):
    content: str
    title: str
    source: str = "upload"

class SearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    source: str

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult]
    graph_nodes: List[dict] = []

indexed_docs: dict = {}
knowledge_graph: dict = {"nodes": [], "edges": []}

@app.get("/")
async def root():
    return {"message": "DeepSearch API", "version": "1.0.0", "docs": "/docs"}

@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    results = []
    for doc_id, doc in indexed_docs.items():
        score = 0.0
        for word in request.query.lower().split():
            if word in doc["content"].lower():
                score += 0.2
        if score > 0:
            results.append(SearchResult(
                id=doc_id,
                title=doc["title"],
                snippet=doc["content"][:200],
                score=round(score, 4),
                source=doc.get("source", "unknown"),
            ))
    results.sort(key=lambda x: x.score, reverse=True)
    answer = f"Found {len(results)} results for '{request.query}'"
    return SearchResponse(
        query=request.query,
        answer=answer,
        sources=results[:request.max_results],
        graph_nodes=knowledge_graph["nodes"][:10],
    )

@app.post("/api/index/url")
async def index_url(request: IndexURLRequest):
    doc_id = str(uuid.uuid4())[:8]
    indexed_docs[doc_id] = {
        "title": f"Page from {request.url}",
        "content": f"Indexed content from {request.url}. This is a placeholder for web crawler output.",
        "source": request.url,
        "indexed_at": datetime.now().isoformat(),
    }
    node_id = f"node_{doc_id}"
    knowledge_graph["nodes"].append({"id": node_id, "label": request.url, "type": "url"})
    return {"id": doc_id, "status": "indexed", "url": request.url}

@app.post("/api/index/document")
async def index_document(request: IndexDocumentRequest):
    doc_id = str(uuid.uuid4())[:8]
    indexed_docs[doc_id] = {
        "title": request.title,
        "content": request.content,
        "source": request.source,
        "indexed_at": datetime.now().isoformat(),
    }
    return {"id": doc_id, "status": "indexed", "title": request.title}

@app.get("/api/graph/explore")
async def explore_graph(node_id: Optional[str] = None):
    if node_id:
        connected = [e for e in knowledge_graph["edges"] if e["source"] == node_id or e["target"] == node_id]
        return {"node_id": node_id, "connections": connected}
    return {"nodes": knowledge_graph["nodes"], "edges": knowledge_graph["edges"]}

@app.get("/api/stats")
async def get_stats():
    return {
        "total_documents": len(indexed_docs),
        "graph_nodes": len(knowledge_graph["nodes"]),
        "graph_edges": len(knowledge_graph["edges"]),
    }
