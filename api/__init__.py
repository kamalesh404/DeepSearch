"""DeepSearch API layer: FastAPI routes, middleware, and request/response models."""

from api.routes import router
from api.models import SearchRequest, SearchResponse, SuggestionResponse

__all__ = ["router", "SearchRequest", "SearchResponse", "SuggestionResponse"]
