"""DeepSearch core: search engine orchestration, configuration, and ranking."""

from core.engine import SearchEngine, SearchResult
from core.config import Config, get_config
from core.ranking import Ranker

__all__ = ["SearchEngine", "SearchResult", "Config", "get_config", "Ranker"]
__version__ = "0.1.0"
