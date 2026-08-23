"""Polite web crawler: robots.txt compliance, rate limiting, content extraction."""

from __future__ import annotations

import logging
import time
import urllib.robotparser as robotparser
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

import requests

try:
    import trafilatura  # type: ignore

    HAS_TRAFILATURA = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_TRAFILATURA = False

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "DeepSearchBot/0.1 (+https://example.com/bot)"}


@dataclass
class CrawledPage:
    """Result of fetching and extracting one URL."""

    url: str
    text: str
    status_code: int
    fetched_at: float = field(default_factory=time.time)


class RateLimiter:
    """Per-host minimum-delay throttle to stay polite with origin servers."""

    def __init__(self, min_delay_ms: int = 500) -> None:
        self.min_delay = min_delay_ms / 1000.0
        self._last_hit: dict[str, float] = {}

    def wait(self, host: str) -> None:
        """Block until this host's cooldown has elapsed."""
        elapsed = time.monotonic() - self._last_hit.get(host, 0.0)
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_hit[host] = time.monotonic()


class RobotsCache:
    """Caches parsed robots.txt rules per host."""

    def __init__(self) -> None:
        self._cache: dict[str, robotparser.RobotFileParser] = {}

    def allowed(self, url: str, agent: str = "*") -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            parser = robotparser.RobotFileParser()
            parser.set_url(f"{base}/robots.txt")
            try:
                parser.read()
            except OSError:
                parser.allow_all = True  # unreachable robots.txt -> permissive
            self._cache[base] = parser
        return self._cache[base].can_fetch(agent, url)


class WebCrawler:
    """Breadth-first crawler honoring robots.txt and per-host rate limits."""

    def __init__(self, max_depth: int = 3, max_pages: int = 50,
                 delay_ms: int = 500, timeout_s: float = 10.0) -> None:
        self.max_depth, self.max_pages = max_depth, max_pages
        self.timeout = timeout_s
        self.limiter = RateLimiter(delay_ms)
        self.robots = RobotsCache()
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    @staticmethod
    def _normalize(url: str) -> str:
        """Strip fragments/trailing slashes so the frontier stays deduplicated."""
        parts = urlparse(url)
        path = parts.path.rstrip("/")
        return urlunparse((parts.scheme, parts.netloc, path,
                           parts.params, parts.query, ""))

    def crawl(self, seeds: list[str]) -> list[CrawledPage]:
        """Run BFS from seed URLs, extracting main-body text from each page."""
        frontier: deque[tuple[str, int]] = deque(
            (self._normalize(s), 0) for s in seeds
        )
        visited: set[str] = set()
        pages: list[CrawledPage] = []

        while frontier and len(pages) < self.max_pages:
            url, depth = frontier.popleft()
            if url in visited or not self.robots.allowed(url):
                continue
            visited.add(url)
            host = urlparse(url).netloc
            self.limiter.wait(host)
            page = self._fetch(url)
            if page is None:
                continue
            pages.append(page)
            if depth < self.max_depth:
                frontier.extend(
                    (self._normalize(link), depth + 1)
                    for link in self._extract_links(page.text or "", url)
                )
        logger.info("crawl finished: %d pages", len(pages))
        return pages

    def _fetch(self, url: str) -> CrawledPage | None:
        """Download one URL and extract readable content."""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("fetch failed %s: %s", url, exc)
            return None
        return CrawledPage(url=url, text=self._extract(resp.text), status_code=resp.status_code)

    @staticmethod
    def _extract(html: str) -> str:
        """Prefer trafilatura main-content extraction; regex-strip fallback."""
        if HAS_TRAFILATURA:
            extracted = trafilatura.extract(html)
            if extracted:
                return extracted
        import re
        return re.sub(r"<[^>]+>", " ", html)[:20000]

    @staticmethod
    def _extract_links(text: str, base_url: str) -> list[str]:
        from urllib.parse import urljoin
        import re
        hrefs = re.findall(r'href=["\'](https?://[^"\']+)["\']', text)
        return [urljoin(base_url, h) for h in hrefs][:20]
