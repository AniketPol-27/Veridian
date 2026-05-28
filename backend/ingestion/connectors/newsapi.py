# backend/ingestion/connectors/newsapi.py
"""
NewsAPI connector — real-time news from 150,000+ sources.
Watches for competitor mentions, market signals, industry news.
API docs: https://newsapi.org/docs
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
import structlog

from backend.ingestion.base_connector import BaseConnector, RawSignal

logger = structlog.get_logger(__name__)

# Signal type mapping based on content keywords
SIGNAL_TYPE_KEYWORDS = {
    "hiring": ["hiring", "recrui", "job opening", "we're growing", "join our team"],
    "funding": ["raises", "funding", "series", "investment", "valuation", "ipo"],
    "product_launch": ["launches", "announces", "release", "new product", "unveils"],
    "partnership": ["partnership", "integration", "collaborat", "acqui"],
    "leadership": ["ceo", "cto", "appoints", "names", "resigns", "departs", "joins as"],
    "competitive": ["competitor", "rival", "market share", "beats", "outperforms"],
    "financial": ["revenue", "profit", "earnings", "quarterly", "annual report"],
    "regulatory": ["regulation", "lawsuit", "fine", "compliance", "ftc", "gdpr"],
}


def _detect_signal_type(text: str) -> str:
    """Detect signal type from article text. Returns most specific match."""
    text_lower = text.lower()
    for signal_type, keywords in SIGNAL_TYPE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return signal_type
    return "general_news"


def _compute_urgency(article: dict, entity_hints: list[str]) -> float:
    """
    Compute urgency score 0-1.
    Higher urgency = more time-sensitive signal.
    """
    urgency = 0.2  # Base urgency for all news

    # Direct competitor mention → higher urgency
    if entity_hints:
        urgency += 0.2

    # Breaking/recent news keywords
    title = (article.get("title") or "").lower()
    if any(kw in title for kw in ["breaking", "just in", "urgent", "alert"]):
        urgency += 0.2

    # High-stakes signal types
    signal_type = _detect_signal_type(
        (article.get("title") or "") + " " + (article.get("description") or "")
    )
    if signal_type in ("funding", "leadership", "regulatory"):
        urgency += 0.2

    if signal_type in ("product_launch", "partnership"):
        urgency += 0.1

    return min(urgency, 1.0)


def _extract_entity_hints(text: str, watch_entities: list[str]) -> list[str]:
    """Find which watched entities are mentioned in the text."""
    text_lower = text.lower()
    return [e for e in watch_entities if e.lower() in text_lower]


def _build_content(article: dict) -> str:
    """Build rich content string from article fields."""
    parts = []

    title = article.get("title") or ""
    description = article.get("description") or ""
    content = article.get("content") or ""
    source_name = (article.get("source") or {}).get("name") or ""

    if title:
        parts.append(f"TITLE: {title}")
    if source_name:
        parts.append(f"SOURCE: {source_name}")
    if description:
        parts.append(f"DESCRIPTION: {description}")
    if content:
        # NewsAPI truncates content at ~200 chars with "[+N chars]"
        clean_content = content.split("[+")[0].strip()
        if clean_content and clean_content != description:
            parts.append(f"CONTENT: {clean_content}")

    return "\n".join(parts)[:3000]  # Cap at 3000 chars


class NewsAPIConnector(BaseConnector):
    """
    Fetches news articles from NewsAPI.
    Watches for mentions of configured entities (competitors, partners, etc).

    Config keys:
        api_key (str): NewsAPI key — required
        watch_entities (list[str]): Company/product names to monitor
        language (str): Article language filter, default "en"
        rate_limit_per_minute (int): API calls per minute, default 5
    """

    BASE_URL = "https://newsapi.org/v2/everything"
    HEADLINES_URL = "https://newsapi.org/v2/top-headlines"

    def __init__(self, config: dict):
        # Set conservative rate limit — NewsAPI free tier = 100 req/day
        config.setdefault("rate_limit_per_minute", 5)
        super().__init__(config)

        self.api_key = config.get("api_key") or os.getenv("NEWSAPI_KEY")
        if not self.api_key:
            raise ValueError("NewsAPI key required. Set NEWSAPI_KEY in .env")

        self.watch_entities: list[str] = config.get("watch_entities", [])
        if not self.watch_entities:
            # Fall back to env var
            entities_env = os.getenv("VERIDIAN_WATCH_ENTITIES", "")
            self.watch_entities = [
                e.strip() for e in entities_env.split(",") if e.strip()
            ]

        self.language = config.get("language", "en")
        self.page_size = min(config.get("page_size", 20), 100)

        logger.info(
            "newsapi.initialized",
            watch_entities=self.watch_entities,
            language=self.language,
        )

    async def fetch(self, since: datetime) -> AsyncIterator[RawSignal]:
        """
        Fetch articles mentioning watched entities since the given datetime.
        Yields one RawSignal per article (deduplicated by URL hash).
        """
        if not self.watch_entities:
            logger.warning("newsapi.no_watch_entities", msg="Nothing to watch")
            return

        # Track seen URLs to avoid duplicate signals across queries
        seen_urls: set[str] = set()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for entity in self.watch_entities:
                await self.rate_limiter.acquire()

                # Format since as ISO 8601 — NewsAPI requires this
                since_str = since.strftime("%Y-%m-%dT%H:%M:%S")

                params = {
                    "q": entity,          
                    "language": self.language,
                    "sortBy": "publishedAt",
                    "pageSize": self.page_size,
                    "apiKey": self.api_key,
  
                }

                logger.debug(
                    "newsapi.fetching",
                    entity=entity,
                    since=since_str,
                )

                try:
                    response = client.get(self.BASE_URL, params=params)
                    response = await response
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "newsapi.http_error",
                        entity=entity,
                        status=e.response.status_code,
                        body=e.response.text[:200],
                    )
                    raise

                data = response.json()

                if data.get("status") != "ok":
                    logger.error(
                        "newsapi.api_error",
                        entity=entity,
                        error=data.get("message"),
                        code=data.get("code"),
                    )
                    continue

                articles = data.get("articles", [])
                logger.info(
                    "newsapi.articles_found",
                    entity=entity,
                    count=len(articles),
                )

                for article in articles:
                    # Skip articles with no content
                    title = article.get("title") or ""
                    if not title or title == "[Removed]":
                        continue

                    # Deduplicate by URL
                    url = article.get("url") or ""
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Build stable source_id from URL hash
                    source_id = hashlib.sha256(url.encode()).hexdigest()[:16]

                    # Build full content string
                    content = _build_content(article)
                    if not content.strip():
                        continue

                    # Parse timestamp
                    published_at = article.get("publishedAt") or ""
                    try:
                        timestamp = datetime.fromisoformat(
                            published_at.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        timestamp = datetime.now(timezone.utc)

                    # Extract which entities are mentioned
                    entity_hints = _extract_entity_hints(
                        title + " " + (article.get("description") or ""),
                        self.watch_entities,
                    )

                    # Detect signal type
                    signal_type = _detect_signal_type(content)

                    yield RawSignal(
                        source="newsapi",
                        source_id=source_id,
                        raw_content=content,
                        url=url,
                        timestamp=timestamp,
                        entity_hints=entity_hints,
                        signal_category="news",
                        signal_type=signal_type,
                        urgency_score=_compute_urgency(article, entity_hints),
                        metadata={
                            "source_name": (article.get("source") or {}).get("name"),
                            "author": article.get("author"),
                            "queried_entity": entity,
                        },
                    )

    async def health_check(self) -> bool:
        """Verify NewsAPI is reachable and key is valid."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    self.BASE_URL,
                    params={
                        "q": "test",
                        "pageSize": 1,
                        "apiKey": self.api_key,
                    },
                )
                data = response.json()
                healthy = data.get("status") == "ok"
                logger.info("newsapi.health_check", healthy=healthy)
                return healthy
            except Exception as e:
                logger.error("newsapi.health_check_failed", error=str(e))
                return False