# backend/ingestion/connectors/hackernews.py
"""
HackerNews connector via Algolia API.
Zero auth required. Best signal-to-noise for B2B tech intelligence.
Catches developer sentiment, product criticism, competitor buzz early.
API docs: https://hn.algolia.com/api
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx
import structlog

from backend.ingestion.base_connector import BaseConnector, RawSignal

logger = structlog.get_logger(__name__)

# HN signal scoring — points and comments indicate community importance
def _compute_urgency_from_hn(hit: dict, entity_hints: list[str]) -> float:
    """
    HN-specific urgency scoring.
    Points + comment count = community signal strength.
    """
    urgency = 0.15  # HN is slow-burn signal, lower base

    points = hit.get("points") or 0
    comments = hit.get("num_comments") or 0

    # High engagement = higher urgency
    if points > 100:
        urgency += 0.3
    elif points > 50:
        urgency += 0.2
    elif points > 20:
        urgency += 0.1

    if comments > 50:
        urgency += 0.2
    elif comments > 20:
        urgency += 0.1

    # Direct entity mention
    if entity_hints:
        urgency += 0.15

    return min(urgency, 1.0)


def _detect_hn_signal_type(text: str) -> str:
    """Classify HN signal based on content."""
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["ask hn", "ask:", "how do you", "what do you"]):
        return "community_discussion"
    if any(kw in text_lower for kw in ["show hn", "i built", "we built", "launch"]):
        return "product_launch"
    if any(kw in text_lower for kw in ["hiring", "we're hiring", "job"]):
        return "hiring"
    if any(kw in text_lower for kw in ["funding", "raises", "series", "acquired"]):
        return "funding"
    if any(kw in text_lower for kw in ["down", "outage", "broken", "critical bug"]):
        return "incident"

    return "developer_sentiment"


def _build_hn_content(hit: dict) -> str:
    """Build content string from HN hit fields."""
    parts = []

    title = hit.get("title") or ""
    story_text = hit.get("story_text") or ""
    comment_text = hit.get("comment_text") or ""
    url = hit.get("url") or ""
    author = hit.get("author") or ""
    points = hit.get("points") or 0
    comments = hit.get("num_comments") or 0

    if title:
        parts.append(f"TITLE: {title}")
    if author:
        parts.append(f"AUTHOR: {author}")
    if points or comments:
        parts.append(f"ENGAGEMENT: {points} points, {comments} comments")
    if url:
        parts.append(f"URL: {url}")
    if story_text:
        parts.append(f"CONTENT: {story_text[:1000]}")
    elif comment_text:
        parts.append(f"COMMENT: {comment_text[:1000]}")

    return "\n".join(parts)[:3000]


class HackerNewsConnector(BaseConnector):
    """
    Fetches HackerNews stories and comments mentioning watched entities.
    Uses Algolia search API — zero auth, generous rate limits.

    Config keys:
        watch_entities (list[str]): Terms to search for
        min_points (int): Minimum points to include, default 5
        include_comments (bool): Include comment hits, default True
        rate_limit_per_minute (int): default 30 (Algolia is generous)
    """

    ALGOLIA_URL = "https://hn.algolia.com/api/v1/search"

    def __init__(self, config: dict):
        config.setdefault("rate_limit_per_minute", 30)
        super().__init__(config)

        self.watch_entities: list[str] = config.get("watch_entities", [])
        if not self.watch_entities:
            entities_env = __import__("os").getenv("VERIDIAN_WATCH_ENTITIES", "")
            self.watch_entities = [
                e.strip() for e in entities_env.split(",") if e.strip()
            ]

        self.min_points = config.get("min_points", 5)
        self.include_comments = config.get("include_comments", True)

        logger.info(
            "hackernews.initialized",
            watch_entities=self.watch_entities,
            min_points=self.min_points,
        )

    async def fetch(self, since: datetime) -> AsyncIterator[RawSignal]:
        """
        Search HN for stories and comments mentioning watched entities.
        Filters by timestamp and minimum points threshold.
        """
        if not self.watch_entities:
            logger.warning("hackernews.no_watch_entities")
            return

        seen_ids: set[str] = set()
        since_timestamp = int(since.timestamp())

        async with httpx.AsyncClient(timeout=30.0) as client:
            for entity in self.watch_entities:
                await self.rate_limiter.acquire()

                # Search stories + comments
                tags = "story,comment" if self.include_comments else "story"

                params = {
                    "query": entity,
                    "tags": tags,
                    "hitsPerPage": 50,
                }
                logger.debug("hackernews.fetching", entity=entity)

                try:
                    response = await client.get(self.ALGOLIA_URL, params=params)
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.error(
                        "hackernews.http_error",
                        entity=entity,
                        status=e.response.status_code,
                    )
                    raise

                data = response.json()
                hits = data.get("hits", [])

                logger.info(
                    "hackernews.hits_found",
                    entity=entity,
                    count=len(hits),
                )

                for hit in hits:
                    object_id = str(hit.get("objectID", ""))
                    if not object_id or object_id in seen_ids:
                        continue
                    created_at_i = hit.get("created_at_i") or 0
                    if created_at_i < since_timestamp:
                        continue

                    seen_ids.add(object_id)

                    # Filter low-signal hits
                    points = hit.get("points") or 0
                    if points < self.min_points and hit.get("_tags", []) == ["story"]:
                        continue

                    # Build content
                    content = _build_hn_content(hit)
                    if not content.strip():
                        continue

                    # Parse timestamp
                    created_at_i = hit.get("created_at_i") or 0
                    try:
                        timestamp = datetime.fromtimestamp(
                            created_at_i, tz=timezone.utc
                        )
                    except (ValueError, OSError):
                        timestamp = datetime.now(timezone.utc)

                    # Extract entity hints from title + text
                    full_text = (hit.get("title") or "") + " " + (
                        hit.get("story_text") or hit.get("comment_text") or ""
                    )
                    entity_hints = [
                        e for e in self.watch_entities
                        if e.lower() in full_text.lower()
                    ]

                    signal_type = _detect_hn_signal_type(content)

                    # Build stable source_id
                    source_id = hashlib.sha256(
                        f"hn_{object_id}".encode()
                    ).hexdigest()[:16]

                    yield RawSignal(
                        source="hackernews",
                        source_id=source_id,
                        raw_content=content,
                        url=f"https://news.ycombinator.com/item?id={object_id}",
                        timestamp=timestamp,
                        entity_hints=entity_hints,
                        signal_category="community",
                        signal_type=signal_type,
                        urgency_score=_compute_urgency_from_hn(hit, entity_hints),
                        metadata={
                            "hn_id": object_id,
                            "points": points,
                            "num_comments": hit.get("num_comments") or 0,
                            "author": hit.get("author"),
                            "queried_entity": entity,
                        },
                    )

    async def health_check(self) -> bool:
        """Verify Algolia HN API is reachable."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    self.ALGOLIA_URL,
                    params={"query": "test", "hitsPerPage": 1},
                )
                healthy = response.status_code == 200
                logger.info("hackernews.health_check", healthy=healthy)
                return healthy
            except Exception as e:
                logger.error("hackernews.health_check_failed", error=str(e))
                return False