# backend/ingestion/base_connector.py
"""
Base connector class for all Veridian signal sources.
Every connector implements fetch() and health_check().
Auth, rate limiting, retry — written once here, never duplicated.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator

import httpx
import structlog

logger = structlog.get_logger(__name__)


# ============================================================
# THE UNIVERSAL SIGNAL SCHEMA
# Every connector outputs this. No exceptions.
# ============================================================

@dataclass
class RawSignal:
    """Universal signal schema. Every connector outputs this."""

    source: str                    # "newsapi" | "hackernews" | "hubspot" | etc
    source_id: str                 # Unique ID within that source
    raw_content: str               # The actual text Oracle reasons about
    timestamp: datetime            # When this signal occurred in the world
    entity_hints: list[str]        # Company names, people, products extracted
    signal_category: str           # "news"|"community"|"internal"|"financial"|"hiring"
    signal_type: str               # "competitor_news"|"churn_signal"|"hiring"|etc
    urgency_score: float           # 0-1: How time-sensitive is this?
    url: str | None = None         # Original URL if applicable
    metadata: dict = field(default_factory=dict)  # Source-specific extras

    def __post_init__(self):
        """Validate scores are in range."""
        for attr in ["urgency_score"]:
            val = getattr(self, attr)
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {val}")

        if not self.raw_content or not self.raw_content.strip():
            raise ValueError("raw_content cannot be empty")

        if not self.source_id:
            raise ValueError("source_id cannot be empty")


# ============================================================
# RATE LIMITER
# Token bucket — prevents hammering external APIs
# ============================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    Allows bursts up to capacity, refills at calls_per_minute rate.
    """

    def __init__(self, calls_per_minute: int = 10):
        self.calls_per_minute = calls_per_minute
        self.capacity = max(calls_per_minute, 5)
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available."""
        async with self._lock:
            await self._refill()
            while self.tokens < 1.0:
                wait_time = (1.0 - self.tokens) / (self.calls_per_minute / 60.0)
                await asyncio.sleep(wait_time)
                await self._refill()
            self.tokens -= 1.0

    async def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = elapsed * (self.calls_per_minute / 60.0)
        self.tokens = min(self.capacity, self.tokens + refill_amount)
        self.last_refill = now


# ============================================================
# BASE CONNECTOR
# All auth, rate limiting, retry lives here — never duplicated
# ============================================================

class BaseConnector(ABC):
    """
    Abstract base class for all Veridian connectors.

    Subclasses implement:
        fetch(since) -> AsyncIterator[RawSignal]
        health_check() -> bool

    Everything else (retry, rate limiting, error handling)
    is handled here and never needs to be repeated.
    """

    def __init__(self, config: dict):
        self.config = config
        self.rate_limiter = TokenBucketRateLimiter(
            calls_per_minute=config.get("rate_limit_per_minute", 10)
        )
        self.max_retries = config.get("max_retries", 3)
        self.base_delay = config.get("retry_base_delay", 1.0)
        self.max_delay = config.get("retry_max_delay", 30.0)
        self._fetch_count = 0
        self._error_count = 0

    @abstractmethod
    async def fetch(self, since: datetime) -> AsyncIterator[RawSignal]:
        """
        Fetch signals from the source since the given datetime.
        Yields RawSignal objects. Must be an async generator.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the data source is reachable and responding.
        Returns True if healthy, False otherwise.
        """
        ...

    async def safe_fetch(self, since: datetime) -> AsyncIterator[RawSignal]:
        """
        Wraps fetch() with retry logic and error handling.
        NEVER raises — connector failure never crashes the pipeline.
        """
        connector_name = self.__class__.__name__

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "connector.fetch_start",
                    connector=connector_name,
                    since=since.isoformat(),
                    attempt=attempt,
                )

                signal_count = 0
                async for signal in self.fetch(since):
                    self._fetch_count += 1
                    signal_count += 1
                    yield signal

                logger.info(
                    "connector.fetch_complete",
                    connector=connector_name,
                    signals_yielded=signal_count,
                    attempt=attempt,
                )
                return  # Success — exit retry loop

            except httpx.TimeoutException as e:
                self._error_count += 1
                logger.warning(
                    "connector.timeout",
                    connector=connector_name,
                    attempt=attempt,
                    error=str(e),
                )

            except httpx.HTTPStatusError as e:
                self._error_count += 1

                # Don't retry on auth errors or rate limit exhaustion
                if e.response.status_code in (401, 403):
                    logger.error(
                        "connector.auth_error",
                        connector=connector_name,
                        status_code=e.response.status_code,
                    )
                    return

                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("Retry-After", 60))
                    logger.warning(
                        "connector.rate_limited",
                        connector=connector_name,
                        retry_after_seconds=retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                logger.warning(
                    "connector.http_error",
                    connector=connector_name,
                    status_code=e.response.status_code,
                    attempt=attempt,
                )

            except Exception as e:
                self._error_count += 1
                logger.error(
                    "connector.unexpected_error",
                    connector=connector_name,
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            # Exponential backoff before retry
            if attempt < self.max_retries:
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                logger.info(
                    "connector.retry_wait",
                    connector=connector_name,
                    delay_seconds=delay,
                    next_attempt=attempt + 1,
                )
                await asyncio.sleep(delay)

        logger.error(
            "connector.all_retries_exhausted",
            connector=connector_name,
            total_errors=self._error_count,
        )

    @property
    def stats(self) -> dict:
        """Return fetch statistics for monitoring."""
        return {
            "connector": self.__class__.__name__,
            "total_fetched": self._fetch_count,
            "total_errors": self._error_count,
        }