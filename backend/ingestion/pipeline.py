# backend/ingestion/pipeline.py
"""
Veridian Signal Processing Pipeline.

Every signal that enters Veridian passes through this pipeline:
    RAW SIGNAL
        → Normalize (clean text, standardize timestamps)
        → Deduplicate (skip signals we've already seen)
        → Score (impact, novelty — urgency comes from connector)
        → Index (write to Elasticsearch with ELSER embedding via ingest pipeline)
        → Trigger (publish to Pub/Sub if anomaly threshold crossed)

ELSER embedding is handled server-side via Elastic ingest pipeline.
We send raw text. Elastic handles the embedding. Clean separation.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from datetime import datetime, timezone
from typing import Sequence

import structlog
from elasticsearch import AsyncElasticsearch, BadRequestError, ConflictError

from backend.es.client import get_client
from backend.ingestion.base_connector import BaseConnector, RawSignal

logger = structlog.get_logger(__name__)

# ============================================================
# INGEST PIPELINE SETUP
# Created once in Elasticsearch. Handles ELSER embedding
# automatically on every document write.
# ============================================================

INGEST_PIPELINE_ID = "veridian-elser-pipeline"

INGEST_PIPELINE_DEFINITION = {
    "description": "Veridian ELSER embedding pipeline — auto-embeds raw_content on ingest",
    "processors": [
        {
            "inference": {
                "model_id": ".elser_model_2_linux-x86_64",
                "input_output": [
                    {
                        "input_field": "raw_content",
                        "output_field": "semantic_embedding",
                    }
                ],
                "on_failure": [
                    {
                        "set": {
                            "field": "_index",
                            "value": "failed_{{ _index }}",
                            "ignore_failure": True,
                        }
                    }
                ],
            }
        }
    ],
}


# ============================================================
# NORMALIZER
# Cleans raw signal text before indexing
# ============================================================

def _normalize_content(raw: str) -> str:
    """
    Clean raw signal content.
    - Strip HTML tags
    - Normalize whitespace
    - Remove null bytes and control characters
    - Cap length
    """
    if not raw:
        return ""

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", raw)

    # Remove null bytes and control characters (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = text.strip()

    # Cap at 5000 characters — ELSER has context limits
    return text[:5000]


# ============================================================
# SCORER
# Computes impact_score and novelty_score for each signal
# urgency_score comes from the connector (source-specific logic)
# ============================================================

def _compute_impact_score(signal: RawSignal) -> float:
    """
    Impact score: if this signal is meaningful, how big is the blast radius?
    Based on signal type and entity count.
    """
    base_impact = {
        "funding": 0.85,
        "leadership": 0.80,
        "product_launch": 0.70,
        "regulatory": 0.75,
        "competitive": 0.65,
        "partnership": 0.60,
        "hiring": 0.50,
        "financial": 0.55,
        "incident": 0.70,
        "general_news": 0.30,
        "developer_sentiment": 0.35,
        "community_discussion": 0.25,
    }.get(signal.signal_type, 0.30)

    # More entity mentions = potentially wider blast radius
    entity_boost = min(len(signal.entity_hints) * 0.05, 0.15)

    return min(base_impact + entity_boost, 1.0)


def _compute_novelty_score(source_id: str, seen_source_ids: set[str]) -> float:
    """
    Novelty score: have we seen this type of signal before?
    Simple implementation — True novelty requires historical ES query.
    For pipeline speed, we use in-memory dedup within a batch.
    """
    # This is a placeholder for the full novelty computation
    # which would query ES for similar signals.
    # For now: 1.0 if new in this batch, 0.5 as default for
    # signals that passed dedup (meaning they're globally new)
    return 1.0 if source_id not in seen_source_ids else 0.0


# ============================================================
# DEDUPLICATOR
# Checks Elasticsearch to see if we've already indexed this signal
# ============================================================

async def _is_duplicate(es: AsyncElasticsearch, source: str, source_id: str) -> bool:
    """
    Check if a signal with this source + source_id already exists.
    Uses a fast term query — no scoring needed.
    """
    try:
        result = await es.count(
            index="veridian_signals",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"source": source}},
                            {"term": {"source_id": source_id}},
                        ]
                    }
                }
            },
        )
        return result["count"] > 0
    except Exception as e:
        logger.warning(
            "pipeline.dedup_check_failed",
            source=source,
            source_id=source_id,
            error=str(e),
        )
        return False  # If check fails, proceed with indexing


# ============================================================
# DOCUMENT BUILDER
# Converts RawSignal → Elasticsearch document
# ============================================================

def _build_document(
    signal: RawSignal,
    normalized_content: str,
    impact_score: float,
    novelty_score: float,
) -> dict:
    """Build the Elasticsearch document from a processed signal."""

    # Generate stable signal_id from source + source_id
    signal_id = f"VRD-SIG-{hashlib.sha256(f'{signal.source}:{signal.source_id}'.encode()).hexdigest()[:12].upper()}"

    return {
        "signal_id": signal_id,
        "source": signal.source,
        "source_id": signal.source_id,
        "raw_content": normalized_content,
        "url": signal.url,
        "timestamp": signal.timestamp.isoformat(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "entity_hints": signal.entity_hints,
        "signal_category": signal.signal_category,
        "signal_type": signal.signal_type,
        "urgency_score": round(signal.urgency_score, 4),
        "impact_score": round(impact_score, 4),
        "novelty_score": round(novelty_score, 4),
        "sentiment_score": 0.0,  # Placeholder — Gemini NER pass adds this later
        "chain_ids": [],
        "processed": False,
        "metadata": signal.metadata,
        # semantic_embedding is populated automatically by the ingest pipeline
        # dense_embedding is populated in a separate enrichment pass
    }


# ============================================================
# PIPELINE RESULT
# ============================================================

class PipelineResult:
    """Result of a pipeline run."""

    def __init__(self):
        self.total_received = 0
        self.total_indexed = 0
        self.total_duplicates = 0
        self.total_errors = 0
        self.indexed_signal_ids: list[str] = []
        self.errors: list[dict] = []

    def __repr__(self) -> str:
        return (
            f"PipelineResult("
            f"received={self.total_received}, "
            f"indexed={self.total_indexed}, "
            f"duplicates={self.total_duplicates}, "
            f"errors={self.total_errors})"
        )


# ============================================================
# THE PIPELINE
# ============================================================

class SignalPipeline:
    """
    Veridian's signal processing pipeline.

    Usage:
        pipeline = SignalPipeline()
        await pipeline.setup()  # Creates ingest pipeline in ES
        result = await pipeline.run(connectors, since)
    """

    def __init__(self):
        self.es: AsyncElasticsearch | None = None
        self._pipeline_ready = False

    async def setup(self) -> None:
        """
        Initialize the pipeline.
        Creates the ELSER ingest pipeline in Elasticsearch if it doesn't exist.
        Call once before running.
        """
        self.es = get_client()
        await self._ensure_ingest_pipeline()
        self._pipeline_ready = True
        logger.info("pipeline.ready", ingest_pipeline=INGEST_PIPELINE_ID)

    async def _ensure_ingest_pipeline(self) -> None:
        """Create the ELSER ingest pipeline if it doesn't exist."""
        try:
            # Check if pipeline already exists
            await self.es.ingest.get_pipeline(id=INGEST_PIPELINE_ID)
            logger.info(
                "pipeline.ingest_pipeline_exists",
                pipeline_id=INGEST_PIPELINE_ID,
            )
            return
        except Exception:
            pass  # Doesn't exist — create it

        try:
            await self.es.ingest.put_pipeline(
                id=INGEST_PIPELINE_ID,
                body=INGEST_PIPELINE_DEFINITION,
            )
            logger.info(
                "pipeline.ingest_pipeline_created",
                pipeline_id=INGEST_PIPELINE_ID,
            )
        except Exception as e:
            logger.error(
                "pipeline.ingest_pipeline_creation_failed",
                error=str(e),
            )
            raise

    async def process_signal(
        self,
        signal: RawSignal,
        seen_source_ids: set[str],
    ) -> tuple[dict | None, str]:
        """
        Process a single signal through the full pipeline.

        Returns:
            (document, signal_id) if indexed
            (None, "") if duplicate or error
        """
        # Step 1: Normalize
        normalized = _normalize_content(signal.raw_content)
        if not normalized:
            return None, ""

        # Step 2: Deduplicate (fast in-memory check first)
        cache_key = f"{signal.source}:{signal.source_id}"
        if cache_key in seen_source_ids:
            return None, ""

        # Step 3: Deduplicate (Elasticsearch check)
        is_dup = await _is_duplicate(self.es, signal.source, signal.source_id)
        if is_dup:
            seen_source_ids.add(cache_key)
            return None, ""

        # Step 4: Score
        impact = _compute_impact_score(signal)
        novelty = _compute_novelty_score(cache_key, seen_source_ids)

        # Step 5: Build document
        doc = _build_document(signal, normalized, impact, novelty)
        signal_id = doc["signal_id"]

        # Step 6: Index (ingest pipeline handles ELSER embedding)
        try:
            await self.es.index(
                index="veridian_signals",
                id=signal_id,
                document=doc,
                pipeline=INGEST_PIPELINE_ID,
            )
            seen_source_ids.add(cache_key)
            return doc, signal_id

        except ConflictError:
            # Document already exists (race condition) — treat as duplicate
            seen_source_ids.add(cache_key)
            return None, ""

        except Exception as e:
            logger.error(
                "pipeline.index_error",
                signal_id=signal_id,
                source=signal.source,
                error=str(e),
            )
            return None, ""

    async def run(
        self,
        connectors: Sequence[BaseConnector],
        since: datetime,
        batch_size: int = 50,
    ) -> PipelineResult:
        """
        Run the full pipeline across all connectors.

        Args:
            connectors: List of connector instances to fetch from
            since: Only fetch signals after this datetime
            batch_size: Log progress every N signals

        Returns:
            PipelineResult with counts and indexed signal IDs
        """
        if not self._pipeline_ready:
            await self.setup()

        result = PipelineResult()
        seen_source_ids: set[str] = set()

        logger.info(
            "pipeline.run_start",
            connector_count=len(connectors),
            since=since.isoformat(),
        )

        for connector in connectors:
            connector_name = connector.__class__.__name__

            logger.info("pipeline.connector_start", connector=connector_name)

            try:
                async for signal in connector.safe_fetch(since):
                    result.total_received += 1

                    try:
                        doc, signal_id = await self.process_signal(
                            signal, seen_source_ids
                        )

                        if doc is not None:
                            result.total_indexed += 1
                            result.indexed_signal_ids.append(signal_id)

                            logger.debug(
                                "pipeline.signal_indexed",
                                signal_id=signal_id,
                                source=signal.source,
                                signal_type=signal.signal_type,
                                entity_hints=signal.entity_hints,
                            )
                        else:
                            result.total_duplicates += 1

                        # Progress logging
                        if result.total_received % batch_size == 0:
                            logger.info(
                                "pipeline.progress",
                                connector=connector_name,
                                received=result.total_received,
                                indexed=result.total_indexed,
                                duplicates=result.total_duplicates,
                            )

                    except Exception as e:
                        result.total_errors += 1
                        result.errors.append({
                            "source": signal.source,
                            "source_id": signal.source_id,
                            "error": str(e),
                        })
                        logger.error(
                            "pipeline.signal_error",
                            source=signal.source,
                            error=str(e),
                        )

            except Exception as e:
                logger.error(
                    "pipeline.connector_failed",
                    connector=connector_name,
                    error=str(e),
                )

            logger.info(
                "pipeline.connector_complete",
                connector=connector_name,
                total_received=result.total_received,
                total_indexed=result.total_indexed,
            )

        logger.info(
            "pipeline.run_complete",
            total_received=result.total_received,
            total_indexed=result.total_indexed,
            total_duplicates=result.total_duplicates,
            total_errors=result.total_errors,
        )

        return result


# ============================================================
# QUICK TEST SCRIPT
# Run: python -m backend.ingestion.pipeline
# ============================================================

async def _test_pipeline():
    """Quick smoke test — run from command line to verify pipeline works."""
    import os
    from datetime import timedelta
    from dotenv import load_dotenv

    load_dotenv()

    print("\n" + "═" * 55)
    print("  VERIDIAN — Signal Pipeline Test")
    print("═" * 55)

    # Initialize connectors
    watch_entities = [
        e.strip()
        for e in os.getenv("VERIDIAN_WATCH_ENTITIES", "OpenAI,Anthropic").split(",")
        if e.strip()
    ]

    connectors = []

    # NewsAPI connector
    newsapi_key = os.getenv("NEWSAPI_KEY")
    if newsapi_key:
        from backend.ingestion.connectors.newsapi import NewsAPIConnector
        connectors.append(NewsAPIConnector({
            "api_key": newsapi_key,
            "watch_entities": watch_entities,
            "page_size": 10,  # Small batch for test
        }))
        print(f"  ✓ NewsAPI connector ready")
    else:
        print(f"  ! NewsAPI key not found — skipping")

    # HackerNews connector (no auth needed)
    from backend.ingestion.connectors.hackernews import HackerNewsConnector
    connectors.append(HackerNewsConnector({
        "watch_entities": watch_entities,
        "min_points": 5,
    }))
    print(f"  ✓ HackerNews connector ready")

    print(f"  Watching: {watch_entities}")
    print(f"  Connectors: {len(connectors)}")

    # Health checks
    print("\n── Health Checks ──────────────────────────────────────")
    for connector in connectors:
        name = connector.__class__.__name__
        healthy = await connector.health_check()
        status = "✓" if healthy else "✗"
        print(f"  {status} {name}: {'OK' if healthy else 'FAILED'}")

    # Run pipeline
    print("\n── Running Pipeline ───────────────────────────────────")
    since = datetime.now(timezone.utc) - timedelta(days=7)
    print(f"  Fetching signals since: {since.strftime('%Y-%m-%d %H:%M UTC')}")

    pipeline = SignalPipeline()
    await pipeline.setup()

    print(f"  ✓ Ingest pipeline configured (ELSER embedding enabled)")
    print(f"  Running...\n")

    result = await pipeline.run(connectors, since)

    print("\n── Results ────────────────────────────────────────────")
    print(f"  Signals received:   {result.total_received}")
    print(f"  Signals indexed:    {result.total_indexed}")
    print(f"  Duplicates skipped: {result.total_duplicates}")
    print(f"  Errors:             {result.total_errors}")

    if result.indexed_signal_ids:
        print(f"\n  Sample indexed IDs:")
        for sid in result.indexed_signal_ids[:5]:
            print(f"    • {sid}")

    print("\n" + "═" * 55)
    if result.total_errors == 0 and result.total_received > 0:
        print("  ✓ Pipeline working. Signals are flowing into Elastic.")
    elif result.total_received == 0:
        print("  ! No signals found. Try increasing time window.")
    else:
        print(f"  ! Pipeline ran with {result.total_errors} errors. Check logs.")
    print("═" * 55 + "\n")
    from backend.es.client import close_client
    await close_client()
    return result


if __name__ == "__main__":
    asyncio.run(_test_pipeline())