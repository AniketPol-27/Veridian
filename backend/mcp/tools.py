# backend/mcp/tools.py
"""
Veridian's 6 Elastic MCP Tools.

These are the load-bearing integration points between
Gemini's reasoning and Elasticsearch's intelligence.

Gemini calls these. Elastic does the work.
Every tool is typed, logged, and returns structured output
ready for direct inclusion in Gemini prompts.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog
from elasticsearch import AsyncElasticsearch, NotFoundError

from backend.es.client import get_client
from backend.mcp.schemas import (
    AccuracyByType,
    AnomalyDirection,
    AnomalyResult,
    CalibrationReport,
    CalibrationVerdict,
    ComputeAnomalyInput,
    DimensionState,
    EntityState,
    FindHistoricalPatternInput,
    FindHistoricalPatternOutput,
    FindRelatedSignalsInput,
    FindRelatedSignalsOutput,
    GetEntityStateInput,
    PatternMatch,
    SignalResult,
    StorePredictionInput,
    StorePredictionOutput,
)

logger = structlog.get_logger(__name__)

ELSER_MODEL_ID = os.getenv("ELSER_MODEL_ID", ".elser_model_2_linux-x86_64")


# ============================================================
# TOOL 1: find_related_signals
# Semantic memory retrieval — this IS Oracle's memory
# Gemini has no persistent memory. This gives it one.
# ============================================================

async def find_related_signals(
    params: FindRelatedSignalsInput,
) -> FindRelatedSignalsOutput:
    """
    Retrieve semantically related signals from Elasticsearch.

    Uses ELSER sparse vector search for semantic matching —
    finds signals that are conceptually related even when
    they use completely different terminology.

    This is the primary tool Gemini uses to ground its
    reasoning in actual evidence rather than training data.
    """
    es = get_client()

    logger.info(
        "mcp.find_related_signals",
        query=params.query[:80],
        time_range_days=params.time_range_days,
        min_score=params.min_score,
        limit=params.limit,
    )

    # Build the query
    must_clauses: list[dict] = [
        {
            "range": {
                "timestamp": {
                    "gte": f"now-{params.time_range_days}d",
                    "lte": "now",
                }
            }
        }
    ]

    # ELSER semantic search — the core of Veridian's memory
    should_clauses: list[dict] = [
    {
        "sparse_vector": {
            "field": "semantic_embedding",
            "inference_id": ELSER_MODEL_ID,
            "query": params.query,
            "boost": 1.0,
        }
    }
]

    # Also include keyword search for hybrid retrieval
    should_clauses.append({
        "multi_match": {
            "query": params.query,
            "fields": ["raw_content", "entity_hints"],
            "type": "best_fields",
            "boost": 0.3,
        }
    })

    # Optional filters
    filter_clauses: list[dict] = []

    if params.signal_types:
        filter_clauses.append({"terms": {"signal_type": params.signal_types}})

    if params.entity_filter:
        filter_clauses.append({
            "terms": {"entity_hints": params.entity_filter}
        })

    if params.exclude_signal_ids:
        filter_clauses.append({
            "bool": {
                "must_not": [
                    {"terms": {"signal_id": params.exclude_signal_ids}}
                ]
            }
        })

    query_body: dict[str, Any] = {
        "query": {
            "bool": {
                "must": must_clauses,
                "should": should_clauses,
                "filter": filter_clauses,
                "minimum_should_match": 1,
            }
        },
        "min_score": params.min_score,
        "size": params.limit,
        "_source": [
            "signal_id",
            "source",
            "signal_type",
            "signal_category",
            "entity_hints",
            "raw_content",
            "timestamp",
            "urgency_score",
            "impact_score",
            "url",
        ],
        "sort": [
            "_score",
            {"timestamp": {"order": "desc"}},
        ],
    }

    try:
        response = await es.search(
            index="veridian_signals",
            body=query_body,
        )
    except Exception as e:
        logger.error("mcp.find_related_signals.error", error=str(e))
        raise

    hits = response["hits"]["hits"]
    total = response["hits"]["total"]["value"]

    signals = []
    for hit in hits:
        src = hit["_source"]
        signals.append(
            SignalResult(
                signal_id=src.get("signal_id", hit["_id"]),
                source=src.get("source", "unknown"),
                signal_type=src.get("signal_type", "unknown"),
                signal_category=src.get("signal_category", "unknown"),
                entity_hints=src.get("entity_hints", []),
                raw_content=src.get("raw_content", ""),
                timestamp=src.get("timestamp", ""),
                urgency_score=src.get("urgency_score", 0.0),
                impact_score=src.get("impact_score", 0.0),
                relevance_score=round(hit["_score"], 4),
                url=src.get("url"),
            )
        )

    # Determine search strategy used
    search_strategy = "hybrid" if len(should_clauses) > 1 else "semantic"

    result = FindRelatedSignalsOutput(
        signals=signals,
        total_found=total,
        query=params.query,
        time_range_days=params.time_range_days,
        search_strategy=search_strategy,
    )

    logger.info(
        "mcp.find_related_signals.complete",
        total_found=total,
        returned=len(signals),
        strategy=search_strategy,
    )

    return result


# ============================================================
# TOOL 2: compute_entity_anomaly_score
# Statistical anomaly detection via ES|QL
# Gemini cannot compute aggregate statistics over stored records.
# This gives it that capability.
# ============================================================

async def compute_entity_anomaly_score(
    params: ComputeAnomalyInput,
) -> AnomalyResult:
    """
    Detect statistical anomalies in entity signal patterns.

    Computes Z-score by comparing the current window's metric
    distribution against the historical baseline.

    Z-score >= 2.0 = anomalous (95th percentile)
    Z-score >= 3.0 = severe anomaly (99.7th percentile)
    Z-score >= 4.0 = extreme anomaly

    Uses standard ES aggregations for reliable array field filtering.
    """
    es = get_client()

    logger.info(
        "mcp.compute_entity_anomaly",
        entity=params.entity_name,
        metric=params.metric,
        lookback_days=params.lookback_days,
        current_window_days=params.current_window_days,
    )

    def _empty_result(reason: str) -> AnomalyResult:
        return AnomalyResult(
            entity_name=params.entity_name,
            metric=params.metric,
            is_anomalous=False,
            z_score=0.0,
            direction=AnomalyDirection.STABLE,
            baseline_mean=0.0,
            current_mean=0.0,
            baseline_count=0,
            current_count=0,
            baseline_std=0.0,
            anomaly_magnitude="mild",
            interpretation=reason,
            confidence="low",
        )

    # ── Baseline query (older period, excluding current window) ──
    try:
        baseline_response = await es.search(
            index="veridian_signals",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"entity_hints": params.entity_name}},
                            {
                                "range": {
                                    "timestamp": {
                                        "gte": f"now-{params.lookback_days}d",
                                        "lte": f"now-{params.current_window_days}d",
                                    }
                                }
                            },
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    "mean_val": {"avg": {"field": params.metric}},
                    "min_val": {"min": {"field": params.metric}},
                    "max_val": {"max": {"field": params.metric}},
                    "doc_count": {"value_count": {"field": params.metric}},
                },
            },
        )
    except Exception as e:
        logger.error(
            "mcp.compute_entity_anomaly.baseline_error",
            error=str(e),
            entity=params.entity_name,
        )
        return _empty_result(f"Could not compute baseline: {str(e)[:100]}")

    # ── Current window query ──
    try:
        current_response = await es.search(
            index="veridian_signals",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"entity_hints": params.entity_name}},
                            {
                                "range": {
                                    "timestamp": {
                                        "gte": f"now-{params.current_window_days}d",
                                    }
                                }
                            },
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    "mean_val": {"avg": {"field": params.metric}},
                    "doc_count": {"value_count": {"field": params.metric}},
                },
            },
        )
    except Exception as e:
        logger.error(
            "mcp.compute_entity_anomaly.current_error",
            error=str(e),
            entity=params.entity_name,
        )
        return _empty_result(f"Could not compute current window: {str(e)[:100]}")

    # ── Parse results ──
    baseline_aggs = baseline_response["aggregations"]
    current_aggs = current_response["aggregations"]

    baseline_mean = float(baseline_aggs["mean_val"].get("value") or 0)
    baseline_count = int(baseline_aggs["doc_count"].get("value") or 0)
    baseline_min = float(baseline_aggs["min_val"].get("value") or 0)
    baseline_max = float(baseline_aggs["max_val"].get("value") or 0)

    current_mean = float(current_aggs["mean_val"].get("value") or 0)
    current_count = int(current_aggs["doc_count"].get("value") or 0)

    # Need both periods to compute anomaly
    if baseline_count == 0:
        return _empty_result(
            f"No baseline signals found for '{params.entity_name}' "
            f"in the {params.lookback_days}d lookback window."
        )

    if current_count == 0:
        return _empty_result(
            f"No current signals found for '{params.entity_name}' "
            f"in the last {params.current_window_days} days."
        )

    # ── Compute std from range (range/4 approximation) ──
    if baseline_max > baseline_min and baseline_count >= 2:
        baseline_std = max((baseline_max - baseline_min) / 4.0, 0.001)
    else:
        baseline_std = 0.1  # Safe default for single-value baseline

    # ── Z-score ──
    z_score = (current_mean - baseline_mean) / baseline_std

    # ── Classify ──
    abs_z = abs(z_score)
    is_anomalous = abs_z >= 2.0

    if abs_z >= 4.0:
        magnitude = "extreme"
    elif abs_z >= 3.0:
        magnitude = "severe"
    elif abs_z >= 2.0:
        magnitude = "moderate"
    else:
        magnitude = "mild"

    direction = (
        AnomalyDirection.INCREASING if z_score > 0.05
        else AnomalyDirection.DECREASING if z_score < -0.05
        else AnomalyDirection.STABLE
    )

    # ── Confidence based on sample size ──
    min_count = min(baseline_count, current_count)
    confidence = (
        "high" if min_count >= 20
        else "medium" if min_count >= 5
        else "low"
    )

    # ── Human-readable interpretation ──
    pct_change = (
        ((current_mean - baseline_mean) / baseline_mean * 100)
        if baseline_mean > 0 else 0
    )
    direction_word = "increased" if z_score > 0 else "decreased"
    interpretation = (
        f"Signal {params.metric} for '{params.entity_name}' has {direction_word} "
        f"by {abs(pct_change):.1f}% ({magnitude} anomaly, Z={z_score:.2f}). "
        f"Baseline: {baseline_mean:.3f} ({baseline_count} signals) → "
        f"Current: {current_mean:.3f} ({current_count} signals)."
    )

    result = AnomalyResult(
        entity_name=params.entity_name,
        metric=params.metric,
        is_anomalous=is_anomalous,
        z_score=round(z_score, 4),
        direction=direction,
        baseline_mean=round(baseline_mean, 4),
        current_mean=round(current_mean, 4),
        baseline_count=baseline_count,
        current_count=current_count,
        baseline_std=round(baseline_std, 4),
        anomaly_magnitude=magnitude,
        interpretation=interpretation,
        confidence=confidence,
    )

    logger.info(
        "mcp.compute_entity_anomaly.complete",
        entity=params.entity_name,
        is_anomalous=is_anomalous,
        z_score=round(z_score, 4),
        magnitude=magnitude,
        baseline_count=baseline_count,
        current_count=current_count,
    )

    return result


# ============================================================
# TOOL 3: find_historical_pattern_match
# KNN dense vector search against chain fingerprints
# Veridian's institutional memory — finds similar past situations
# ============================================================

async def find_historical_pattern_match(
    params: FindHistoricalPatternInput,
) -> FindHistoricalPatternOutput:
    """
    Find historically similar causal chains using KNN vector search.

    Averages the dense embeddings of current signals into a
    "situation fingerprint" and finds the most similar historical
    chains — including their actual outcomes.

    This is how Veridian learns from history:
    "I've seen something like this before. Here's what happened."
    """
    es = get_client()

    logger.info(
        "mcp.find_historical_pattern",
        signal_count=len(params.current_signal_ids),
        top_k=params.top_k,
    )

    # Fetch dense embeddings for current signals
    try:
        docs_response = await es.mget(
            index="veridian_signals",
            body={"ids": params.current_signal_ids},
            source_includes=["dense_embedding", "signal_id"],
        )
    except Exception as e:
        logger.error("mcp.find_historical_pattern.mget_error", error=str(e))
        return FindHistoricalPatternOutput(
            matches=[],
            total_searched=0,
            fingerprint_signal_count=0,
            has_strong_precedent=False,
        )

    # Extract embeddings from found documents
    embeddings = []
    for doc in docs_response.get("docs", []):
        if doc.get("found") and doc.get("_source", {}).get("dense_embedding"):
            embeddings.append(doc["_source"]["dense_embedding"])

    if not embeddings:
        logger.warning(
            "mcp.find_historical_pattern.no_embeddings",
            msg="No dense embeddings found for provided signal IDs. "
                "dense_embedding field requires separate enrichment pass.",
        )
        # Fall back to semantic search on chain summaries
        return await _fallback_text_pattern_match(params, es)

    # Average embeddings into chain fingerprint
    dims = len(embeddings[0])
    chain_fingerprint = [
        sum(emb[i] for emb in embeddings) / len(embeddings)
        for i in range(dims)
    ]

    # KNN search against historical chain embeddings
    try:
        response = await es.search(
            index="veridian_causal_chains",
            body={
                "knn": {
                    "field": "chain_embedding",
                    "query_vector": chain_fingerprint,
                    "k": params.top_k,
                    "num_candidates": params.top_k * 10,
                    "filter": {
                        "term": {"status": "validated"}
                    },
                },
                "_source": [
                    "chain_id",
                    "chain_summary",
                    "outcome_summary",
                    "outcome_type",
                    "actual_accuracy",
                    "time_to_outcome_days",
                    "confidence_at_creation",
                    "created_at",
                ],
            },
        )
    except Exception as e:
        logger.error("mcp.find_historical_pattern.knn_error", error=str(e))
        return FindHistoricalPatternOutput(
            matches=[],
            total_searched=0,
            fingerprint_signal_count=len(embeddings),
            has_strong_precedent=False,
        )

    hits = response["hits"]["hits"]

    # Count total validated chains in index
    try:
        count_result = await es.count(
            index="veridian_causal_chains",
            body={"query": {"term": {"status": "validated"}}},
        )
        total_searched = count_result["count"]
    except Exception:
        total_searched = len(hits)

    matches = []
    for hit in hits:
        score = hit.get("_score") or 0.0
        if score < params.min_similarity:
            continue

        src = hit["_source"]
        matches.append(
            PatternMatch(
                chain_id=src.get("chain_id", hit["_id"]),
                similarity_score=round(score, 4),
                chain_summary=src.get("chain_summary", "No summary available"),
                outcome_summary=src.get("outcome_summary", "No outcome recorded"),
                outcome_type=src.get("outcome_type", "unknown"),
                historical_accuracy=float(src.get("actual_accuracy") or 0.0),
                time_to_outcome_days=src.get("time_to_outcome_days"),
                confidence_at_creation=float(src.get("confidence_at_creation") or 0.0),
                created_at=src.get("created_at", ""),
            )
        )

    has_strong_precedent = any(m.similarity_score > 0.8 for m in matches)

    result = FindHistoricalPatternOutput(
        matches=matches,
        total_searched=total_searched,
        fingerprint_signal_count=len(embeddings),
        has_strong_precedent=has_strong_precedent,
    )

    logger.info(
        "mcp.find_historical_pattern.complete",
        matches_found=len(matches),
        has_strong_precedent=has_strong_precedent,
        total_searched=total_searched,
    )

    return result


async def _fallback_text_pattern_match(
    params: FindHistoricalPatternInput,
    es: AsyncElasticsearch,
) -> FindHistoricalPatternOutput:
    """
    Fallback: when dense embeddings aren't available,
    use semantic search on chain summaries instead.
    Less precise but still useful.
    """
    # Fetch signal content to build a text query
    try:
        signals_response = await es.search(
            index="veridian_signals",
            body={
                "query": {"terms": {"signal_id": params.current_signal_ids}},
                "size": 5,
                "_source": ["raw_content", "entity_hints", "signal_type"],
            },
        )
        signal_texts = [
            hit["_source"].get("raw_content", "")[:200]
            for hit in signals_response["hits"]["hits"]
        ]
        combined_query = " ".join(signal_texts)[:500]
    except Exception:
        return FindHistoricalPatternOutput(
            matches=[],
            total_searched=0,
            fingerprint_signal_count=0,
            has_strong_precedent=False,
        )

    if not combined_query:
        return FindHistoricalPatternOutput(
            matches=[],
            total_searched=0,
            fingerprint_signal_count=0,
            has_strong_precedent=False,
        )

    try:
        response = await es.search(
            index="veridian_causal_chains",
            body={
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "chain_summary": {
                                        "query": combined_query,
                                        "boost": 1.0,
                                    }
                                }
                            },
                            {
                                "match": {
                                    "proposed_cause": {
                                        "query": combined_query,
                                        "boost": 0.5,
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                        "filter": [{"term": {"status": "validated"}}],
                    }
                },
                "size": params.top_k,
                "_source": [
                    "chain_id", "chain_summary", "outcome_summary",
                    "outcome_type", "actual_accuracy", "time_to_outcome_days",
                    "confidence_at_creation", "created_at",
                ],
            },
        )
    except Exception as e:
        logger.error("mcp.fallback_pattern_match.error", error=str(e))
        return FindHistoricalPatternOutput(
            matches=[],
            total_searched=0,
            fingerprint_signal_count=0,
            has_strong_precedent=False,
        )

    hits = response["hits"]["hits"]
    matches = [
        PatternMatch(
            chain_id=hit["_source"].get("chain_id", hit["_id"]),
            similarity_score=round(min(hit["_score"] / 10.0, 1.0), 4),
            chain_summary=hit["_source"].get("chain_summary", ""),
            outcome_summary=hit["_source"].get("outcome_summary", ""),
            outcome_type=hit["_source"].get("outcome_type", "unknown"),
            historical_accuracy=float(hit["_source"].get("actual_accuracy") or 0.0),
            time_to_outcome_days=hit["_source"].get("time_to_outcome_days"),
            confidence_at_creation=float(hit["_source"].get("confidence_at_creation") or 0.0),
            created_at=hit["_source"].get("created_at", ""),
        )
        for hit in hits
    ]

    return FindHistoricalPatternOutput(
        matches=matches,
        total_searched=0,
        fingerprint_signal_count=0,
        has_strong_precedent=any(m.similarity_score > 0.8 for m in matches),
    )


# ============================================================
# TOOL 4: get_entity_state
# Living knowledge graph — ES|QL powered entity intelligence
# ============================================================

async def get_entity_state(
    params: GetEntityStateInput,
) -> EntityState:
    """
    Retrieve the current state of a business entity across all dimensions.
    Uses standard ES queries for array field filtering + ES|QL for aggregations.
    """
    es = get_client()

    logger.info(
        "mcp.get_entity_state",
        entity=params.entity_name,
        dimensions=params.dimensions,
        lookback_days=params.lookback_days,
    )

    dimension_states: dict[str, DimensionState] = {}

    for dimension in params.dimensions:
        try:
            # Use standard search with aggregations — handles keyword arrays correctly
            agg_response = await es.search(
                index="veridian_signals",
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"entity_hints": params.entity_name}},
                                {"term": {"signal_category": dimension}},
                                {"range": {"timestamp": {"gte": f"now-{params.lookback_days}d"}}},
                            ]
                        }
                    },
                    "size": 0,
                    "aggs": {
                        "avg_impact": {"avg": {"field": "impact_score"}},
                        "avg_urgency": {"avg": {"field": "urgency_score"}},
                        "avg_sentiment": {"avg": {"field": "sentiment_score"}},
                        "latest": {"max": {"field": "timestamp"}},
                        "recent_count": {"value_count": {"field": "signal_id"}},
                        # Trend: split into recent 14d vs prior
                        "recent_14d": {
                            "filter": {"range": {"timestamp": {"gte": "now-14d"}}},
                            "aggs": {
                                "count": {"value_count": {"field": "signal_id"}}
                            }
                        },
                    },
                },
            )

            total_hits = agg_response["hits"]["total"]["value"]
            aggs = agg_response["aggregations"]

            if total_hits > 0:
                signal_count = total_hits
                avg_impact = float(aggs["avg_impact"].get("value") or 0)
                avg_urgency = float(aggs["avg_urgency"].get("value") or 0)
                avg_sentiment = float(aggs["avg_sentiment"].get("value") or 0)
                latest_ts = aggs["latest"].get("value_as_string")
                recent_14d = aggs["recent_14d"]["count"]["value"]

                # Compute trend
                prior_count = signal_count - recent_14d
                prior_days = max(params.lookback_days - 14, 1)
                recent_rate = recent_14d / 14
                prior_rate = prior_count / prior_days

                if signal_count < 3:
                    trend = "insufficient_data"
                elif prior_rate == 0:
                    trend = "accelerating" if recent_14d > 0 else "insufficient_data"
                else:
                    ratio = recent_rate / prior_rate
                    if ratio > 1.5:
                        trend = "accelerating"
                    elif ratio < 0.5:
                        trend = "declining"
                    else:
                        trend = "stable"

                # Get most recent content
                recent_content = await _get_recent_content(
                    es, params.entity_name, dimension, params.lookback_days
                )

                dimension_states[dimension] = DimensionState(
                    dimension=dimension,
                    signal_count=signal_count,
                    avg_impact=round(avg_impact, 4),
                    avg_urgency=round(avg_urgency, 4),
                    avg_sentiment=round(avg_sentiment, 4),
                    latest_timestamp=latest_ts,
                    recent_content=recent_content,
                    trend=trend,
                )
            else:
                dimension_states[dimension] = DimensionState(
                    dimension=dimension,
                    signal_count=0,
                    avg_impact=0.0,
                    avg_urgency=0.0,
                    avg_sentiment=0.0,
                    latest_timestamp=None,
                    recent_content=None,
                    trend="insufficient_data",
                )

        except Exception as e:
            logger.warning(
                "mcp.get_entity_state.dimension_error",
                entity=params.entity_name,
                dimension=dimension,
                error=str(e),
            )
            dimension_states[dimension] = DimensionState(
                dimension=dimension,
                signal_count=0,
                avg_impact=0.0,
                avg_urgency=0.0,
                avg_sentiment=0.0,
                latest_timestamp=None,
                recent_content=None,
                trend="insufficient_data",
            )

    # Overall entity metrics using standard search aggregations
    total_count = 0
    first_seen = None
    last_seen = None
    risk_score = 0.0
    opportunity_score = 0.0

    try:
        total_response = await es.search(
            index="veridian_signals",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"entity_hints": params.entity_name}},
                            {"range": {"timestamp": {"gte": f"now-{params.lookback_days}d"}}},
                        ]
                    }
                },
                "size": 0,
                "aggs": {
                    "first_seen": {"min": {"field": "timestamp"}},
                    "last_seen": {"max": {"field": "timestamp"}},
                    "avg_impact": {"avg": {"field": "impact_score"}},
                    "avg_urgency": {"avg": {"field": "urgency_score"}},
                },
            },
        )

        total_count = total_response["hits"]["total"]["value"]
        total_aggs = total_response["aggregations"]

        first_seen = total_aggs["first_seen"].get("value_as_string")
        last_seen = total_aggs["last_seen"].get("value_as_string")
        avg_impact = float(total_aggs["avg_impact"].get("value") or 0)
        avg_urgency = float(total_aggs["avg_urgency"].get("value") or 0)

        risk_score = round(min(avg_urgency * 0.6 + avg_impact * 0.4, 1.0), 4)

        news_state = dimension_states.get("news")
        financial_state = dimension_states.get("financial")
        opp_signals = []
        if news_state and news_state.signal_count > 0:
            opp_signals.append(news_state.avg_impact)
        if financial_state and financial_state.signal_count > 0:
            opp_signals.append(financial_state.avg_impact)
        opportunity_score = round(
            sum(opp_signals) / len(opp_signals) if opp_signals else 0.0, 4
        )

    except Exception as e:
        logger.warning(
            "mcp.get_entity_state.total_error",
            entity=params.entity_name,
            error=str(e),
        )

    result = EntityState(
        entity_name=params.entity_name,
        entity_type=None,
        total_signal_count=total_count,
        first_seen=first_seen,
        last_seen=last_seen,
        overall_risk_score=risk_score,
        overall_opportunity_score=opportunity_score,
        dimension_states=dimension_states,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
    )

    logger.info(
        "mcp.get_entity_state.complete",
        entity=params.entity_name,
        total_signals=total_count,
        dimensions_with_data=sum(
            1 for d in dimension_states.values() if d.signal_count > 0
        ),
    )

    return result


async def _get_recent_content(
    es: AsyncElasticsearch,
    entity_name: str,
    dimension: str,
    lookback_days: int,
) -> str | None:
    """Fetch the most recent signal content for an entity/dimension."""
    try:
        response = await es.search(
            index="veridian_signals",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"signal_category": dimension}},
                            {"range": {"timestamp": {"gte": f"now-{lookback_days}d"}}},
                        ],
                        "filter": [
                            {"term": {"entity_hints": entity_name}}
                        ],
                    }
                },
                "size": 1,
                "sort": [{"timestamp": {"order": "desc"}}],
                "_source": ["raw_content"],
            },
        )
        hits = response["hits"]["hits"]
        if hits:
            content = hits[0]["_source"].get("raw_content", "")
            return content[:300] if content else None
    except Exception:
        pass
    return None
# ============================================================
# TOOL 5: store_prediction
# Every prediction stored with full provenance for calibration
# This is what makes Veridian's self-improvement possible
# ============================================================

async def store_prediction(
    params: StorePredictionInput,
) -> StorePredictionOutput:
    """
    Persist a prediction to Elasticsearch with full provenance.

    Every prediction Veridian makes is logged here:
    - The exact text of the prediction
    - The confidence at time of creation
    - All signals that drove it
    - The scenario analysis (what happens if we act vs don't)
    - When to validate it against actual outcomes

    This write-back is what enables the self-calibration loop.
    Veridian measures its own accuracy by querying these records
    and comparing predictions against what actually happened.
    """
    es = get_client()

    prediction_id = f"VRD-PRD-{uuid.uuid4().hex[:8].upper()}"
    now = datetime.now(timezone.utc)
    validate_at = now + timedelta(days=params.validate_in_days)

    logger.info(
        "mcp.store_prediction",
        prediction_id=prediction_id,
        chain_id=params.chain_id,
        prediction_type=params.prediction_type.value,
        confidence=params.confidence,
        validate_in_days=params.validate_in_days,
    )

    document = {
        "prediction_id": prediction_id,
        "chain_id": params.chain_id,
        "created_at": now.isoformat(),
        "validate_at": validate_at.isoformat(),
        "validated_at": None,
        "prediction_text": params.prediction_text,
        "prediction_type": params.prediction_type.value,
        "confidence": params.confidence,
        "signal_ids": params.signal_ids,
        "key_assumptions": params.key_assumptions,
        "stakes_level": params.stakes_level,
        "scenario_analysis": (
            params.scenario_analysis.model_dump()
            if params.scenario_analysis else None
        ),
        "outcome": None,
    }

    try:
        await es.index(
            index="veridian_predictions",
            id=prediction_id,
            document=document,
        )

        logger.info(
            "mcp.store_prediction.complete",
            prediction_id=prediction_id,
            validate_at=validate_at.isoformat(),
        )

        return StorePredictionOutput(
            prediction_id=prediction_id,
            chain_id=params.chain_id,
            created_at=now.isoformat(),
            validate_at=validate_at.isoformat(),
            confidence=params.confidence,
            prediction_type=params.prediction_type.value,
            success=True,
            message=f"Prediction stored. Will validate on {validate_at.strftime('%Y-%m-%d')}.",
        )

    except Exception as e:
        logger.error(
            "mcp.store_prediction.error",
            prediction_id=prediction_id,
            error=str(e),
        )
        return StorePredictionOutput(
            prediction_id=prediction_id,
            chain_id=params.chain_id,
            created_at=now.isoformat(),
            validate_at=validate_at.isoformat(),
            confidence=params.confidence,
            prediction_type=params.prediction_type.value,
            success=False,
            message=f"Failed to store prediction: {str(e)[:100]}",
        )


# ============================================================
# TOOL 6: run_calibration_cycle
# The self-improvement loop — Veridian measures its own accuracy
# This is the feature no other business intelligence system has
# ============================================================

async def run_calibration_cycle() -> CalibrationReport:
    """
    Evaluate all mature predictions against actual outcomes.

    For each prediction past its validate_at date:
    1. Find outcome evidence in Elasticsearch (what actually happened)
    2. Score accuracy against the prediction
    3. Identify systematic biases
    4. Write calibration report back to Elasticsearch

    This is how Veridian gets measurably smarter over time.
    Every wrong prediction is an opportunity to improve.
    """
    es = get_client()
    now = datetime.now(timezone.utc)
    calibration_id = f"VRD-CAL-{uuid.uuid4().hex[:8].upper()}"

    logger.info("mcp.calibration_cycle.start", calibration_id=calibration_id)

    # Step 1: Find all predictions past their validation date
    try:
        mature_response = await es.search(
            index="veridian_predictions",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"range": {"validate_at": {"lte": now.isoformat()}}},
                        ],
                        "must_not": [
                            {"exists": {"field": "validated_at"}},
                        ],
            }
        },
                "size": 100,
                "_source": [
                    "prediction_id",
                    "prediction_type",
                    "confidence",
                    "prediction_text",
                    "chain_id",
                    "signal_ids",
                    "validate_at",
        ],
    },
)
    except Exception as e:
        logger.error("mcp.calibration_cycle.fetch_error", error=str(e))
        return CalibrationReport(
            calibration_id=calibration_id,
            ran_at=now.isoformat(),
            predictions_evaluated=0,
            predictions_correct=0,
            predictions_incorrect=0,
            predictions_insufficient_evidence=0,
            overall_accuracy=0.0,
            accuracy_by_type=[],
            bias_findings=[],
            weight_updates=[],
            report_text="Calibration failed: could not fetch mature predictions.",
            success=False,
        )

    mature_predictions = mature_response["hits"]["hits"]
    logger.info(
        "mcp.calibration_cycle.mature_found",
        count=len(mature_predictions),
    )

    # Step 2: Evaluate each prediction
    results_by_type: dict[str, list[dict]] = {}
    correct = 0
    incorrect = 0
    insufficient = 0
    weight_updates = []

    for pred_hit in mature_predictions:
        pred = pred_hit["_source"]
        pred_id = pred.get("prediction_id", pred_hit["_id"])
        pred_type = pred.get("prediction_type", "unknown")
        pred_confidence = float(pred.get("confidence") or 0.5)
        pred_text = pred.get("prediction_text", "")

        # Find outcome evidence: search for signals after prediction was made
        # that are semantically related to the prediction text
        validate_at_str = pred.get("validate_at", now.isoformat())
        try:
            validate_dt = datetime.fromisoformat(validate_at_str.replace("Z", "+00:00"))
        except Exception:
            validate_dt = now

        # Search for outcome signals in the 30 days around validation date
        outcome_response = await es.search(
            index="veridian_signals",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "timestamp": {
                                        "gte": (validate_dt - timedelta(days=15)).isoformat(),
                                        "lte": (validate_dt + timedelta(days=15)).isoformat(),
                                    }
                                }
                            },
                        ],
                        "should": [
                            {
                                "text_expansion": {
                                    "semantic_embedding": {
                                        "model_id": ELSER_MODEL_ID,
                                        "model_text": pred_text[:500],
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "min_score": 1.0,
                "size": 5,
                "_source": ["signal_id", "raw_content", "signal_type", "timestamp"],
            },
        )

        outcome_signals = outcome_response["hits"]["hits"]
        outcome_signal_ids = [
            h["_source"].get("signal_id", h["_id"]) for h in outcome_signals
        ]

        # Score accuracy based on outcome evidence
        if not outcome_signals:
            # No outcome evidence found
            verdict = CalibrationVerdict.INSUFFICIENT_EVIDENCE
            accuracy_score = pred_confidence  # Assume neither right nor wrong
            insufficient += 1
        else:
            # Evidence found — compute relevance-weighted accuracy
            avg_relevance = sum(
                h["_score"] for h in outcome_signals
            ) / len(outcome_signals)

            # Normalize: higher relevance = more likely prediction was correct
            # This is a heuristic — real calibration needs domain-specific scoring
            normalized_relevance = min(avg_relevance / 5.0, 1.0)

            if normalized_relevance > 0.6:
                verdict = CalibrationVerdict.CORRECT
                accuracy_score = min(normalized_relevance, 0.95)
                correct += 1
            elif normalized_relevance > 0.3:
                verdict = CalibrationVerdict.PARTIALLY_CORRECT
                accuracy_score = normalized_relevance
                correct += 1  # Count partial as correct for now
            else:
                verdict = CalibrationVerdict.INCORRECT
                accuracy_score = normalized_relevance
                incorrect += 1

        # Write validation result back to Elasticsearch
        try:
            await es.update(
                index="veridian_predictions",
                id=pred_id,
                body={
                    "doc": {
                        "validated_at": now.isoformat(),
                        "outcome": {
                            "accuracy_score": round(accuracy_score, 4),
                            "outcome_signal_ids": outcome_signal_ids,
                            "validated_by": "calibration_worker_v1",
                            "verdict": verdict.value,
                            "notes": (
                                f"Found {len(outcome_signals)} outcome signals. "
                                f"Verdict: {verdict.value}."
                            ),
                        },
                    }
                },
            )
        except Exception as e:
            logger.warning(
                "mcp.calibration_cycle.update_error",
                pred_id=pred_id,
                error=str(e),
            )

        # Accumulate results by type
        if pred_type not in results_by_type:
            results_by_type[pred_type] = []
        results_by_type[pred_type].append({
            "predicted_confidence": pred_confidence,
            "actual_accuracy": accuracy_score,
            "verdict": verdict.value,
        })

        # Generate weight update suggestion if significantly miscalibrated
        calibration_error = abs(pred_confidence - accuracy_score)
        if calibration_error > 0.2:
            direction = "reduce" if pred_confidence > accuracy_score else "increase"
            weight_updates.append({
                "prediction_type": pred_type,
                "action": f"{direction}_confidence",
                "magnitude": round(calibration_error, 4),
                "reason": f"Predicted {pred_confidence:.0%} but actual was {accuracy_score:.0%}",
            })

    # Step 3: Compute accuracy statistics by prediction type
    accuracy_by_type = []
    for pred_type, type_results in results_by_type.items():
        if not type_results:
            continue

        avg_predicted = sum(r["predicted_confidence"] for r in type_results) / len(type_results)
        avg_actual = sum(r["actual_accuracy"] for r in type_results) / len(type_results)
        cal_error = abs(avg_predicted - avg_actual)

        if cal_error < 0.1:
            verdict_str = "well_calibrated"
        elif avg_predicted > avg_actual:
            verdict_str = "overconfident"
        else:
            verdict_str = "underconfident"

        accuracy_by_type.append(AccuracyByType(
            prediction_type=pred_type,
            total_evaluated=len(type_results),
            avg_predicted_confidence=round(avg_predicted, 4),
            avg_actual_accuracy=round(avg_actual, 4),
            calibration_error=round(cal_error, 4),
            verdict=verdict_str,
        ))

    # Step 4: Detect systematic biases
    bias_findings = []
    overconfident_types = [a for a in accuracy_by_type if a.verdict == "overconfident"]
    if overconfident_types:
        types_str = ", ".join(a.prediction_type for a in overconfident_types)
        bias_findings.append(
            f"Systematic overconfidence detected in: {types_str}. "
            f"Consider reducing base confidence by 10-15% for these types."
        )

    total_evaluated = len(mature_predictions)
    overall_accuracy = (
        (correct / total_evaluated) if total_evaluated > 0 else 0.0
    )

    # Generate report text
    if total_evaluated == 0:
        report_text = (
            "No mature predictions to evaluate. "
            "Calibration will be meaningful once predictions have had time to validate."
        )
    else:
        report_text = (
            f"Calibration cycle complete. Evaluated {total_evaluated} predictions. "
            f"Overall accuracy: {overall_accuracy:.0%}. "
            f"Correct: {correct}, Incorrect: {incorrect}, "
            f"Insufficient evidence: {insufficient}. "
            f"{'Biases detected: ' + '; '.join(bias_findings) if bias_findings else 'No systematic biases detected.'}"
        )

    # Step 5: Write calibration report to Elasticsearch
    calibration_doc = {
        "calibration_id": calibration_id,
        "ran_at": now.isoformat(),
        "predictions_evaluated": total_evaluated,
        "accuracy_by_type": [a.model_dump() for a in accuracy_by_type],
        "weight_updates": weight_updates,
        "bias_findings": bias_findings,
        "report_text": report_text,
        "overall_accuracy": round(overall_accuracy, 4),
    }

    try:
        await es.index(
            index="veridian_calibration",
            id=calibration_id,
            document=calibration_doc,
        )
        logger.info(
            "mcp.calibration_cycle.report_stored",
            calibration_id=calibration_id,
        )
    except Exception as e:
        logger.warning(
            "mcp.calibration_cycle.report_store_error",
            error=str(e),
        )

    report = CalibrationReport(
        calibration_id=calibration_id,
        ran_at=now.isoformat(),
        predictions_evaluated=total_evaluated,
        predictions_correct=correct,
        predictions_incorrect=incorrect,
        predictions_insufficient_evidence=insufficient,
        overall_accuracy=round(overall_accuracy, 4),
        accuracy_by_type=accuracy_by_type,
        bias_findings=bias_findings,
        weight_updates=weight_updates,
        report_text=report_text,
        success=True,
    )

    logger.info(
        "mcp.calibration_cycle.complete",
        calibration_id=calibration_id,
        evaluated=total_evaluated,
        accuracy=round(overall_accuracy, 4),
    )

    return report