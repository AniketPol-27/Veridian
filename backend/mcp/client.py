# backend/mcp/client.py
"""
Veridian MCP Client — unified interface for all 6 tools.

This is the single entry point Gemini agents use to
interact with Elasticsearch. Every tool call goes through
here — logged, timed, and error-handled consistently.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from backend.mcp.schemas import (
    ComputeAnomalyInput,
    FindHistoricalPatternInput,
    FindRelatedSignalsInput,
    GetEntityStateInput,
    StorePredictionInput,
)
from backend.mcp import tools

logger = structlog.get_logger(__name__)


class VeridianMCPClient:
    """
    Unified MCP client for Veridian's 6 Elastic tools.

    Usage:
        mcp = VeridianMCPClient()

        # Find signals related to a query
        signals = await mcp.find_related_signals(
            query="OpenAI competitor hiring platform engineers",
            time_range_days=90,
        )

        # Check if an entity has anomalous signal patterns
        anomaly = await mcp.compute_entity_anomaly(
            entity_name="OpenAI",
            metric="impact_score",
        )
    """

    async def find_related_signals(self, **kwargs: Any):
        """
        Search Elasticsearch for semantically related signals.
        Uses ELSER hybrid search — semantic + keyword.
        """
        params = FindRelatedSignalsInput(**kwargs)
        return await self._call("find_related_signals", tools.find_related_signals, params)

    async def compute_entity_anomaly(self, **kwargs: Any):
        """
        Detect statistical anomalies in entity signal patterns via ES|QL.
        """
        params = ComputeAnomalyInput(**kwargs)
        return await self._call("compute_entity_anomaly", tools.compute_entity_anomaly_score, params)

    async def find_historical_pattern(self, **kwargs: Any):
        """
        Find historically similar causal chains via KNN vector search.
        """
        params = FindHistoricalPatternInput(**kwargs)
        return await self._call("find_historical_pattern", tools.find_historical_pattern_match, params)

    async def get_entity_state(self, **kwargs: Any):
        """
        Retrieve multi-dimensional entity state via ES|QL aggregations.
        """
        params = GetEntityStateInput(**kwargs)
        return await self._call("get_entity_state", tools.get_entity_state, params)

    async def store_prediction(self, **kwargs: Any):
        """
        Persist a prediction with full provenance for self-calibration.
        """
        params = StorePredictionInput(**kwargs)
        return await self._call("store_prediction", tools.store_prediction, params)

    async def run_calibration_cycle(self):
        """
        Run the self-improvement loop — evaluate mature predictions.
        """
        return await self._call("run_calibration_cycle", tools.run_calibration_cycle)

    async def _call(self, tool_name: str, fn, params=None):
        """
        Execute a tool call with timing and error handling.
        Every tool call is logged with duration for monitoring.
        """
        start = time.monotonic()
        logger.info("mcp.tool_call", tool=tool_name)

        try:
            result = await fn(params) if params is not None else await fn()
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "mcp.tool_call.complete",
                tool=tool_name,
                duration_ms=duration_ms,
            )
            return result

        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(
                "mcp.tool_call.error",
                tool=tool_name,
                duration_ms=duration_ms,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise