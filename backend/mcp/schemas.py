# backend/mcp/schemas.py
"""
Pydantic schemas for all MCP tool inputs and outputs.
Every tool has a typed input model and typed output model.
No raw dicts crossing boundaries — everything is validated.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ============================================================
# ENUMS
# ============================================================

class SignalCategory(str, Enum):
    NEWS = "news"
    COMMUNITY = "community"
    INTERNAL = "internal"
    FINANCIAL = "financial"
    HIRING = "hiring"
    REGULATORY = "regulatory"


class SignalType(str, Enum):
    FUNDING = "funding"
    LEADERSHIP = "leadership"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    HIRING = "hiring"
    COMPETITIVE = "competitive"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"
    INCIDENT = "incident"
    GENERAL_NEWS = "general_news"
    DEVELOPER_SENTIMENT = "developer_sentiment"
    COMMUNITY_DISCUSSION = "community_discussion"


class AnomalyDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"


class PredictionType(str, Enum):
    CHURN = "churn"
    REVENUE = "revenue"
    COMPETITIVE = "competitive"
    ATTRITION = "attrition"
    MARKET_SHIFT = "market_shift"
    OPERATIONAL = "operational"
    REGULATORY = "regulatory"


class CalibrationVerdict(str, Enum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# ============================================================
# TOOL 1: find_related_signals
# ============================================================

class FindRelatedSignalsInput(BaseModel):
    query: str = Field(
        ...,
        description="Natural language query to search for semantically related signals",
        min_length=3,
        max_length=500,
    )
    time_range_days: int = Field(
        default=90,
        description="How far back to search in days",
        ge=1,
        le=365,
    )
    min_score: float = Field(
        default=0.6,
        description="Minimum relevance score threshold (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    signal_types: list[str] | None = Field(
        default=None,
        description="Filter by specific signal types",
    )
    entity_filter: list[str] | None = Field(
        default=None,
        description="Filter by specific entity names",
    )
    limit: int = Field(
        default=20,
        description="Maximum number of signals to return",
        ge=1,
        le=100,
    )
    exclude_signal_ids: list[str] | None = Field(
        default=None,
        description="Signal IDs to exclude from results",
    )


class SignalResult(BaseModel):
    signal_id: str
    source: str
    signal_type: str
    signal_category: str
    entity_hints: list[str]
    raw_content: str
    timestamp: str
    urgency_score: float
    impact_score: float
    relevance_score: float
    url: str | None = None

    @property
    def content_preview(self) -> str:
        return self.raw_content[:200] + "..." if len(self.raw_content) > 200 else self.raw_content


class FindRelatedSignalsOutput(BaseModel):
    signals: list[SignalResult]
    total_found: int
    query: str
    time_range_days: int
    search_strategy: str  # "semantic" | "hybrid" | "keyword"

    @property
    def is_empty(self) -> bool:
        return len(self.signals) == 0

    def format_for_prompt(self) -> str:
        """Format signals for inclusion in a Gemini prompt."""
        if self.is_empty:
            return "No related signals found."

        lines = [f"Found {self.total_found} related signals:\n"]
        for i, sig in enumerate(self.signals, 1):
            lines.append(
                f"[{i}] ID: {sig.signal_id}\n"
                f"    Source: {sig.source} | Type: {sig.signal_type}\n"
                f"    Entities: {', '.join(sig.entity_hints) or 'none detected'}\n"
                f"    Relevance: {sig.relevance_score:.2f} | "
                f"Impact: {sig.impact_score:.2f} | "
                f"Urgency: {sig.urgency_score:.2f}\n"
                f"    Timestamp: {sig.timestamp}\n"
                f"    Content: {sig.content_preview}\n"
            )
        return "\n".join(lines)


# ============================================================
# TOOL 2: compute_entity_anomaly_score
# ============================================================

class ComputeAnomalyInput(BaseModel):
    entity_name: str = Field(
        ...,
        description="Name of the business entity to analyze",
        min_length=1,
    )
    metric: str = Field(
        default="impact_score",
        description="Which metric to compute anomaly on",
    )
    lookback_days: int = Field(
        default=90,
        description="Baseline period in days",
        ge=7,
        le=365,
    )
    current_window_days: int = Field(
        default=7,
        description="Current window to compare against baseline",
        ge=1,
        le=30,
    )

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        valid = {"impact_score", "urgency_score", "novelty_score", "sentiment_score"}
        if v not in valid:
            raise ValueError(f"metric must be one of {valid}")
        return v


class AnomalyResult(BaseModel):
    entity_name: str
    metric: str
    is_anomalous: bool
    z_score: float
    direction: AnomalyDirection
    baseline_mean: float
    current_mean: float
    baseline_count: int
    current_count: int
    baseline_std: float
    anomaly_magnitude: str  # "mild" | "moderate" | "severe" | "extreme"
    interpretation: str
    confidence: str  # "high" | "medium" | "low" based on sample size

    @property
    def severity_emoji(self) -> str:
        return {
            "mild": "🟡",
            "moderate": "🟠",
            "severe": "🔴",
            "extreme": "🚨",
        }.get(self.anomaly_magnitude, "⚪")

    def format_for_prompt(self) -> str:
        if not self.is_anomalous:
            return (
                f"Entity '{self.entity_name}' shows NO anomaly in {self.metric}. "
                f"Z-score: {self.z_score:.2f} (threshold: ±2.0). "
                f"Baseline: {self.baseline_mean:.3f}, Current: {self.current_mean:.3f}."
            )
        return (
            f"{self.severity_emoji} ANOMALY DETECTED for '{self.entity_name}'\n"
            f"  Metric: {self.metric}\n"
            f"  Z-score: {self.z_score:.2f} ({self.anomaly_magnitude} anomaly)\n"
            f"  Direction: {self.direction.value}\n"
            f"  Baseline ({self.baseline_count} signals): {self.baseline_mean:.3f}\n"
            f"  Current ({self.current_count} signals): {self.current_mean:.3f}\n"
            f"  Confidence: {self.confidence}\n"
            f"  Interpretation: {self.interpretation}"
        )


# ============================================================
# TOOL 3: find_historical_pattern_match
# ============================================================

class FindHistoricalPatternInput(BaseModel):
    current_signal_ids: list[str] = Field(
        ...,
        description="Signal IDs to build the current pattern fingerprint from",
        min_length=1,
    )
    top_k: int = Field(
        default=5,
        description="Number of historical pattern matches to return",
        ge=1,
        le=20,
    )
    min_similarity: float = Field(
        default=0.5,
        description="Minimum similarity score to include in results",
        ge=0.0,
        le=1.0,
    )


class PatternMatch(BaseModel):
    chain_id: str
    similarity_score: float
    chain_summary: str
    outcome_summary: str
    outcome_type: str
    historical_accuracy: float
    time_to_outcome_days: int | None
    confidence_at_creation: float
    created_at: str

    def format_for_prompt(self) -> str:
        return (
            f"Historical Pattern (similarity: {self.similarity_score:.2f}):\n"
            f"  Chain: {self.chain_id}\n"
            f"  What happened: {self.chain_summary}\n"
            f"  Outcome: {self.outcome_summary}\n"
            f"  Time to outcome: {self.time_to_outcome_days or 'unknown'} days\n"
            f"  Historical accuracy: {self.historical_accuracy:.0%}\n"
            f"  Original confidence: {self.confidence_at_creation:.0%}"
        )


class FindHistoricalPatternOutput(BaseModel):
    matches: list[PatternMatch]
    total_searched: int
    fingerprint_signal_count: int
    has_strong_precedent: bool  # True if any match > 0.8 similarity

    def format_for_prompt(self) -> str:
        if not self.matches:
            return (
                "No historical patterns found similar to this signal cluster. "
                "This appears to be a novel situation — no precedent exists in memory."
            )

        lines = [
            f"Found {len(self.matches)} historical pattern matches "
            f"(searched {self.total_searched} chains):\n"
        ]
        for i, match in enumerate(self.matches, 1):
            lines.append(f"[{i}] {match.format_for_prompt()}\n")

        if self.has_strong_precedent:
            lines.append(
                "⚠ STRONG PRECEDENT EXISTS: At least one historical pattern "
                "is highly similar (>80%). Weight this heavily in confidence calibration."
            )

        return "\n".join(lines)


# ============================================================
# TOOL 4: get_entity_state
# ============================================================

class GetEntityStateInput(BaseModel):
    entity_name: str = Field(
        ...,
        description="Name of the entity to retrieve state for",
        min_length=1,
    )
    dimensions: list[str] = Field(
        default=["news", "community", "hiring", "financial"],
        description="Signal categories to retrieve state for",
    )
    lookback_days: int = Field(
        default=90,
        description="How far back to build entity state from",
        ge=7,
        le=365,
    )


class DimensionState(BaseModel):
    dimension: str
    signal_count: int
    avg_impact: float
    avg_urgency: float
    avg_sentiment: float
    latest_timestamp: str | None
    recent_content: str | None
    trend: str  # "accelerating" | "stable" | "declining" | "insufficient_data"

    def format_for_prompt(self) -> str:
        trend_emoji = {
            "accelerating": "📈",
            "stable": "➡️",
            "declining": "📉",
            "insufficient_data": "❓",
        }.get(self.trend, "❓")

        return (
            f"  {self.dimension.upper()} {trend_emoji}\n"
            f"    Signals: {self.signal_count} | "
            f"Avg Impact: {self.avg_impact:.2f} | "
            f"Avg Sentiment: {self.avg_sentiment:.2f}\n"
            f"    Latest: {self.latest_timestamp or 'none'}\n"
            f"    Recent: {(self.recent_content or 'none')[:120]}"
        )


class EntityState(BaseModel):
    entity_name: str
    entity_type: str | None
    total_signal_count: int
    first_seen: str | None
    last_seen: str | None
    overall_risk_score: float
    overall_opportunity_score: float
    dimension_states: dict[str, DimensionState]
    retrieved_at: str

    def format_for_prompt(self) -> str:
        lines = [
            f"Entity State: {self.entity_name}\n"
            f"  Total signals: {self.total_signal_count} | "
            f"First seen: {self.first_seen or 'unknown'} | "
            f"Last seen: {self.last_seen or 'unknown'}\n"
            f"  Risk score: {self.overall_risk_score:.2f} | "
            f"Opportunity score: {self.overall_opportunity_score:.2f}\n\n"
            f"  Signal dimensions:\n"
        ]
        for dim_state in self.dimension_states.values():
            lines.append(dim_state.format_for_prompt())
            lines.append("")

        return "\n".join(lines)


# ============================================================
# TOOL 5: store_prediction
# ============================================================

class ScenarioAnalysis(BaseModel):
    timeline_null: dict[str, Any] = Field(
        description="Outcome if no action taken"
    )
    timeline_alpha: dict[str, Any] = Field(
        description="Outcome if action set A taken"
    )
    timeline_beta: dict[str, Any] | None = Field(
        default=None,
        description="Outcome if action set B taken",
    )
    recommended_timeline: str = Field(
        description="Which timeline is recommended: null, alpha, or beta"
    )
    recommendation_reasoning: str


class StorePredictionInput(BaseModel):
    chain_id: str
    prediction_text: str = Field(..., min_length=20)
    prediction_type: PredictionType
    confidence: float = Field(..., ge=0.1, le=0.95)
    signal_ids: list[str] = Field(..., min_length=1)
    validate_in_days: int = Field(
        default=30,
        description="How many days until we check if this prediction was correct",
        ge=1,
        le=365,
    )
    scenario_analysis: ScenarioAnalysis | None = None
    key_assumptions: list[str] = Field(default_factory=list)
    stakes_level: int = Field(default=1, ge=1, le=4)


class StorePredictionOutput(BaseModel):
    prediction_id: str
    chain_id: str
    created_at: str
    validate_at: str
    confidence: float
    prediction_type: str
    success: bool
    message: str


# ============================================================
# TOOL 6: run_calibration_cycle
# ============================================================

class AccuracyByType(BaseModel):
    prediction_type: str
    total_evaluated: int
    avg_predicted_confidence: float
    avg_actual_accuracy: float
    calibration_error: float  # |predicted - actual|
    verdict: str  # "well_calibrated" | "overconfident" | "underconfident"

    def format_for_prompt(self) -> str:
        calibration_status = {
            "well_calibrated": "✅ Well calibrated",
            "overconfident": "⚠️ Overconfident",
            "underconfident": "📉 Underconfident",
        }.get(self.verdict, "❓ Unknown")

        return (
            f"  {self.prediction_type}: {calibration_status}\n"
            f"    Evaluated: {self.total_evaluated} predictions\n"
            f"    Predicted confidence: {self.avg_predicted_confidence:.0%} | "
            f"Actual accuracy: {self.avg_actual_accuracy:.0%}\n"
            f"    Calibration error: {self.calibration_error:.0%}"
        )


class CalibrationReport(BaseModel):
    calibration_id: str
    ran_at: str
    predictions_evaluated: int
    predictions_correct: int
    predictions_incorrect: int
    predictions_insufficient_evidence: int
    overall_accuracy: float
    accuracy_by_type: list[AccuracyByType]
    bias_findings: list[str]
    weight_updates: list[dict[str, Any]]
    report_text: str
    success: bool

    def format_for_prompt(self) -> str:
        lines = [
            f"Calibration Report — {self.ran_at}\n"
            f"  Evaluated: {self.predictions_evaluated} mature predictions\n"
            f"  Overall accuracy: {self.overall_accuracy:.0%}\n"
            f"  Correct: {self.predictions_correct} | "
            f"Incorrect: {self.predictions_incorrect} | "
            f"Insufficient evidence: {self.predictions_insufficient_evidence}\n\n"
            f"  Accuracy by prediction type:\n"
        ]
        for acc in self.accuracy_by_type:
            lines.append(acc.format_for_prompt())

        if self.bias_findings:
            lines.append("\n  Bias findings:")
            for finding in self.bias_findings:
                lines.append(f"    • {finding}")

        return "\n".join(lines)