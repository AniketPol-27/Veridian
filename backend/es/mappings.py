"""
Veridian Elasticsearch Index Mappings

These mappings define the structure of every index in Veridian's
memory layer. Each field is carefully chosen to support:
- Semantic search (ELSER sparse_vector)
- Vector similarity (dense_vector for KNN)
- Aggregations (ES|QL analytics)
- Time-series queries

Note: Elastic Cloud Serverless manages shards and replicas
automatically — we do not specify number_of_shards or
number_of_replicas.
"""

# ── ELSER model ID — must match what's deployed ───────────────
ELSER_MODEL_ID = ".elser_model_2_linux-x86_64"


# ════════════════════════════════════════════════════════════
# INDEX 1: veridian_signals
# Every signal Veridian ingests — the raw memory
# ════════════════════════════════════════════════════════════
VERIDIAN_SIGNALS = {
    "mappings": {
        "properties": {
            "signal_id":          {"type": "keyword"},
            "source":             {"type": "keyword"},
            "source_id":          {"type": "keyword"},
            "raw_content":        {"type": "text"},
            "url":                {"type": "keyword"},
            "timestamp":          {"type": "date"},
            "ingested_at":        {"type": "date"},
            "entity_hints":       {"type": "keyword"},
            "signal_category":    {"type": "keyword"},
            "signal_type":        {"type": "keyword"},
            "urgency_score":      {"type": "float"},
            "impact_score":       {"type": "float"},
            "novelty_score":      {"type": "float"},
            "sentiment_score":    {"type": "float"},
            "chain_ids":          {"type": "keyword"},
            "processed":          {"type": "boolean"},
            # ELSER sparse embedding for semantic search
            "semantic_embedding": {"type": "sparse_vector"},
            # Dense embedding for KNN similarity search
            "dense_embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
            "metadata": {"type": "object", "enabled": True},
        }
    },
}


# ════════════════════════════════════════════════════════════
# INDEX 2: veridian_causal_chains
# Reasoning chains constructed by the 7-step protocol
# ════════════════════════════════════════════════════════════
VERIDIAN_CAUSAL_CHAINS = {
    "mappings": {
        "properties": {
            "chain_id":              {"type": "keyword"},
            "trigger_signal_id":     {"type": "keyword"},
            "center_entity":         {"type": "keyword"},
            "proposed_cause":        {"type": "text"},
            "proposed_effect":       {"type": "text"},
            "predicted_timeline_days": {"type": "integer"},
            "confidence":            {"type": "float"},
            "supporting_signal_ids": {"type": "keyword"},
            "contradicting_signal_ids": {"type": "keyword"},
            "key_assumptions":       {"type": "text"},
            "disconfirmation_evidence": {"type": "text"},
            "hypotheses_generated":  {"type": "integer"},
            "hypotheses_eliminated": {"type": "integer"},
            "survived_disconfirmation": {"type": "boolean"},
            "status":                {"type": "keyword"},
            "created_at":            {"type": "date"},
            "validate_at":           {"type": "date"},
            # Chain fingerprint for historical pattern matching
            "chain_embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
            "chain_summary":         {"type": "text"},
            "outcome_summary":       {"type": "text"},
            "actual_accuracy":       {"type": "float"},
            "time_to_outcome_days":  {"type": "integer"},
            # Full reasoning audit trail (not indexed, just stored)
            "reasoning_steps": {"type": "object", "enabled": False},
        }
    },
}


# ════════════════════════════════════════════════════════════
# INDEX 3: veridian_predictions
# Every prediction Veridian makes with full provenance
# ════════════════════════════════════════════════════════════
VERIDIAN_PREDICTIONS = {
    "mappings": {
        "properties": {
            "prediction_id":     {"type": "keyword"},
            "chain_id":          {"type": "keyword"},
            "created_at":        {"type": "date"},
            "validate_at":       {"type": "date"},
            "validated_at":      {"type": "date"},
            "prediction_text":   {"type": "text"},
            "prediction_type":   {"type": "keyword"},
            "confidence":        {"type": "float"},
            "signal_ids":        {"type": "keyword"},
            "scenario_analysis": {"type": "object", "enabled": False},
            "chain_embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine",
            },
            "outcome": {
                "type": "object",
                "properties": {
                    "accuracy_score":      {"type": "float"},
                    "outcome_signal_ids":  {"type": "keyword"},
                    "validated_by":        {"type": "keyword"},
                    "notes":               {"type": "text"},
                },
            },
        }
    },
}


# ════════════════════════════════════════════════════════════
# INDEX 4: veridian_entities
# Living knowledge graph of every business entity Veridian tracks
# ════════════════════════════════════════════════════════════
VERIDIAN_ENTITIES = {
    "mappings": {
        "properties": {
            "entity_id":         {"type": "keyword"},
            "entity_name":       {"type": "keyword"},
            "entity_type":       {"type": "keyword"},
            "first_seen":        {"type": "date"},
            "last_updated":      {"type": "date"},
            "signal_count":      {"type": "integer"},
            "risk_score":        {"type": "float"},
            "opportunity_score": {"type": "float"},
            "dimension_states":  {"type": "object", "enabled": True},
            "chain_ids":         {"type": "keyword"},
            "aliases":           {"type": "keyword"},
            "metadata":          {"type": "object", "enabled": True},
        }
    },
}


# ════════════════════════════════════════════════════════════
# INDEX 5: veridian_actions
# Every action Veridian takes or proposes
# ════════════════════════════════════════════════════════════
VERIDIAN_ACTIONS = {
    "mappings": {
        "properties": {
            "action_id":      {"type": "keyword"},
            "chain_id":       {"type": "keyword"},
            "prediction_id":  {"type": "keyword"},
            "action_type":    {"type": "keyword"},
            "stakes_level":   {"type": "integer"},
            "action_data":    {"type": "object", "enabled": False},
            "status":         {"type": "keyword"},
            "proposed_at":    {"type": "date"},
            "decided_at":     {"type": "date"},
            "human_decision": {"type": "keyword"},
            "executed_at":    {"type": "date"},
            "outcome":        {"type": "object", "enabled": True},
        }
    },
}


# ════════════════════════════════════════════════════════════
# INDEX 6: veridian_calibration
# Self-improvement records — every weight update Veridian makes
# ════════════════════════════════════════════════════════════
VERIDIAN_CALIBRATION = {
    "mappings": {
        "properties": {
            "calibration_id":      {"type": "keyword"},
            "ran_at":              {"type": "date"},
            "predictions_evaluated": {"type": "integer"},
            "accuracy_by_type":    {"type": "object", "enabled": True},
            "weight_updates":      {"type": "object", "enabled": True},
            "bias_findings":       {"type": "object", "enabled": True},
            "report_text":         {"type": "text"},
        }
    },
}


# ════════════════════════════════════════════════════════════
# ALL INDEXES — used by the setup script
# ════════════════════════════════════════════════════════════
ALL_INDEXES = {
    "veridian_signals":        VERIDIAN_SIGNALS,
    "veridian_causal_chains":  VERIDIAN_CAUSAL_CHAINS,
    "veridian_predictions":    VERIDIAN_PREDICTIONS,
    "veridian_entities":       VERIDIAN_ENTITIES,
    "veridian_actions":        VERIDIAN_ACTIONS,
    "veridian_calibration":    VERIDIAN_CALIBRATION,
}