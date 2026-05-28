"""
Veridian Index Setup Script

Run this once to:
1. Verify Elasticsearch connection
2. Verify ELSER is deployed
3. Create all 6 Veridian indexes with proper mappings

Usage:
    python -m backend.es.indexes
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.es.client import get_client, close_client
from backend.es.mappings import ALL_INDEXES, ELSER_MODEL_ID


# ── Pretty output helpers ─────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def success(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}✗{RESET} {msg}")


def info(msg: str) -> None:
    print(f"{BLUE}→{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET} {msg}")


def header(msg: str) -> None:
    print(f"\n{BOLD}{msg}{RESET}")
    print("─" * len(msg))


# ── Setup steps ───────────────────────────────────────────────

async def verify_connection() -> bool:
    """Verify we can reach Elasticsearch."""
    header("STEP 1: Verifying Elasticsearch connection")
    es = get_client()
    try:
        info_response = await es.info()
        success(f"Connected to Elasticsearch {info_response['version']['number']}")
        success(f"Cluster: {info_response.get('cluster_name', 'serverless')}")
        return True
    except Exception as e:
        fail(f"Could not connect: {e}")
        return False


async def verify_cluster_health() -> bool:
    """Check cluster health status (skipped on serverless)."""
    header("STEP 2: Checking cluster health")
    es = get_client()
    try:
        health = await es.cluster.health()
        status = health["status"]
        if status == "green":
            success(f"Cluster status: {status}")
        elif status == "yellow":
            warn(f"Cluster status: {status} (acceptable for serverless)")
        else:
            fail(f"Cluster status: {status}")
            return False
        return True
    except Exception as e:
        warn(f"Cluster health unavailable (normal for serverless): {e}")
        return True


async def verify_elser_deployed() -> bool:
    """Verify ELSER model is deployed and running."""
    header("STEP 3: Verifying ELSER deployment")
    es = get_client()
    try:
        stats = await es.ml.get_trained_models_stats(
            model_id=ELSER_MODEL_ID
        )

        if not stats.get("trained_model_stats"):
            fail(f"Model {ELSER_MODEL_ID} not found")
            return False

        model_stats = stats["trained_model_stats"][0]
        deployment_stats = model_stats.get("deployment_stats", {})
        state = deployment_stats.get("state", "not_deployed")

        if state == "started":
            success(f"ELSER deployed and running: {ELSER_MODEL_ID}")
            return True
        else:
            fail(f"ELSER state: {state} (should be 'started')")
            warn("Go to Kibana → Machine Learning → Trained Models and deploy ELSER")
            return False
    except Exception as e:
        fail(f"Could not verify ELSER: {e}")
        warn("Make sure you deployed .elser_model_2_linux-x86_64 in Kibana")
        return False


async def create_indexes() -> bool:
    """Create all Veridian indexes with proper mappings."""
    header("STEP 4: Creating Veridian indexes")
    es = get_client()
    all_success = True

    for index_name, mapping_config in ALL_INDEXES.items():
        try:
            exists = await es.indices.exists(index=index_name)

            if exists:
                # Check if it has our proper mapping or was auto-created
                mapping = await es.indices.get_mapping(index=index_name)
                current_fields = set(
                    mapping[index_name]["mappings"]
                    .get("properties", {})
                    .keys()
                )
                expected_fields = set(mapping_config["mappings"]["properties"].keys())

                if expected_fields.issubset(current_fields):
                    warn(f"Index '{index_name}' already exists with correct mapping — skipping")
                    continue
                else:
                    warn(f"Index '{index_name}' exists but mapping is incomplete — recreating")
                    await es.indices.delete(index=index_name)

            await es.indices.create(
                index=index_name,
                mappings=mapping_config["mappings"],
            )
            success(f"Created index: {index_name}")

        except Exception as e:
            fail(f"Failed to create '{index_name}': {e}")
            all_success = False

    return all_success


async def verify_indexes() -> bool:
    """Verify all indexes are accessible."""
    header("STEP 5: Verifying indexes")
    es = get_client()
    all_success = True

    for index_name in ALL_INDEXES.keys():
        try:
            response = await es.indices.get(index=index_name)
            if index_name in response:
                success(f"Index accessible: {index_name}")
            else:
                fail(f"Index missing: {index_name}")
                all_success = False
        except Exception as e:
            fail(f"Could not access '{index_name}': {e}")
            all_success = False

    return all_success


async def write_test_document() -> bool:
    """Write a test signal to verify end-to-end functionality."""
    header("STEP 6: Writing test signal")
    es = get_client()

    test_signal = {
        "signal_id": "TEST-SETUP-001",
        "source": "setup_script",
        "source_id": "init",
        "raw_content": "Veridian setup verification signal. If you see this in Elasticsearch, the foundation is working.",
        "url": None,
        "timestamp": "2026-01-01T00:00:00Z",
        "ingested_at": "2026-01-01T00:00:00Z",
        "entity_hints": ["Veridian"],
        "signal_category": "system",
        "signal_type": "setup_verification",
        "urgency_score": 0.0,
        "impact_score": 0.0,
        "novelty_score": 1.0,
        "sentiment_score": 0.0,
        "chain_ids": [],
        "processed": True,
        "metadata": {"purpose": "setup_test"},
    }

    try:
        response = await es.index(
            index="veridian_signals",
            id=test_signal["signal_id"],
            document=test_signal,
        )

        if response.get("result") in ("created", "updated"):
            success(f"Test signal written: {response['result']}")

            # Read it back
            fetched = await es.get(
                index="veridian_signals",
                id=test_signal["signal_id"],
            )
            if fetched["_source"]["signal_id"] == test_signal["signal_id"]:
                success("Test signal read back successfully")
                return True
            else:
                fail("Test signal read back mismatch")
                return False
        else:
            fail(f"Unexpected write result: {response}")
            return False

    except Exception as e:
        fail(f"Test signal write failed: {e}")
        return False


# ── Main entry point ──────────────────────────────────────────

async def main() -> None:
    print(f"\n{BOLD}╔══════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║          VERIDIAN — Elasticsearch Setup           ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════╝{RESET}")

    steps = [
        ("Connection",         verify_connection),
        ("Cluster Health",     verify_cluster_health),
        ("ELSER Deployment",   verify_elser_deployed),
        ("Index Creation",     create_indexes),
        ("Index Verification", verify_indexes),
        ("End-to-End Test",    write_test_document),
    ]

    results = {}
    for name, step_fn in steps:
        try:
            results[name] = await step_fn()
        except Exception as e:
            fail(f"Unexpected error in {name}: {e}")
            results[name] = False

    # ── Final summary ─────────────────────────────────────────
    print(f"\n{BOLD}╔══════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║                     SUMMARY                        ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════╝{RESET}\n")

    for name, ok in results.items():
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")

    all_passed = all(results.values())

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}✓ Veridian foundation is ready.{RESET}")
        print(f"{BLUE}Next step: Build the signal ingestion pipeline.{RESET}\n")
    else:
        print(f"{RED}{BOLD}✗ Setup incomplete. Fix the issues above before continuing.{RESET}\n")

    await close_client()


if __name__ == "__main__":
    asyncio.run(main())