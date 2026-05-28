"""
Veridian Elasticsearch Client

Singleton async client for all Elasticsearch operations.
Centralizes connection configuration and error handling.
"""

import os
from pathlib import Path
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv


# Find .env at the project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# Load it explicitly
load_dotenv(dotenv_path=ENV_PATH)


def get_elasticsearch_client() -> AsyncElasticsearch:
    """
    Returns a configured AsyncElasticsearch client connected to
    Veridian's Elastic Cloud Serverless project.
    """
    endpoint = os.getenv("ELASTIC_ENDPOINT")
    api_key = os.getenv("ELASTIC_API_KEY")

    if not endpoint:
        raise ValueError(
            f"ELASTIC_ENDPOINT not set in .env\n"
            f"Looking for .env at: {ENV_PATH}\n"
            f"File exists: {ENV_PATH.exists()}"
        )
    if not api_key:
        raise ValueError(
            f"ELASTIC_API_KEY not set in .env\n"
            f"Looking for .env at: {ENV_PATH}"
        )

    client = AsyncElasticsearch(
        hosts=[endpoint],
        api_key=api_key,
        request_timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    return client


# Convenience singleton
_client: AsyncElasticsearch | None = None


def get_client() -> AsyncElasticsearch:
    """Returns the singleton Elasticsearch client."""
    global _client
    if _client is None:
        _client = get_elasticsearch_client()
    return _client


async def close_client() -> None:
    """Closes the singleton client. Call at app shutdown."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        