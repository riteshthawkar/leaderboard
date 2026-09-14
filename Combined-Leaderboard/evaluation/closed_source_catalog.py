"""Trusted metadata for retained closed-source visual-evaluation outputs."""

from __future__ import annotations


CLOSED_SOURCE_MODEL_CATALOG: dict[str, dict[str, object]] = {
    "gpt-4.1-mini_2025-04-14": {
        "slug": "gpt-4-1-mini-2025-04-14",
        "model_id": "gpt-4.1-mini-2025-04-14",
        "model_revision": "2025-04-14",
        "display_name": "GPT-4.1 mini (2025-04-14)",
        "organization": "OpenAI",
        "reasoning_profile": "nonthinking",
        "temperature": 0,
    },
    "gpt-4.1_2025-04-14": {
        "slug": "gpt-4-1-2025-04-14",
        "model_id": "gpt-4.1-2025-04-14",
        "model_revision": "2025-04-14",
        "display_name": "GPT-4.1 (2025-04-14)",
        "organization": "OpenAI",
        "reasoning_profile": "nonthinking",
        "temperature": 0,
    },
    "gpt-4o-mini_2024-07-18": {
        "slug": "gpt-4o-mini-2024-07-18",
        "model_id": "gpt-4o-mini-2024-07-18",
        "model_revision": "2024-07-18",
        "display_name": "GPT-4o mini (2024-07-18)",
        "organization": "OpenAI",
        "reasoning_profile": "nonthinking",
        "temperature": 0,
    },
    "gpt-4o_2024-11-20": {
        "slug": "gpt-4o-2024-11-20",
        "model_id": "gpt-4o-2024-11-20",
        "model_revision": "2024-11-20",
        "display_name": "GPT-4o (2024-11-20)",
        "organization": "OpenAI",
        "reasoning_profile": "nonthinking",
        "temperature": 0,
    },
    "gpt-5-mini_2025-08-07": {
        "slug": "gpt-5-mini-2025-08-07",
        "model_id": "gpt-5-mini-2025-08-07",
        "model_revision": "2025-08-07",
        "display_name": "GPT-5 mini (2025-08-07)",
        "organization": "OpenAI",
        "reasoning_profile": "thinking",
        "temperature": "provider_default",
    },
    "gpt-5-nano_2025-08-07": {
        "slug": "gpt-5-nano-2025-08-07",
        "model_id": "gpt-5-nano-2025-08-07",
        "model_revision": "2025-08-07",
        "display_name": "GPT-5 nano (2025-08-07)",
        "organization": "OpenAI",
        "reasoning_profile": "thinking",
        "temperature": "provider_default",
    },
    "gpt-5_2025-08-07": {
        "slug": "gpt-5-2025-08-07",
        "model_id": "gpt-5-2025-08-07",
        "model_revision": "2025-08-07",
        "display_name": "GPT-5 (2025-08-07)",
        "organization": "OpenAI",
        "reasoning_profile": "thinking",
        "temperature": "provider_default",
    },
    "grok-4.3_1": {
        "slug": "grok-4-3-1",
        "model_id": "grok-4.3_1",
        "model_revision": "4.3_1",
        "display_name": "Grok 4.3",
        "organization": "xAI",
        "reasoning_profile": "nonthinking",
        "temperature": 0,
    },
    "gemini-3.5-flash-lite": {
        "slug": "gemini-3-5-flash-lite",
        "model_id": "gemini-3.5-flash-lite",
        "model_revision": "stable-api-2026-07-31",
        "display_name": "Gemini 3.5 Flash-Lite",
        "organization": "Google",
        "reasoning_profile": "thinking",
        "temperature": "provider_default",
        "source_declared_temperature": 0,
        "temperature_behavior": "provider_deprecated_ignored",
        "thinking_level": "provider_default_not_verified_from_request_log",
        "evaluation_date": "2026-07-31",
        "provider_version_pinning": "stable_alias_without_immutable_backend_revision",
    },
}


def importer_catalog() -> dict[str, dict[str, str]]:
    """Return closed-source entries in the leaderboard importer's shape."""
    return {
        str(record["slug"]): {
            "repository": str(record["model_id"]),
            "model_revision": str(record["model_revision"]),
            "display_name": str(record["display_name"]),
            "organization": str(record["organization"]),
            "parameter_count": "",
            "reasoning_profile": str(record["reasoning_profile"]),
            "access": "closed",
        }
        for record in CLOSED_SOURCE_MODEL_CATALOG.values()
    }
