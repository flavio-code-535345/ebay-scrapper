"""AI provider factory — Gemini + OpenAI-compatible (OpenRouter, etc.)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_providers.base import BaseAssessor

logger = logging.getLogger(__name__)

VALID_PROVIDERS = ("gemini", "openai")

PROVIDER_META = {
    "gemini": {
        "label": "Google Gemini",
        "default_model": "gemini-2.0-flash-lite",
        "model_setting_key": "gemini_model",
        "api_key_env": "GEMINI_API_KEY",
    },
    "openai": {
        "label": "OpenAI Compatible (OpenRouter, etc.)",
        "default_model": "openai/gpt-4o",
        "model_setting_key": "ai_backend_model",
        "api_key_env": "AI_BACKEND_KEY",
    },
}

_CACHE: dict[str, BaseAssessor] = {}
_EBAY_CLIENT = None


def normalize_provider(value: str | None) -> str:
    if not value:
        return "gemini"
    v = value.strip().lower().replace("_", "-")
    aliases = {"openrouter": "openai", "openai": "openai", "gpt": "openai", "claude": "openai", "google": "gemini"}
    v = aliases.get(v, v)
    return v if v in VALID_PROVIDERS else "gemini"


def _create_provider(provider: str) -> BaseAssessor:
    provider = normalize_provider(provider)
    if provider == "openai":
        from ai_providers.openai_provider import OpenAICompatAssessor

        return OpenAICompatAssessor()
    from ai_providers.gemini import GeminiAssessor

    return GeminiAssessor()


def list_providers() -> list[dict]:
    out = []
    for pid in VALID_PROVIDERS:
        meta = PROVIDER_META[pid]
        configured = bool(os.environ.get(meta["api_key_env"], "").strip())
        out.append(
            {
                "id": pid,
                "label": meta["label"],
                "default_model": meta["default_model"],
                "configured": configured,
            }
        )
    return out


def get_assessor(provider: str) -> BaseAssessor:
    pid = normalize_provider(provider)
    if pid not in _CACHE:
        a = _create_provider(pid)
        if _EBAY_CLIENT is not None:
            a.set_ebay_client(_EBAY_CLIENT)
        _CACHE[pid] = a
        logger.info("AI provider loaded: %s (enabled=%s, model=%s)", pid, a.enabled, a.model_name)
    return _CACHE[pid]


def set_ebay_client_for_all(client) -> None:
    global _EBAY_CLIENT
    _EBAY_CLIENT = client
    for a in _CACHE.values():
        a.set_ebay_client(client)


def apply_user_enabled(enabled: bool) -> None:
    for a in _CACHE.values():
        a.user_enabled = enabled


def create_assessor(provider: str | None = None) -> BaseAssessor:
    if provider is None:
        provider = os.environ.get("AI_PROVIDER", "gemini")
    return get_assessor(provider)


def reset_assessors() -> None:
    _CACHE.clear()
