"""AI provider factory — Gemini and OpenCode Go (Grok, etc.)."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_providers.base import BaseAssessor

logger = logging.getLogger(__name__)

VALID_PROVIDERS = ("gemini", "opencode-go")

PROVIDER_META = {
    "gemini": {
        "label": "Google Gemini",
        "default_model": "gemini-2.0-flash-lite",
        "model_setting_key": "gemini_model",
        "supports_images": True,
        "api_key_env": "GEMINI_API_KEY",
    },
    "opencode-go": {
        "label": "OpenCode Go (Grok & more)",
        "default_model": "grok-4.5",
        "model_setting_key": "opencode_go_model",
        "supports_images": False,
        "api_key_env": "OPENCODE_GO_API_KEY",
    },
}

_CACHE: dict[str, BaseAssessor] = {}
_EBAY_CLIENT = None


def normalize_provider(value: str | None) -> str:
    """Return a valid provider id; default gemini."""
    if not value:
        return "gemini"
    v = value.strip().lower().replace("_", "-")
    aliases = {
        "go": "opencode-go",
        "opencode": "opencode-go",
        "opencodego": "opencode-go",
        "grok": "opencode-go",
        "google": "gemini",
    }
    v = aliases.get(v, v)
    return v if v in VALID_PROVIDERS else "gemini"


def _provider_has_api_key(pid: str) -> bool:
    """Check whether any known env var provides an API key for *pid*."""
    meta = PROVIDER_META[pid]
    if bool(os.environ.get(meta["api_key_env"], "").strip()):
        return True
    if pid == "opencode-go":
        for alias in ("OPENCODE_API_KEY", "DEEPSEEK_API_KEY"):
            if bool(os.environ.get(alias, "").strip()):
                return True
    return False


def list_providers() -> list[dict]:
    """Metadata for settings UI / health."""
    out = []
    for pid in VALID_PROVIDERS:
        meta = PROVIDER_META[pid]
        out.append(
            {
                "id": pid,
                "label": meta["label"],
                "default_model": meta["default_model"],
                "supports_images": meta["supports_images"],
                "configured": _provider_has_api_key(pid),
            }
        )
    return out


def _create_provider(provider: str) -> BaseAssessor:
    provider = normalize_provider(provider)
    if provider == "opencode-go":
        from ai_providers.opencode_go import OpenCodeGoAssessor

        return OpenCodeGoAssessor()
    from ai_providers.gemini import GeminiAssessor

    return GeminiAssessor()


def get_assessor(provider: str | None = None) -> BaseAssessor:
    """Return a cached assessor for *provider* (default: gemini)."""
    pid = normalize_provider(provider)
    if pid not in _CACHE:
        assessor = _create_provider(pid)
        if _EBAY_CLIENT is not None:
            assessor.set_ebay_client(_EBAY_CLIENT)
        _CACHE[pid] = assessor
        logger.info(
            "AI provider loaded: %s (enabled=%s, model=%s)",
            pid,
            assessor.enabled,
            assessor.model_name,
        )
    return _CACHE[pid]


def set_ebay_client_for_all(client) -> None:
    """Register the eBay client on every cached (and future) assessor."""
    global _EBAY_CLIENT
    _EBAY_CLIENT = client
    for assessor in _CACHE.values():
        assessor.set_ebay_client(client)


def apply_user_enabled(enabled: bool) -> None:
    """Sync the AI toggle onto all cached assessors."""
    for assessor in _CACHE.values():
        assessor.user_enabled = enabled


def create_assessor(provider: str | None = None) -> BaseAssessor:
    """Backward-compatible factory entry point."""
    if provider is None:
        provider = os.environ.get("AI_PROVIDER", "gemini")
    return get_assessor(provider)


def reset_assessors() -> None:
    """Clear cache (tests)."""
    _CACHE.clear()
