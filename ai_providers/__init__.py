"""AI provider factory — select Gemini or DeepSeek based on setting or environment."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_AI_PROVIDER_CACHE: dict | None = None


def create_assessor(provider: str | None = None):
    """Return an assessor instance.

    *provider* can be ``"gemini"`` (default) or ``"deepseek"``.
    When ``None`` the ``AI_PROVIDER`` env var is read (default ``"gemini"``).
    The instance is cached per provider so that switching providers at
    runtime (e.g. from the settings UI) recreates the assessor.
    """
    global _AI_PROVIDER_CACHE

    if provider is None:
        provider = os.environ.get("AI_PROVIDER", "gemini").strip().lower()
    else:
        provider = provider.strip().lower()

    if _AI_PROVIDER_CACHE is not None:
        cached_provider, cached_assessor = _AI_PROVIDER_CACHE
        if cached_provider == provider:
            return cached_assessor

    if provider == "deepseek":
        from ai_providers.deepseek import DeepSeekAssessor

        assessor = DeepSeekAssessor()
    else:
        from ai_providers.gemini import GeminiAssessor

        assessor = GeminiAssessor()

    _AI_PROVIDER_CACHE = (provider, assessor)
    logger.info("AI provider: %s (loaded=%s)", provider, assessor.enabled)
    return assessor
