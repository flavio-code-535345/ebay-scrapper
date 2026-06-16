"""AI provider factory — select Gemini or DeepSeek based on environment."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_AI_PROVIDER_CACHE: dict | None = None


def create_assessor():
    """Return an assessor instance based on the ``AI_PROVIDER`` env var.

    ``AI_PROVIDER`` can be ``"gemini"`` (default) or ``"deepseek"``.
    The instance is cached so that only one assessor is created per process.
    """
    global _AI_PROVIDER_CACHE
    if _AI_PROVIDER_CACHE is not None:
        return _AI_PROVIDER_CACHE

    provider = os.environ.get("AI_PROVIDER", "gemini").strip().lower()
    if provider == "deepseek":
        from ai_providers.deepseek import DeepSeekAssessor

        _AI_PROVIDER_CACHE = DeepSeekAssessor()
    else:
        from ai_providers.gemini import GeminiAssessor

        _AI_PROVIDER_CACHE = GeminiAssessor()

    logger.info("AI provider: %s (loaded=%s)", provider, _AI_PROVIDER_CACHE.enabled)
    return _AI_PROVIDER_CACHE
