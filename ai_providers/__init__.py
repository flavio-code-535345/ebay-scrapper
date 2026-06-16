"""AI provider factory — creates Gemini assessor only."""

from __future__ import annotations

import logging

from ai_providers.gemini import GeminiAssessor

logger = logging.getLogger(__name__)

_DEFAULT_ASSESSOR: GeminiAssessor | None = None


def create_assessor() -> GeminiAssessor:
    """Return a GeminiAssessor instance.

    The instance is cached so that the same assessor is reused.
    """
    global _DEFAULT_ASSESSOR

    if _DEFAULT_ASSESSOR is not None:
        return _DEFAULT_ASSESSOR

    assessor = GeminiAssessor()
    _DEFAULT_ASSESSOR = assessor
    logger.info("AI provider: gemini (loaded=%s)", assessor.enabled)
    return assessor
