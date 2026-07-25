"""AI provider factory — Gemini only."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_providers.base import BaseAssessor

logger = logging.getLogger(__name__)


def create_assessor(provider: str | None = None) -> BaseAssessor:
    from ai_providers.gemini import GeminiAssessor

    a = GeminiAssessor()
    logger.info("AI provider: gemini (loaded=%s)", a.enabled)
    return a


def reset_assessors() -> None:
    pass
