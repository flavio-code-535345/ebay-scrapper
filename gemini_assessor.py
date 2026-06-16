"""Backward-compatibility shim — delegates to ``ai_providers`` package.

All original symbols are re-exported so that existing ``import`` and
``from gemini_assessor import …`` statements continue to work.
"""

from ai_providers.gemini import (  # noqa: F401
    GeminiAssessor,
    _GEMINI_REQUEST_TIMEOUT,
)
from ai_providers.base import (  # noqa: F401, E402
    _ASSESS_TOTAL_BUDGET_S,
    _BATCH_SIZE,
    _EBAY_PREFETCH_BUDGET_S,
    _apply_scam_override,
    _apply_sports_kinect_override,
    _build_single_game_search_query,
    _detect_bundle_individual_sale_scam,
    _detect_sports_kinect_deal,
    _extract_platform_name,
    _extract_potential_game_titles,
    _is_aggregate_placeholder,
    _sanitize_json_text,
    _MAX_GAMES_PER_BUNDLE,
)
