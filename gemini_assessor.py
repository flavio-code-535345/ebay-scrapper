#!/usr/bin/env python3
"""
Gemini AI Deal Assessor
Analyzes eBay listings using Google Gemini multimodal API (text + images).
Falls back gracefully when the API key is absent or a request fails.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# "eBay Deal Sniper" system instruction (provided by user)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
### ROLE
You are the "eBay Deal Sniper," a professional resale expert and professional \
authenticator. Your goal is to analyze eBay listings (text + images) to \
determine if a deal is a "Must Buy," "Fair," or "Avoid."

### ANALYSIS PROTOCOL
1. IMAGE SCAN:
   - Condition Check: Zoom into images to find scratches, dents, or signs of \
heavy wear not mentioned in the text.
   - Authenticity: Look for logos, serial numbers, or stitching patterns that \
indicate authenticity or counterfeits.
   - Completeness: Count the items in the photo. Are cables, boxes, or \
accessories missing?
   - Placeholder / Stock Photo Detection (CRITICAL Red Flag): If the image \
appears to be a manufacturer stock photo, a generic product render, or a \
watermarked image rather than an actual photo of the seller's item, add \
"Stock/placeholder photo detected" to red_flags and lower confidence \
accordingly. Real seller photos show the actual item in a home/desk/table \
setting with natural lighting.
   - Missing Images: If image_issues contains "no_images" or "low_res_only", \
treat this as a significant red flag — a seller who doesn't provide real, \
high-resolution photos of their bundle is a risk.

2. TEXTUAL DATA SCAN:
   - Description Analysis: Flag phrases like "Untested," "For parts only," \
or "As-is."
   - Specifics: Check "Item Specifics" for discrepancies (e.g., Title says \
'New' but specifics say 'Used').
   - Seller Reputation: Factor in seller feedback and location if provided.

3. DEAL ASSESSMENT:
   - Compare the current price + shipping cost against the perceived market \
value of the item's condition.
   - Calculate a "Risk Score" (1-10) based on photo clarity and description \
detail.

### OUTPUT FORMAT
You MUST return your analysis in a structured JSON format with the following \
keys:
- "deal_rating": (Must Buy / Fair / Avoid)
- "confidence_score": (1-100)
- "visual_findings": (List any damage or missing parts found in photos)
- "red_flags": (List any suspicious text or photo details)
- "fair_market_estimate": (Based on condition)
- "verdict_summary": (A 2-sentence explanation of your choice)
"""

_BATCH_SYSTEM_PROMPT = """\
### ROLE
You are the "eBay Deal Sniper," a professional resale expert and professional \
authenticator. Your goal is to analyze multiple eBay listings (text + images) \
and determine if each deal is a "Must Buy," "Fair," or "Avoid."

### ANALYSIS PROTOCOL
Apply the following to EACH listing:
1. IMAGE SCAN:
   - Condition Check: Zoom into images to find scratches, dents, or signs of \
heavy wear not mentioned in the text.
   - Authenticity: Look for logos, serial numbers, or stitching patterns that \
indicate authenticity or counterfeits.
   - Completeness: Count the items in the photo. Are cables, boxes, or \
accessories missing?
   - Placeholder / Stock Photo Detection (CRITICAL Red Flag): If the image \
appears to be a manufacturer stock photo, a generic product render, or a \
watermarked image rather than an actual photo of the seller's item, add \
"Stock/placeholder photo detected" to red_flags and lower confidence \
accordingly. Real seller photos show the actual item in a home/desk/table \
setting with natural lighting.
   - Missing Images: If image_issues contains "no_images" or "low_res_only", \
treat this as a significant red flag — a seller who doesn't provide real, \
high-resolution photos of their bundle is a risk.

2. TEXTUAL DATA SCAN:
   - Description Analysis: Flag phrases like "Untested," "For parts only," \
or "As-is."
   - Specifics: Check "Item Specifics" for discrepancies (e.g., Title says \
'New' but specifics say 'Used').
   - Seller Reputation: Factor in seller feedback and location if provided.

3. DEAL ASSESSMENT:
   - Compare the current price + shipping cost against the perceived market \
value of the item's condition.
   - Calculate a "Risk Score" (1-10) based on photo clarity and description \
detail.

### OUTPUT FORMAT
You MUST return a **single JSON array** where each element corresponds to one \
listing in the order they were presented. Each element must have these keys:
- "deal_rating": (Must Buy / Fair / Avoid)
- "confidence_score": (1-100)
- "visual_findings": (List any damage or missing parts found in photos)
- "red_flags": (List any suspicious text or photo details)
- "fair_market_estimate": (Based on condition)
- "verdict_summary": (A 2-sentence explanation of your choice)

CRITICAL: Output ONLY the JSON array — no markdown fences, no explanation \
text, no concatenated separate objects. The entire response must be parseable \
as a single `json.loads()` call that returns a list.
"""

# Gemini model to use – gemini-flash-latest supports multimodal (text + images).
_MODEL_NAME = "gemini-flash-latest"

# Maximum number of listing images sent per item per request (keeps latency reasonable).
_MAX_IMAGES = 3

# Request timeout when downloading listing images (seconds).
_IMAGE_FETCH_TIMEOUT = 5

# Default back-off (seconds) when no retryDelay is provided by the API.
_DEFAULT_BACKOFF_SECONDS = 60

# Maximum number of deals bundled into a single Gemini generateContent call.
_BATCH_SIZE = 50

# Retry configuration for transient (non-rate-limit) API errors.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled on each retry (exponential back-off)

# ---------------------------------------------------------------------------
# Rate-limit / back-off state (shared across all GeminiAssessor instances).
# ---------------------------------------------------------------------------
_rate_limit_lock = threading.Lock()
_rate_limited_until: float = 0.0  # monotonic timestamp


def _extract_json_objects(text: str) -> list:
    """Extract all top-level JSON values (objects or arrays) from *text*.

    Handles Gemini responses that return concatenated JSON objects instead of a
    single JSON array, e.g. ``{"a":1}{"b":2}`` instead of ``[{"a":1},{"b":2}]``.
    Returns a (possibly empty) list of decoded Python values.
    """
    decoder = json.JSONDecoder()
    results = []
    pos = 0
    while pos < len(text):
        # Skip whitespace and stray commas between objects.
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(text):
            break
        if text[pos] in "{[":
            try:
                obj, end_pos = decoder.raw_decode(text, pos)
                results.append(obj)
                pos = end_pos
            except json.JSONDecodeError:
                pos += 1
        else:
            pos += 1
    return results


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a 429 / RESOURCE_EXHAUSTED error."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* is a transient error that warrants a retry.

    Rate-limit errors are NOT transient (they are handled separately with a
    long back-off window).  Network timeouts and 5xx server errors are.
    """
    if _is_rate_limit_error(exc):
        return False
    msg = str(exc).lower()
    return any(x in msg for x in ("timeout", "connection", "503", "500", "502", "504"))


def _parse_retry_delay(exc: Exception) -> Optional[float]:
    """Try to extract retryDelay (seconds) from the Gemini API error payload."""
    try:
        msg = str(exc)
        # Match patterns like "retryDelay": "30s" or "retry_delay": "30s"
        match = re.search(r'"retry[_\s]?[Dd]elay"\s*:\s*"(\d+(?:\.\d+)?)s?"', msg)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


class GeminiAssessor:
    """Wraps the Gemini API for multimodal eBay deal assessment."""

    def __init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.enabled = bool(api_key)
        self._client = None
        self._types = None

        if self.enabled:
            try:
                from google import genai  # lazy import
                from google.genai import types

                self._client = genai.Client(api_key=api_key)
                self._types = types
                logger.info("GeminiAssessor: Gemini API initialised (model=%s)", _MODEL_NAME)
            except Exception as exc:
                logger.error("GeminiAssessor: Failed to initialise Gemini client: %s", exc)
                self.enabled = False
        else:
            logger.info(
                "GeminiAssessor: GEMINI_API_KEY not set — AI assessment disabled; "
                "falling back to rules engine."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rate_limited_until(self) -> float:
        """Return the monotonic timestamp until which Gemini is rate-limited (0 if none)."""
        with _rate_limit_lock:
            return _rate_limited_until

    @property
    def is_rate_limited(self) -> bool:
        """Return True if the Gemini quota is currently exhausted."""
        return time.monotonic() < self.rate_limited_until

    def assess_deal(self, deal: Dict) -> Optional[Dict]:
        """Analyse *deal* with Gemini and return an AI-assessment dict.

        Returns ``None`` when:
        - The API key is not configured.
        - The Gemini API call fails (network error, etc.).
        Returns a dict with ``ai_error_type`` set when:
        - A 429 rate-limit is hit (``ai_error_type="rate_limit"``).
        - The response cannot be parsed as JSON (``ai_error_type="parse_error"``).
        Callers should fall back to the rules-based engine in these cases.
        """
        global _rate_limited_until

        if not self.enabled or self._client is None or self._types is None:
            return None

        # If we are currently in a rate-limit back-off window, skip the call.
        pause_remaining = self.rate_limited_until - time.monotonic()
        if pause_remaining > 0:
            logger.warning(
                "GeminiAssessor: rate-limited – skipping AI assessment for %r "
                "(%.0f s remaining in back-off).",
                deal.get("title", "?"),
                pause_remaining,
            )
            return {"ai_error_type": "rate_limit", "ai_assessed": False}

        try:
            contents = self._build_contents(deal)
            # The new google.genai SDK requires per-request config; there is no
            # global model object that holds a system instruction.
            response = self._client.models.generate_content(
                model=_MODEL_NAME,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                ),
            )
            return self._parse_response(response.text)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                with _rate_limit_lock:
                    _rate_limited_until = time.monotonic() + delay
                logger.warning(
                    "GeminiAssessor: 429 RESOURCE_EXHAUSTED for %r – "
                    "backing off %.0f s. Gemini AI temporarily paused.",
                    deal.get("title", "?"),
                    delay,
                )
                return {"ai_error_type": "rate_limit", "ai_assessed": False}

            logger.error(
                "GeminiAssessor: API error for listing %r: %s",
                deal.get("title", "?"),
                exc,
            )
            return None

    def assess_deals_batch(self, deals: List[Dict]) -> List[Optional[Dict]]:
        """Assess a list of *deals* in as few Gemini requests as possible.

        Deals are grouped into batches of up to ``_BATCH_SIZE`` items and sent
        together in a single ``generateContent`` call, dramatically reducing
        API quota consumption compared to one call per deal.

        Returns a list of the same length as *deals*.  Each element is either:
        - A dict with AI assessment fields (``ai_assessed=True``).
        - A dict with ``ai_assessed=False`` and ``ai_error_type`` set.
        - ``None`` if the assessor is disabled or an unrecoverable error occurred.
        """
        if not self.enabled or self._client is None or self._types is None:
            return [None] * len(deals)

        if not deals:
            return []

        # If we are currently in a rate-limit back-off window, skip everything.
        pause_remaining = self.rate_limited_until - time.monotonic()
        if pause_remaining > 0:
            logger.warning(
                "GeminiAssessor: rate-limited – skipping batch of %d items "
                "(%.0f s remaining in back-off).",
                len(deals),
                pause_remaining,
            )
            return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)

        results: List[Optional[Dict]] = []
        for batch_start in range(0, len(deals), _BATCH_SIZE):
            batch = deals[batch_start : batch_start + _BATCH_SIZE]
            batch_results = self._assess_batch_with_retry(batch)
            results.extend(batch_results)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_image_issues_line(deal: Dict) -> str:
        """Return a formatted 'Image Issues: …\n' line for a deal, or empty string."""
        issues: List[str] = deal.get("image_issues", [])
        return f"Image Issues: {', '.join(issues)}\n" if issues else ""

    def _build_contents(self, deal: Dict) -> List:
        """Construct the Gemini contents list (text + image parts)."""
        title = deal.get("title", "Unknown")
        price = deal.get("price", 0)
        condition = deal.get("condition", "Unknown")
        shipping = deal.get("shipping", "Unknown")
        seller_rating = deal.get("seller_rating", 0)

        text_prompt = (
            f"Analyze this eBay listing:\n\n"
            f"Title: {title}\n"
            f"Price: €{price:.2f}\n"
            f"Condition: {condition}\n"
            f"Shipping: {shipping}\n"
            f"Seller Rating: {seller_rating}%\n"
            f"{self._format_image_issues_line(deal)}"
            "\nReturn your analysis in the required JSON format."
        )

        parts: List = [self._types.Part.from_text(text=text_prompt)]

        image_urls: List[str] = deal.get("image_urls", [])
        for url in image_urls[:_MAX_IMAGES]:
            image_part = self._fetch_image_part(url)
            if image_part is not None:
                parts.append(image_part)

        return parts

    def _build_batch_contents(self, deals: List[Dict]) -> List:
        """Construct the Gemini contents list for a batch of *deals*.

        Each deal is introduced with a numbered separator so that Gemini can
        unambiguously map its array response back to the original items.
        """
        parts: List = []

        intro = (
            f"Below are {len(deals)} eBay listings to analyze. "
            f"Return a JSON array of exactly {len(deals)} objects in the same "
            "order. Each object must contain: deal_rating, confidence_score, "
            "visual_findings, red_flags, fair_market_estimate, verdict_summary."
        )
        parts.append(self._types.Part.from_text(text=intro))

        for idx, deal in enumerate(deals, 1):
            title = deal.get("title", "Unknown")
            price = deal.get("price", 0)
            condition = deal.get("condition", "Unknown")
            shipping = deal.get("shipping", "Unknown")
            seller_rating = deal.get("seller_rating", 0)

            item_text = (
                f"\n--- ITEM {idx} ---\n"
                f"Title: {title}\n"
                f"Price: €{price:.2f}\n"
                f"Condition: {condition}\n"
                f"Shipping: {shipping}\n"
                f"Seller Rating: {seller_rating}%\n"
                f"{self._format_image_issues_line(deal)}"
            )
            parts.append(self._types.Part.from_text(text=item_text))

            for url in deal.get("image_urls", [])[:_MAX_IMAGES]:
                image_part = self._fetch_image_part(url)
                if image_part is not None:
                    parts.append(image_part)

        parts.append(
            self._types.Part.from_text(
                text=(
                    f"\nNow return a JSON array of exactly {len(deals)} analysis "
                    "objects, one per item in order, with no other text."
                )
            )
        )
        return parts

    def _assess_batch_with_retry(self, deals: List[Dict]) -> List[Optional[Dict]]:
        """Send *deals* as a single batch request, retrying on transient errors."""
        global _rate_limited_until

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                contents = self._build_batch_contents(deals)
                response = self._client.models.generate_content(
                    model=_MODEL_NAME,
                    contents=contents,
                    config=self._types.GenerateContentConfig(
                        system_instruction=_BATCH_SYSTEM_PROMPT,
                    ),
                )
                return self._parse_batch_response(response.text, len(deals))
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                    with _rate_limit_lock:
                        _rate_limited_until = time.monotonic() + delay
                    logger.warning(
                        "GeminiAssessor: 429 RESOURCE_EXHAUSTED (batch of %d) – "
                        "backing off %.0f s. Gemini AI temporarily paused.",
                        len(deals),
                        delay,
                    )
                    return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)

                last_exc = exc
                if _is_transient_error(exc) and attempt < _MAX_RETRIES - 1:
                    retry_delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "GeminiAssessor: Transient error on attempt %d/%d "
                        "(batch of %d) – retrying in %.1f s: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        len(deals),
                        retry_delay,
                        exc,
                    )
                    time.sleep(retry_delay)
                    continue

                # Non-transient error or retries exhausted.
                logger.error(
                    "GeminiAssessor: Batch API error (attempt %d/%d, batch of %d): %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    len(deals),
                    exc,
                )
                return [None] * len(deals)

        logger.error(
            "GeminiAssessor: All %d retries exhausted for batch of %d: %s",
            _MAX_RETRIES,
            len(deals),
            last_exc,
        )
        return [None] * len(deals)

    def _fetch_image_part(self, url: str):
        """Download *url* and return a Gemini-compatible image Part, or None."""
        try:
            resp = requests.get(url, timeout=_IMAGE_FETCH_TIMEOUT)
            resp.raise_for_status()
            mime_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            if not mime_type.startswith("image/"):
                mime_type = "image/jpeg"

            return self._types.Part.from_bytes(data=resp.content, mime_type=mime_type)
        except Exception as exc:
            logger.warning("GeminiAssessor: Could not fetch image %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_response(text: str) -> Dict:
        """Extract the JSON payload from Gemini's response text."""
        original_text = text
        text = text.strip()

        # Strip optional markdown code fences.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: try to find a JSON object anywhere in the text.
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                except json.JSONDecodeError as inner_exc:
                    logger.error(
                        "GeminiAssessor: JSON parse failed after extraction – %s. "
                        "Raw response (first 500 chars): %r",
                        inner_exc,
                        original_text[:500],
                    )
            else:
                logger.error(
                    "GeminiAssessor: No JSON object found in Gemini response. "
                    "Raw response (first 500 chars): %r",
                    original_text[:500],
                )

        if data is None:
            return {
                "ai_deal_rating": "Unknown",
                "ai_confidence_score": 0,
                "ai_visual_findings": [],
                "ai_red_flags": [],
                "ai_fair_market_estimate": "",
                "ai_verdict_summary": "AI response could not be parsed.",
                "ai_assessed": False,
                "ai_error_type": "parse_error",
            }

        return {
            "ai_deal_rating": str(data.get("deal_rating", "Unknown")),
            "ai_confidence_score": int(data.get("confidence_score", 0)),
            "ai_visual_findings": data.get("visual_findings", []),
            "ai_red_flags": data.get("red_flags", []),
            "ai_fair_market_estimate": str(data.get("fair_market_estimate", "")),
            "ai_verdict_summary": str(data.get("verdict_summary", "")),
            "ai_assessed": True,
        }

    @staticmethod
    def _parse_batch_response(text: str, expected_count: int) -> List[Dict]:
        """Parse a batch Gemini response as a JSON array.

        Returns a list of exactly *expected_count* assessment dicts.  Missing
        or unparseable items are filled with a parse-error sentinel so the
        caller always gets a list of the right length.
        """
        _parse_error: Dict = {
            "ai_deal_rating": "Unknown",
            "ai_confidence_score": 0,
            "ai_visual_findings": [],
            "ai_red_flags": [],
            "ai_fair_market_estimate": "",
            "ai_verdict_summary": "AI response could not be parsed.",
            "ai_assessed": False,
            "ai_error_type": "parse_error",
        }

        original_text = text
        text = text.strip()

        # Strip optional markdown code fences.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("GeminiAssessor: Direct JSON parse failed: %s", exc)

            # Fallback 1: find a JSON array anywhere in the text.
            arr_match = re.search(r"\[[\s\S]*\]", text)
            if arr_match:
                try:
                    data = json.loads(arr_match.group())
                except json.JSONDecodeError as inner_exc:
                    logger.warning(
                        "GeminiAssessor: Batch JSON array parse failed – %s. "
                        "Trying concatenated-object extraction.",
                        inner_exc,
                    )

            # Fallback 2: extract concatenated JSON objects (e.g. {...}{...}).
            if data is None:
                extracted = _extract_json_objects(text)
                if extracted:
                    items: List = []
                    for obj in extracted:
                        if isinstance(obj, list):
                            items.extend(obj)
                        elif isinstance(obj, dict):
                            items.append(obj)
                    if items:
                        data = items
                        logger.info(
                            "GeminiAssessor: Extracted %d items via "
                            "concatenated-object fallback.",
                            len(items),
                        )

            if data is None:
                logger.error(
                    "GeminiAssessor: All JSON parse strategies failed for batch. "
                    "Raw response (first 500 chars): %r",
                    original_text[:500],
                )

        # Normalise: if Gemini returned a single object, wrap it.
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return [dict(_parse_error)] * expected_count

        results: List[Dict] = []
        for item_data in data:
            if not isinstance(item_data, dict):
                results.append(dict(_parse_error))
            else:
                try:
                    confidence = int(float(item_data.get("confidence_score", 0)))
                except (TypeError, ValueError):
                    confidence = 0
                results.append(
                    {
                        "ai_deal_rating": str(item_data.get("deal_rating", "Unknown")),
                        "ai_confidence_score": confidence,
                        "ai_visual_findings": item_data.get("visual_findings", []),
                        "ai_red_flags": item_data.get("red_flags", []),
                        "ai_fair_market_estimate": str(
                            item_data.get("fair_market_estimate", "")
                        ),
                        "ai_verdict_summary": str(item_data.get("verdict_summary", "")),
                        "ai_assessed": True,
                    }
                )

        # Pad or truncate to match the expected count.
        if len(results) < expected_count:
            missing = expected_count - len(results)
            logger.warning(
                "GeminiAssessor: Batch response had %d items but expected %d; "
                "padding %d with parse-error sentinels.",
                len(results),
                expected_count,
                missing,
            )
            results.extend([dict(_parse_error)] * missing)
        elif len(results) > expected_count:
            results = results[:expected_count]

        return results
