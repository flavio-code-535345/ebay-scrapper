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
determine if a deal is a "Must Buy," "Fair," or "Hard Pass."

### ANALYSIS PROTOCOL
1. IMAGE SCAN:
   - Condition Check: Zoom into images to find scratches, dents, or signs of \
heavy wear not mentioned in the text.
   - Authenticity: Look for logos, serial numbers, or stitching patterns that \
indicate authenticity or counterfeits.
   - Completeness: Count the items in the photo. Are cables, boxes, or \
accessories missing?
   - Context: Does the photo look like a stock photo (Red Flag) or a real \
photo from a seller's home?

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

# Gemini model to use – gemini-1.5-flash supports multimodal (text + images).
_MODEL_NAME = "gemini-1.5-flash"

# Maximum number of listing images sent per request (keeps latency reasonable).
_MAX_IMAGES = 3

# Request timeout when downloading listing images (seconds).
_IMAGE_FETCH_TIMEOUT = 5


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

    def assess_deal(self, deal: Dict) -> Optional[Dict]:
        """Analyse *deal* with Gemini and return an AI-assessment dict.

        Returns ``None`` when:
        - The API key is not configured.
        - The Gemini API call fails (network error, quota, etc.).
        Callers should fall back to the rules-based engine in these cases.
        """
        if not self.enabled or self._client is None or self._types is None:
            return None

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
            logger.error(
                "GeminiAssessor: API error for listing %r: %s",
                deal.get("title", "?"),
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
            f"Seller Rating: {seller_rating}%\n\n"
            "Return your analysis in the required JSON format."
        )

        parts: List = [self._types.Part.from_text(text=text_prompt)]

        image_urls: List[str] = deal.get("image_urls", [])
        for url in image_urls[:_MAX_IMAGES]:
            image_part = self._fetch_image_part(url)
            if image_part is not None:
                parts.append(image_part)

        return parts

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
        text = text.strip()

        # Strip optional markdown code fences.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: try to find a JSON object anywhere in the text.
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                data = json.loads(obj_match.group())
            else:
                raise ValueError(f"No JSON object found in Gemini response: {text[:200]!r}")

        return {
            "ai_deal_rating": str(data.get("deal_rating", "Unknown")),
            "ai_confidence_score": int(data.get("confidence_score", 0)),
            "ai_visual_findings": data.get("visual_findings", []),
            "ai_red_flags": data.get("red_flags", []),
            "ai_fair_market_estimate": str(data.get("fair_market_estimate", "")),
            "ai_verdict_summary": str(data.get("verdict_summary", "")),
            "ai_assessed": True,
        }
