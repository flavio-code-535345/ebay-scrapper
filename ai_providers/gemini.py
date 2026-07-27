"""Gemini AI provider — uses google.genai Client to call Gemini models."""

from __future__ import annotations

import concurrent.futures
import os
import time

import requests

from ai_providers.base import (
    _ASSESS_TOTAL_BUDGET_S,
    _BATCH_SIZE,
    _BATCH_SYSTEM_PROMPT,
    _DEFAULT_BACKOFF_SECONDS,
    _MAX_RETRIES,
    _RETRY_BASE_DELAY,
    _SYSTEM_PROMPT,
    BaseAssessor,
    _apply_garbage_overrides,
    _apply_scam_override,
    _apply_sports_kinect_override,
    _build_deterministic_garbage,
    _detect_broken_deal,
    _detect_trash_title,
    _is_rate_limit_error,
    _is_transient_error,
    _parse_response,
    _parse_retry_delay,
    _set_rate_limited_until,
    extract_listed_game_prices,
    logger,
)

_GEMINI_REQUEST_TIMEOUT = 35
_MODEL_NAME = "gemini-3.1-flash-lite"
_MAX_IMAGES = 3
_IMAGE_FETCH_TIMEOUT = 5

# Known text-only Gemini models — auto-disable image input to avoid SDK errors.
_TEXT_ONLY_MODELS: frozenset[str] = frozenset()


def _is_text_only_model(model: str) -> bool:
    """Return True for known image-unsupported Gemini model names."""
    m = model.lower().strip()
    if m in _TEXT_ONLY_MODELS:
        return True
    # Heuristic: "lite" suffix models from 3.x+ are text-only
    if m.startswith("gemini-3") and "lite" in m:
        return True
    return m.startswith("gemini-2.5") and "lite" in m and "preview" in m


class GeminiAssessor(BaseAssessor):
    """AI assessor using Google Gemini models."""

    def __init__(self) -> None:
        super().__init__("GEMINI_API_KEY", _MODEL_NAME)
        self._images_supported = not _is_text_only_model(self._model_name)
        if not self._images_supported:
            logger.info("GeminiAssessor: model %r is text-only — images disabled.", self._model_name)

    @BaseAssessor.model_name.setter
    def model_name(self, value: str) -> None:
        BaseAssessor.model_name.fset(self, value)  # type: ignore[attr-defined]
        if _is_text_only_model(self._model_name):
            if self._images_supported:
                logger.info("GeminiAssessor: model %r is text-only — disabling images.", self._model_name)
                self._images_supported = False
        else:
            self._images_supported = True
        if self.enabled:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "").strip())
                self._types = types
                logger.info("GeminiAssessor: Gemini API initialised (model=%s)", self._model_name)
            except Exception as exc:
                logger.error("GeminiAssessor: Failed to initialise Gemini client: %s", exc)
                self.enabled = False
        else:
            logger.info(
                "GeminiAssessor: GEMINI_API_KEY not set — AI assessment disabled; falling back to rules engine."
            )

    # ── Single-deal assessment ────────────────────────────────────────────

    def assess_deal(self, deal: dict) -> dict | None:
        if not self.enabled or not self.user_enabled or self.is_rate_limited:
            return None
        # 1. Check deterministic rules — garbage first, then sports/Kinect, then scam.
        broken = _detect_broken_deal(deal)
        if broken:
            return _build_deterministic_garbage("Garbage", 100, broken)
        trash = _detect_trash_title(deal)
        if trash:
            return _build_deterministic_garbage("Garbage", 100, trash)
        sr = _detect_sports_kinect_deal(deal)
        if sr:
            return {
                "ai_deal_rating": "Avoid",
                "ai_confidence_score": 100,
                "ai_visual_findings": [],
                "ai_red_flags": ["Automatically flagged — sports/Kinect content"],
                "ai_fair_market_estimate": "",
                "ai_itemized_resale_estimates": [],
                "ai_estimated_total_cost": deal.get("price", 0) or 0,
                "ai_estimated_gross_profit": 0,
                "ai_verdict_summary": sr,
                "ai_assessed": True,
                "ai_potential_scam": False,
                "ai_scam_warning": "",
            }
        scam = _detect_bundle_individual_sale_scam(deal)
        if scam:
            return {
                "ai_deal_rating": "Avoid",
                "ai_confidence_score": 100,
                "ai_visual_findings": [],
                "ai_red_flags": [],
                "ai_fair_market_estimate": "",
                "ai_itemized_resale_estimates": [],
                "ai_estimated_total_cost": deal.get("price", 0) or 0,
                "ai_estimated_gross_profit": 0,
                "ai_verdict_summary": scam,
                "ai_assessed": True,
                "ai_potential_scam": True,
                "ai_scam_warning": scam,
            }
        try:
            contents = self._build_contents(deal)
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            assessment = _parse_response(response.text)
            assessment = _apply_garbage_overrides(deal, assessment)
            assessment = _apply_sports_kinect_override(deal, assessment)
            assessment = _apply_scam_override(deal, assessment)
            return assessment
        except Exception as exc:
            if _is_rate_limit_error(exc):
                delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                _set_rate_limited_until(time.monotonic() + delay)
                logger.warning("GeminiAssessor: 429 RESOURCE_EXHAUSTED – backing off %.0f s.", delay)
            exc_msg = str(exc).lower()
            if self._images_supported and (
                "does not support image" in exc_msg or "image input" in exc_msg or "cannot read" in exc_msg
            ):
                logger.info("GeminiAssessor: model %r is text-only — disabling images.", self._model_name)
                self._images_supported = False
                return self.assess_deal(deal)
            logger.error("GeminiAssessor: assess_deal failed: %s", exc, exc_info=True)
            return None

    # ── Batch assessment ──────────────────────────────────────────────────

    def assess_deals_batch(self, deals: list[dict]) -> list[dict | None]:
        if not self.enabled or not self.user_enabled or not deals or self.is_rate_limited:
            return [None] * len(deals) if deals else []
        self._prefetch_ebay_prices_parallel(deals)
        results: list[dict | None] = []
        t_start = time.monotonic()
        for batch_idx, batch_start in enumerate(range(0, len(deals), _BATCH_SIZE)):
            batch = deals[batch_start : batch_start + _BATCH_SIZE]
            elapsed = time.monotonic() - t_start
            if elapsed >= _ASSESS_TOTAL_BUDGET_S:
                logger.warning(
                    "GeminiAssessor: total budget exhausted after batch %d; returning %d unassessed deals as None.",
                    batch_idx,
                    len(deals) - len(results),
                )
                results.extend([None] * (len(deals) - len(results)))
                break
            batch_results = self._assess_batch_with_retry(batch)
            for deal, assessment in zip(batch, batch_results, strict=False):
                if isinstance(assessment, dict):
                    assessment = _apply_garbage_overrides(deal, assessment)
                    assessment = _apply_sports_kinect_override(deal, assessment)
                    assessment = _apply_scam_override(deal, assessment)
                results.append(assessment)
        return results

    # ── Prompt construction (Gemini-specific — uses self._types.Part) ─────

    def _build_contents(self, deal: dict) -> list:
        title = deal.get("title", "Unknown")
        price = deal.get("price", "?")
        condition = deal.get("condition", "?")
        seller_rating = deal.get("seller_rating", "?")
        shipping = deal.get("shipping", "?")
        description = deal.get("description", "")
        seller_count = deal.get("seller_count", "")
        item_location = deal.get("item_location", "")
        listing_date = deal.get("listing_date", "")
        prompt_lines = [
            "Analyze this eBay listing:\n",
            f"Title: {title}",
            f"Price: €{price}",
            f"Shipping: {shipping}",
            f"Condition: {condition}",
            f"Seller rating: {seller_rating}%",
            f"Seller Count: {seller_count}",
            f"Item Location: {item_location}",
            f"Listing Date: {listing_date}",
        ]
        # Bundle price enrichment
        ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
        if not ebay_prices:
            single_price = self._fetch_ebay_price_for_single_listing(deal)
            if single_price is not None:
                prompt_lines.append(f"\nFetched eBay Market Price: €{single_price:.2f}")
        else:
            prompt_lines.append(
                "\nFetched eBay Prices:\n"
                + "\n".join(
                    f"  - {e['game']}: €{e['price_eur']:.2f} ({e['price_source']})"
                    for e in ebay_prices
                    if e.get("price_eur") is not None
                )
            )
        # Image issues
        image_issues_line = self._format_image_issues_line(deal)
        if image_issues_line:
            prompt_lines.append(image_issues_line)
        # Extract individual game prices from description (Kleinanzeigen pattern)
        listed_games = extract_listed_game_prices(description)
        if listed_games:
            prompt_lines.append("\nGames listed in description:")
            for name, pr in listed_games:
                prompt_lines.append(f"  - {name}: €{pr:.2f}")
        # Description (truncated)
        if description:
            prompt_lines.append(f"\nDescription:\n{description[:1500]}")
        text_prompt = "\n".join(prompt_lines)
        parts: list = [self._types.Part.from_text(text=text_prompt)]
        if self._images_supported:
            image_urls: list[str] = deal.get("image_urls", [])
            for url in image_urls[:_MAX_IMAGES]:
                image_part = self._fetch_image_part(url)
                if image_part is not None:
                    parts.append(image_part)
        return parts

    def _build_batch_contents(self, deals: list[dict]) -> list:
        parts: list = []
        intro = (
            f"Below are {len(deals)} eBay listings to analyze. "
            "Return a JSON array of analysis objects, one per listing in order.\n"
        )
        parts.append(self._types.Part.from_text(text=intro))
        for idx, deal in enumerate(deals, 1):
            title = deal.get("title", "Unknown")
            price = deal.get("price", "?")
            condition = deal.get("condition", "?")
            seller_rating = deal.get("seller_rating", "?")
            shipping = deal.get("shipping", "?")
            description = deal.get("description", "")
            seller_count = deal.get("seller_count", "")
            item_location = deal.get("item_location", "")
            listing_date = deal.get("listing_date", "")
            item_text = (
                f"\n--- ITEM {idx} ---\n"
                f"Title: {title}\n"
                f"Price: €{price}\n"
                f"Shipping: {shipping}\n"
                f"Condition: {condition}\n"
                f"Seller rating: {seller_rating}%\n"
                f"Seller Count: {seller_count}\n"
                f"Item Location: {item_location}\n"
                f"Listing Date: {listing_date}\n"
            )
            listed_games = extract_listed_game_prices(description)
            if listed_games:
                item_text += "Games listed in description:\n"
                for name, pr in listed_games:
                    item_text += f"  - {name}: €{pr:.2f}\n"
            item_text += f"Description:\n{description[:800]}\n"
            ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
            if ebay_prices:

                def _fmt(e):
                    price = f"€{e['price_eur']:.2f}" if e.get("price_eur") is not None else "N/A"
                    return f"  - {e['game']}: {price} ({e.get('price_source', '?')})"

                prices_text = "\n".join(_fmt(e) for e in ebay_prices)
                item_text += f"Fetched eBay Prices:\n{prices_text}\n"
            image_issues_line = self._format_image_issues_line(deal)
            if image_issues_line:
                item_text += image_issues_line
            parts.append(self._types.Part.from_text(text=item_text))
            if self._images_supported:
                image_urls: list[str] = deal.get("image_urls", [])
                for url in image_urls[:_MAX_IMAGES]:
                    img_part = self._fetch_image_part(url)
                    if img_part is not None:
                        parts.append(img_part)
        parts.append(
            self._types.Part.from_text(
                text=(
                    f"\nNow return a JSON array of exactly {len(deals)} analysis "
                    "objects, one per item in order, with no other text."
                )
            )
        )
        return parts

    def _assess_batch_with_retry(self, deals: list[dict]) -> list[dict | None]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                contents = self._build_batch_contents(deals)
                t0 = time.monotonic()
                if self._timeout_executor is None:
                    self._timeout_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = self._timeout_executor.submit(
                    self._client.models.generate_content,
                    model=self._model_name,
                    contents=contents,
                    config=self._types.GenerateContentConfig(
                        system_instruction=_BATCH_SYSTEM_PROMPT,
                    ),
                )
                try:
                    response = future.result(timeout=_GEMINI_REQUEST_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    elapsed = time.monotonic() - t0
                    logger.error(
                        "GeminiAssessor: Batch of %d timed out after %.1f s (attempt %d/%d, timeout=%d s).",
                        len(deals),
                        elapsed,
                        attempt + 1,
                        _MAX_RETRIES,
                        _GEMINI_REQUEST_TIMEOUT,
                    )
                    future.cancel()
                    return [{"ai_error_type": "timeout", "ai_assessed": False}] * len(deals)
                elapsed = time.monotonic() - t0
                logger.info(
                    "GeminiAssessor: Batch of %d assessed in %.1f s (attempt %d/%d)",
                    len(deals),
                    elapsed,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                return self._parse_batch_response(response.text, len(deals))
            except Exception as exc:
                exc_msg = str(exc).lower()
                if self._images_supported and (
                    "does not support image" in exc_msg or "image input" in exc_msg or "cannot read" in exc_msg
                ):
                    logger.info("GeminiAssessor: model %r is text-only — disabling images.", self._model_name)
                    self._images_supported = False
                    return self._assess_batch_with_retry(deals)
                if _is_rate_limit_error(exc):
                    delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                    _set_rate_limited_until(time.monotonic() + delay)
                    logger.warning(
                        "GeminiAssessor: 429 RESOURCE_EXHAUSTED (batch of %d) – backing off %.0f s.",
                        len(deals),
                        delay,
                    )
                    return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)
                last_exc = exc
                if _is_transient_error(exc) and attempt < _MAX_RETRIES - 1:
                    retry_delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "GeminiAssessor: Transient error attempt %d/%d (batch of %d) – retrying in %.1f s: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        len(deals),
                        retry_delay,
                        exc,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        "GeminiAssessor: Non-retryable error attempt %d/%d (batch of %d): %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        len(deals),
                        exc,
                    )
                    return [None] * len(deals)
        logger.error("GeminiAssessor: All retries exhausted (batch of %d): %s", len(deals), last_exc)
        return [None] * len(deals)

    def _fetch_image_part(self, url: str):
        if self._types is None:
            return None
        try:
            resp = requests.get(url, timeout=_IMAGE_FETCH_TIMEOUT)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            mime_type = content_type.split(";")[0].strip()
            return self._types.Part.from_bytes(data=resp.content, mime_type=mime_type)
        except Exception as exc:
            logger.debug("GeminiAssessor: Failed to fetch image %r: %s", url, exc)
            return None


# Re-export for backward compatibility (used by gemini_assessor.py shim)
from ai_providers.base import (  # noqa: E402, F811
    _detect_bundle_individual_sale_scam,
    _detect_sports_kinect_deal,
)
