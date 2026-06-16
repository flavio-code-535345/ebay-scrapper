"""DeepSeek AI provider — uses OpenAI-compatible chat completions API for DeepSeek models."""

from __future__ import annotations

import os
import time

import requests

from ai_providers.base import (
    _BATCH_SIZE,
    _BATCH_SYSTEM_PROMPT,
    _DEFAULT_BACKOFF_SECONDS,
    _MAX_RETRIES,
    _RETRY_BASE_DELAY,
    _SYSTEM_PROMPT,
    BaseAssessor,
    _apply_scam_override,
    _apply_sports_kinect_override,
    _detect_bundle_individual_sale_scam,
    _detect_sports_kinect_deal,
    _is_rate_limit_error,
    _is_transient_error,
    _parse_batch_response,
    _parse_response,
    _parse_retry_delay,
    _rate_limit_lock,
    logger,
)

_MODEL_NAME = "deepseek-chat"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_REQUEST_TIMEOUT = 60
_ASSESS_TOTAL_BUDGET_S = 145

# DeepSeek does not support image inputs in its chat API.
_MAX_IMAGES = 0


class DeepSeekAssessor(BaseAssessor):
    """AI assessor using DeepSeek models (OpenAI-compatible chat API)."""

    def __init__(self) -> None:
        super().__init__("DEEPSEEK_API_KEY", _MODEL_NAME)
        self._api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if self.enabled:
            logger.info("DeepSeekAssessor: DeepSeek API initialised (model=%s)", self._model_name)
        else:
            logger.info(
                "DeepSeekAssessor: DEEPSEEK_API_KEY not set — AI assessment disabled; "
                "falling back to rules engine."
            )

    # ── Single-deal assessment ────────────────────────────────────────────

    def assess_deal(self, deal: dict) -> dict | None:
        if not self.enabled or not self.user_enabled or self.is_rate_limited:
            return None
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
            prompt = self._build_contents(deal)
            response = self._chat_completion(prompt, _SYSTEM_PROMPT)
            assessment = _parse_response(response)
            assessment = _apply_sports_kinect_override(deal, assessment)
            assessment = _apply_scam_override(deal, assessment)
            return assessment
        except Exception as exc:
            if _is_rate_limit_error(exc):
                delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                with _rate_limit_lock:
                    _rate_limited_until = time.monotonic() + delay
                logger.warning("DeepSeekAssessor: 429 – backing off %.0f s.", delay)
            logger.error("DeepSeekAssessor: assess_deal failed: %s", exc, exc_info=True)
            return None

    # ── Batch assessment ──────────────────────────────────────────────────

    def assess_deals_batch(self, deals: list[dict]) -> list[dict | None]:
        if not self.enabled or not self.user_enabled or not deals or self.is_rate_limited:
            return [None] * len(deals) if deals else []
        self._prefetch_ebay_prices_parallel(deals)
        results: list[dict | None] = []
        t_start = time.monotonic()
        for batch_idx, batch_start in enumerate(range(0, len(deals), _BATCH_SIZE)):
            batch = deals[batch_start:batch_start + _BATCH_SIZE]
            elapsed = time.monotonic() - t_start
            if elapsed >= _ASSESS_TOTAL_BUDGET_S:
                logger.warning(
                    "DeepSeekAssessor: total budget exhausted after batch %d.", batch_idx
                )
                results.extend([None] * (len(deals) - len(results)))
                break
            batch_results = self._assess_batch_with_retry(batch)
            for deal, assessment in zip(batch, batch_results):
                if isinstance(assessment, dict):
                    assessment = _apply_sports_kinect_override(deal, assessment)
                    assessment = _apply_scam_override(deal, assessment)
                results.append(assessment)
        return results

    # ── Prompt construction (text-only — no image parts) ──────────────────

    def _build_contents(self, deal: dict) -> str:
        title = deal.get("title", "Unknown")
        price = deal.get("price", "?")
        condition = deal.get("condition", "?")
        seller_rating = deal.get("seller_rating", "?")
        shipping = deal.get("shipping", "?")
        description = deal.get("description", "")
        seller_count = deal.get("seller_count", "")
        item_location = deal.get("item_location", "")
        listing_date = deal.get("listing_date", "")
        lines = [
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
        ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
        if not ebay_prices:
            single_price = self._fetch_ebay_price_for_single_listing(deal)
            if single_price is not None:
                lines.append(f"\nFetched eBay Market Price: €{single_price:.2f}")
        else:
            def _fmt(e):
                price = f"€{e['price_eur']:.2f}" if e.get('price_eur') is not None else "N/A"
                return f"  - {e['game']}: {price} ({e.get('price_source', '?')})"
            prices_str = "\n".join(_fmt(e) for e in ebay_prices)
            lines.append(f"\nFetched eBay Prices:\n{prices_str}")
        image_issues_line = self._format_image_issues_line(deal)
        if image_issues_line:
            lines.append(image_issues_line)
        if description:
            lines.append(f"\nDescription:\n{description[:1500]}")
        return "\n".join(lines)

    def _build_batch_contents(self, deals: list[dict]) -> str:
        lines = [
            f"Below are {len(deals)} eBay listings to analyze. "
            "Return a JSON array of analysis objects, one per listing in order.\n"
        ]
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
                f"Description:\n{description[:800]}\n"
            )
            ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
            if ebay_prices:
                def _fmt(e):
                    price = f"€{e['price_eur']:.2f}" if e.get('price_eur') is not None else "N/A"
                    return f"  - {e['game']}: {price} ({e.get('price_source', '?')})"
                prices_text = "\n".join(_fmt(e) for e in ebay_prices)
                item_text += f"Fetched eBay Prices:\n{prices_text}\n"
            image_issues_line = self._format_image_issues_line(deal)
            if image_issues_line:
                item_text += image_issues_line
            lines.append(item_text)
        lines.append(
            f"\nNow return a JSON array of exactly {len(deals)} analysis "
            "objects, one per item in order, with no other text."
        )
        return "\n".join(lines)

    def _assess_batch_with_retry(self, deals: list[dict]) -> list[dict | None]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                prompt = self._build_batch_contents(deals)
                response = self._chat_completion(prompt, _BATCH_SYSTEM_PROMPT)
                return _parse_batch_response(response, len(deals))
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                    with _rate_limit_lock:
                        _rate_limited_until = time.monotonic() + delay
                    logger.warning(
                        "DeepSeekAssessor: 429 (batch of %d) – backing off %.0f s.",
                        len(deals), delay,
                    )
                    return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)
                last_exc = exc
                if _is_transient_error(exc) and attempt < _MAX_RETRIES - 1:
                    retry_delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "DeepSeekAssessor: Transient error attempt %d/%d (batch of %d) – retrying in %.1f s: %s",
                        attempt + 1, _MAX_RETRIES, len(deals), retry_delay, exc,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        "DeepSeekAssessor: Non-retryable error attempt %d/%d (batch of %d): %s",
                        attempt + 1, _MAX_RETRIES, len(deals), exc,
                    )
                    return [None] * len(deals)
        logger.error("DeepSeekAssessor: All retries exhausted (batch of %d): %s", len(deals), last_exc)
        return [None] * len(deals)

    # ── Chat completion helper ────────────────────────────────────────────

    def _chat_completion(self, user_content: str, system_prompt: str) -> str:
        """Call the DeepSeek chat completions API and return the response text."""
        resp = requests.post(
            f"{_DEEPSEEK_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "max_tokens": 4096,
            },
            timeout=_DEEPSEEK_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
