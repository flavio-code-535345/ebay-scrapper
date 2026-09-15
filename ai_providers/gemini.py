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
_MODEL_NAME = "gemini-3.5-flash-lite"
_MAX_IMAGES = 3
_IMAGE_FETCH_TIMEOUT = 5
# A batch of 5 deals × up to 3 images each is up to 15 image downloads.
# Fetched sequentially (the original design) that's up to 15 × 5s = 75s in
# the worst case — entirely unbounded by the batch's own deadline/timeout,
# since it all happens before the API call (and its timeout) even starts.
# That's large enough to plausibly eat most of a search's whole deadline on
# just the FIRST batch, before it ever reaches Gemini — worth bounding with
# real concurrency rather than raising a timeout number.
_IMAGE_FETCH_MAX_WORKERS = 8
_BATCH_DELAY_SECONDS = 4.5  # respect 15 RPM free-tier limit
# How many batch calls may be in flight at once. Submissions are still paced
# _BATCH_DELAY_SECONDS apart (bounding the *request-start* rate to the same
# ~13/min the old fully-serial design respected), but letting calls overlap
# means total wall time is roughly (submission pacing) + (one call's own
# duration) instead of the sum of every call's duration plus every stagger —
# the difference between all _MAX_DISPLAY deals fitting in a search's overall
# deadline versus only the first batch or two.
_BATCH_MAX_CONCURRENCY = 3

# Known text-only Gemini models — auto-disable image input to avoid SDK errors.
_TEXT_ONLY_MODELS: frozenset[str] = frozenset()


def _is_text_only_model(model: str) -> bool:
    """Return True for known image-unsupported Gemini model names.

    Every currently-shipping Gemini model family (2.5 and 3.x, "lite"
    variants included) accepts native multimodal image input — there is no
    general "lite = text-only" rule. Only ``_TEXT_ONLY_MODELS`` entries are
    treated as text-only up front; anything else is assumed multimodal and
    the runtime fallback in :class:`GeminiAssessor` (which disables images
    after an actual "does not support image" error from the API) covers any
    future model that genuinely lacks vision support.
    """
    return model.lower().strip() in _TEXT_ONLY_MODELS


class GeminiAssessor(BaseAssessor):
    """AI assessor using Google Gemini models."""

    def __init__(self) -> None:
        super().__init__("GEMINI_API_KEY", _MODEL_NAME)
        self._images_supported = not _is_text_only_model(self._model_name)
        self._result_cache: dict[str, dict] = {}
        if not self._images_supported:
            logger.info("GeminiAssessor: model %r is text-only — images disabled.", self._model_name)
        self._init_client()

    @BaseAssessor.model_name.setter
    def model_name(self, value: str) -> None:
        BaseAssessor.model_name.fset(self, value)  # type: ignore[attr-defined]
        if _is_text_only_model(self._model_name):
            if self._images_supported:
                logger.info("GeminiAssessor: model %r is text-only — disabling images.", self._model_name)
                self._images_supported = False
        else:
            self._images_supported = True
        self._init_client()

    def _init_client(self) -> None:
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
        url = deal.get("url", "")
        if url and url in self._result_cache:
            return dict(self._result_cache[url])
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
            if url:
                self._result_cache[url] = dict(assessment)
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

    def assess_deals_batch(self, deals: list[dict], deadline: float | None = None) -> list[dict | None]:
        if not self.enabled or not self.user_enabled or not deals or self.is_rate_limited:
            return [None] * len(deals) if deals else []
        if deadline is None:
            # No caller-supplied deadline (e.g. a direct/standalone call outside
            # the Flask request cycle) — fall back to a fixed standalone budget.
            deadline = time.monotonic() + _ASSESS_TOTAL_BUDGET_S
        # Counting the eBay price prefetch against the deadline too (it used
        # to run before the clock even started) — a slow prefetch previously
        # got a free pass on top of the assessment budget.
        self._prefetch_ebay_prices_parallel(deals)

        batches = [deals[i : i + _BATCH_SIZE] for i in range(0, len(deals), _BATCH_SIZE)]
        n_batches = len(batches)
        futures: list[concurrent.futures.Future | None] = [None] * n_batches
        # Batches run concurrently (bounded by _BATCH_MAX_CONCURRENCY) rather
        # than one at a time — see the constant's comment for why. Submission
        # (not completion) is still paced _BATCH_DELAY_SECONDS apart.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(_BATCH_MAX_CONCURRENCY, n_batches)) as pool:
            for batch_idx, batch in enumerate(batches):
                if time.monotonic() >= deadline:
                    logger.warning(
                        "GeminiAssessor: deadline exhausted before submitting batch %d/%d; "
                        "remaining deals returned as None.",
                        batch_idx + 1,
                        n_batches,
                    )
                    break
                futures[batch_idx] = pool.submit(self._assess_batch_with_retry, batch, deadline)
                if batch_idx < n_batches - 1:
                    time.sleep(max(0.0, min(_BATCH_DELAY_SECONDS, deadline - time.monotonic())))

            results: list[dict | None] = []
            for batch, future in zip(batches, futures, strict=True):
                if future is None:
                    results.extend([None] * len(batch))
                    continue
                try:
                    batch_results = future.result()
                except Exception as exc:
                    logger.error("GeminiAssessor: batch raised unexpectedly: %s", exc, exc_info=True)
                    batch_results = [None] * len(batch)
                for deal, assessment in zip(batch, batch_results, strict=False):
                    if isinstance(assessment, dict):
                        assessment = _apply_garbage_overrides(deal, assessment)
                        assessment = _apply_sports_kinect_override(deal, assessment)
                        assessment = _apply_scam_override(deal, assessment)
                    results.append(assessment)
        return results

    # ── Prompt construction (Gemini-specific — uses self._types.Part) ─────

    def _fetch_image_parts_for_deals(self, deals: list[dict]) -> dict[int, list]:
        """Fetch every deal's images concurrently.

        Returns ``{deal_index: [Part, ...]}`` — each deal's own images stay
        in their original order even though the underlying HTTP fetches run
        concurrently and may complete in any order; a deal with no
        successfully-fetched images is simply absent from the result.
        """
        if not self._images_supported:
            return {}
        jobs: list[tuple[int, str]] = [
            (i, url) for i, deal in enumerate(deals) for url in (deal.get("image_urls") or [])[:_MAX_IMAGES]
        ]
        if not jobs:
            return {}
        results: dict[int, list] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(_IMAGE_FETCH_MAX_WORKERS, len(jobs))) as pool:
            # Submit in job order, then collect in that SAME order (not
            # completion order) so each deal's image list stays correctly
            # ordered — the pattern used throughout this module for
            # concurrency without losing a meaningful original ordering.
            futures = [(i, pool.submit(self._fetch_image_part, url)) for i, url in jobs]
            for i, fut in futures:
                part = fut.result()
                if part is not None:
                    results.setdefault(i, []).append(part)
        return results

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
        parts.extend(self._fetch_image_parts_for_deals([deal]).get(0, []))
        return parts

    def _build_batch_contents(self, deals: list[dict]) -> list:
        parts: list = []
        intro = (
            f"Below are {len(deals)} eBay listings to analyze. "
            "Return a JSON array of analysis objects, one per listing in order.\n"
        )
        parts.append(self._types.Part.from_text(text=intro))
        # Fetch every deal's images up front, concurrently, instead of one
        # URL at a time inside the loop below — see _IMAGE_FETCH_MAX_WORKERS'
        # comment for why a sequential fetch here was a real latency bug.
        image_parts_by_deal = self._fetch_image_parts_for_deals(deals)
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
            parts.extend(image_parts_by_deal.get(idx - 1, []))
        parts.append(
            self._types.Part.from_text(
                text=(
                    f"\nNow return a JSON array of exactly {len(deals)} analysis "
                    "objects, one per item in order, with no other text."
                )
            )
        )
        return parts

    def _assess_batch_with_retry(self, deals: list[dict], deadline: float | None = None) -> list[dict | None]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 1:
                    logger.warning(
                        "GeminiAssessor: deadline exhausted before attempt %d/%d (batch of %d) — skipping.",
                        attempt + 1,
                        _MAX_RETRIES,
                        len(deals),
                    )
                    return [None] * len(deals)
                call_timeout = min(_GEMINI_REQUEST_TIMEOUT, remaining)
            else:
                call_timeout = _GEMINI_REQUEST_TIMEOUT
            try:
                contents = self._build_batch_contents(deals)
                t0 = time.monotonic()
                if self._timeout_executor is None:
                    # Sized to _BATCH_MAX_CONCURRENCY: assess_deals_batch now
                    # calls _assess_batch_with_retry from several concurrent
                    # threads at once, each of which submits its own
                    # generate_content call here and blocks on it — a
                    # single-worker executor would silently serialize every
                    # "concurrent" batch back into one at a time.
                    self._timeout_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_BATCH_MAX_CONCURRENCY)
                future = self._timeout_executor.submit(
                    self._client.models.generate_content,
                    model=self._model_name,
                    contents=contents,
                    config=self._types.GenerateContentConfig(
                        system_instruction=_BATCH_SYSTEM_PROMPT,
                    ),
                )
                try:
                    response = future.result(timeout=call_timeout)
                except concurrent.futures.TimeoutError:
                    elapsed = time.monotonic() - t0
                    logger.error(
                        "GeminiAssessor: Batch of %d timed out after %.1f s (attempt %d/%d, timeout=%.0f s).",
                        len(deals),
                        elapsed,
                        attempt + 1,
                        _MAX_RETRIES,
                        call_timeout,
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
                    return self._assess_batch_with_retry(deals, deadline)
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
                    if deadline is not None:
                        retry_delay = max(0.0, min(retry_delay, deadline - time.monotonic()))
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
