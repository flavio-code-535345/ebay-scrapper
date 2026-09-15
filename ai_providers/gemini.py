"""Gemini AI provider — uses google.genai Client to call Gemini models.

Assessment runs in three phases (see ai_providers/enrichment.py's module
docstring for the full rationale):

  Phase A (ai_providers.enrichment.Enricher) — all network I/O other than
    the Gemini call itself: eBay price lookups and image fetches, run
    concurrently for the WHOLE filtered deal set at once, bounded by the
    caller's deadline.
  Phase B (this module's _build_contents/_build_batch_contents) — pure
    formatting over already-enriched data. Zero network calls.
  Phase C (this module's _assess_batch_with_retry) — the actual Gemini API
    call, wrapped in a real, deadline-aware timeout.

Only Phase C used to be deadline-bounded; Phase B used to make its own live,
unbounded eBay calls (in this project's earlier prompt-building code) which
could alone exceed a whole search's deadline before Gemini was ever called.
Splitting enrichment out into its own phase is what makes Phase C's existing
timeout actually sufficient.
"""

from __future__ import annotations

import concurrent.futures
import os
import time

from ai_providers.base import (
    _ASSESS_TOTAL_BUDGET_S,
    _BATCH_SIZE,
    _BATCH_SYSTEM_PROMPT,
    _DEFAULT_BACKOFF_SECONDS,
    _MAX_RETRIES,
    _RETRY_BASE_DELAY,
    _SYSTEM_PROMPT,
    BaseAssessor,
    _is_rate_limit_error,
    _is_transient_error,
    _parse_response,
    _parse_retry_delay,
    _set_rate_limited_until,
    extract_listed_game_prices,
    logger,
)
from ai_providers.enrichment import EnrichedDeal

_GEMINI_REQUEST_TIMEOUT = 35
_MODEL_NAME = "gemini-3.5-flash-lite"
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
        pre = self._try_deterministic_assessment(deal)
        if pre is not None:
            return pre
        deadline = time.monotonic() + _ASSESS_TOTAL_BUDGET_S
        try:
            enriched = self._enricher.enrich_deals([deal], deadline, fetch_images=self._images_supported)[0]
            contents = self._build_contents(deal, enriched)
            call_timeout = max(0.0, min(_GEMINI_REQUEST_TIMEOUT, deadline - time.monotonic()))
            if self._timeout_executor is None:
                # Sized to _BATCH_MAX_CONCURRENCY so a single-worker executor
                # never silently serializes this against the batch path's
                # own concurrent generate_content calls (see
                # _assess_batch_with_retry's identical comment).
                self._timeout_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_BATCH_MAX_CONCURRENCY)
            future = self._timeout_executor.submit(
                self._client.models.generate_content,
                model=self._model_name,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            try:
                response = future.result(timeout=call_timeout)
            except concurrent.futures.TimeoutError:
                future.cancel()
                logger.error("GeminiAssessor: assess_deal timed out after %.0f s.", call_timeout)
                return None
            assessment = _parse_response(response.text)
            assessment = self._finalize_assessment(deal, assessment)
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

        # Phase A: one concurrent, deadline-bounded pass over the WHOLE
        # filtered deal set (not per Gemini sub-batch) — see
        # ai_providers/enrichment.py. Everything from here on (Phase B/C)
        # reads only the result; no more eBay/image network calls happen.
        enriched_list = self._enricher.enrich_deals(deals, deadline, fetch_images=self._images_supported)

        batches = [deals[i : i + _BATCH_SIZE] for i in range(0, len(deals), _BATCH_SIZE)]
        enriched_batches = [enriched_list[i : i + _BATCH_SIZE] for i in range(0, len(enriched_list), _BATCH_SIZE)]
        n_batches = len(batches)
        futures: list[concurrent.futures.Future | None] = [None] * n_batches
        # Batches run concurrently (bounded by _BATCH_MAX_CONCURRENCY) rather
        # than one at a time — see the constant's comment for why. Submission
        # (not completion) is still paced _BATCH_DELAY_SECONDS apart.
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(_BATCH_MAX_CONCURRENCY, n_batches)) as pool:
            for batch_idx, (batch, enriched_batch) in enumerate(zip(batches, enriched_batches, strict=True)):
                if time.monotonic() >= deadline:
                    logger.warning(
                        "GeminiAssessor: deadline exhausted before submitting batch %d/%d; "
                        "remaining deals returned as None.",
                        batch_idx + 1,
                        n_batches,
                    )
                    break
                futures[batch_idx] = pool.submit(self._assess_batch_with_retry, batch, enriched_batch, deadline)
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
                        assessment = self._finalize_assessment(deal, assessment)
                    results.append(assessment)
        return results

    # ── Prompt construction (Phase B — pure formatting, zero I/O) ──────────

    def _format_deal_text(self, deal: dict, enriched: EnrichedDeal, *, header: str, description_limit: int) -> str:
        """Format one deal's prompt text from already-enriched data.

        Shared by both ``_build_contents`` (single-deal) and
        ``_build_batch_contents`` (per-item, batch) — the only difference
        between the two call sites is the header line and how much
        description text is kept.
        """
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
            header,
            f"Title: {title}",
            f"Price: €{price}",
            f"Shipping: {shipping}",
            f"Condition: {condition}",
            f"Seller rating: {seller_rating}%",
            f"Seller Count: {seller_count}",
            f"Item Location: {item_location}",
            f"Listing Date: {listing_date}",
        ]
        priced = [e for e in enriched.bundle_prices if e.get("price_eur") is not None]
        if priced:
            lines.append(
                "\nFetched eBay Prices:\n"
                + "\n".join(f"  - {e['game']}: €{e['price_eur']:.2f} ({e['price_source']})" for e in priced)
            )
        elif enriched.single_price is not None:
            lines.append(f"\nFetched eBay Market Price: €{enriched.single_price:.2f}")
        image_issues_line = self._format_image_issues_line(deal)
        if image_issues_line:
            lines.append(image_issues_line)
        listed_games = extract_listed_game_prices(description)
        if listed_games:
            lines.append(
                "\nGames listed in description:\n" + "\n".join(f"  - {name}: €{pr:.2f}" for name, pr in listed_games)
            )
        if description:
            lines.append(f"\nDescription:\n{description[:description_limit]}")
        return "\n".join(lines)

    def _build_contents(self, deal: dict, enriched: EnrichedDeal) -> list:
        text_prompt = self._format_deal_text(
            deal, enriched, header="Analyze this eBay listing:", description_limit=1500
        )
        parts: list = [self._types.Part.from_text(text=text_prompt)]
        if self._images_supported:
            for data, mime_type in enriched.images:
                parts.append(self._types.Part.from_bytes(data=data, mime_type=mime_type))
        return parts

    def _build_batch_contents(self, deals: list[dict], enriched_list: list[EnrichedDeal]) -> list:
        parts: list = []
        intro = (
            f"Below are {len(deals)} eBay listings to analyze. "
            "Return a JSON array of analysis objects, one per listing in order.\n"
        )
        parts.append(self._types.Part.from_text(text=intro))
        for idx, (deal, enriched) in enumerate(zip(deals, enriched_list, strict=True), 1):
            item_text = self._format_deal_text(deal, enriched, header=f"--- ITEM {idx} ---", description_limit=800)
            parts.append(self._types.Part.from_text(text=item_text))
            if self._images_supported:
                for data, mime_type in enriched.images:
                    parts.append(self._types.Part.from_bytes(data=data, mime_type=mime_type))
        parts.append(
            self._types.Part.from_text(
                text=(
                    f"\nNow return a JSON array of exactly {len(deals)} analysis "
                    "objects, one per item in order, with no other text."
                )
            )
        )
        return parts

    # ── Phase C — the actual, deadline-timed Gemini API call ───────────────

    def _assess_batch_with_retry(
        self, deals: list[dict], enriched_batch: list[EnrichedDeal], deadline: float | None = None
    ) -> list[dict | None]:
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
                contents = self._build_batch_contents(deals, enriched_batch)
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
                    return self._assess_batch_with_retry(deals, enriched_batch, deadline)
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
