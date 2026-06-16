"""Claude Haiku AI provider — uses Anthropic's Claude 3.5 Haiku model (free tier friendly).

Architecture:
- ClaudeClient: HTTP client with connection pooling for Anthropic API
- ClaudeAssessor: Uses client for API calls, inherits prompt building from BaseAssessor
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    _parse_batch_response,
    _parse_response,
    _rate_limit_lock,
    logger,
)

# ── Configuration ───────────────────────────────────────────────────────────

_MODEL_NAME = "claude-3-5-haiku-20241022"
_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
_ANTHROPIC_API_VERSION = "2023-06-01"
_ANTHROPIC_REQUEST_TIMEOUT = 60
_ASSESS_TOTAL_BUDGET_S = 145

# Claude does not support image inputs in the free tier.
_MAX_IMAGES = 0

# HTTP configuration
_CONNECTION_POOL_SIZE = 10
_MAX_RETRIES_HTTP = 3
_BACKOFF_FACTOR = 0.5
_RETRY_STATUS_CODES = (408, 429, 500, 502, 503, 504)

# Response parsing
_MAX_RESPONSE_SIZE = 32 * 1024  # 32 KB limit for safety


# ── Custom Exceptions ───────────────────────────────────────────────────────


class ClaudeError(Exception):
    """Base exception for Claude API errors."""

    def __init__(self, message: str, status_code: int | None = None, details: str | None = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(f"{message}" + (f" (status={status_code})" if status_code else ""))


class ClaudeConfigError(ClaudeError):
    """Configuration error (e.g., missing API key)."""

    pass


class ClaudeRateLimitError(ClaudeError):
    """Rate limit error (HTTP 429)."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class ClaudeTransientError(ClaudeError):
    """Transient error (e.g., connection timeout, 5xx errors)."""

    pass


class ClaudeAPIError(ClaudeError):
    """API error (invalid response format, content parsing error)."""

    pass


# ── Claude HTTP Client ──────────────────────────────────────────────────────


class ClaudeClient:
    """HTTP client for Anthropic Claude API with connection pooling and error handling."""

    def __init__(
        self,
        api_key: str,
        base_url: str = _ANTHROPIC_BASE_URL,
        timeout: int = _ANTHROPIC_REQUEST_TIMEOUT,
        model: str = _MODEL_NAME,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key
            base_url: API base URL (default: production)
            timeout: Request timeout in seconds
            model: Model name to use (default: claude-3-5-haiku-20241022)

        Raises:
            ClaudeConfigError: If API key is empty
        """
        if not api_key or not api_key.strip():
            raise ClaudeConfigError("Anthropic API key is required")

        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._model = model
        self._session = self._create_session()
        self._call_count = 0
        self._error_count = 0

        logger.debug(
            "ClaudeClient: Initialized (model=%s, base_url=%s, timeout=%d)",
            self._model,
            self._base_url,
            self._timeout,
        )

    def _create_session(self) -> requests.Session:
        """Create requests session with connection pooling and retry strategy."""
        session = requests.Session()

        # Retry strategy for transient errors
        retry_strategy = Retry(
            total=_MAX_RETRIES_HTTP,
            status_forcelist=_RETRY_STATUS_CODES,
            backoff_factor=_BACKOFF_FACTOR,
            allowed_methods=["POST", "GET"],
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=_CONNECTION_POOL_SIZE,
            pool_maxsize=_CONNECTION_POOL_SIZE,
        )

        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _build_headers(self) -> dict[str, str]:
        """Build request headers for Anthropic API."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
            "User-Agent": "ClaudeAssessor/1.0",
        }

    def _validate_response(self, data: Any) -> str:
        """Validate and extract response content.

        Args:
            data: Parsed JSON response from API

        Returns:
            Response content text

        Raises:
            ClaudeAPIError: If response is malformed
        """
        if not isinstance(data, dict):
            raise ClaudeAPIError(f"Expected dict response, got {type(data).__name__}")

        # Check for error in response
        if "error" in data:
            error_info = data["error"]
            if isinstance(error_info, dict):
                error_msg = error_info.get("message", "Unknown error")
                error_type = error_info.get("type", "unknown")
                raise ClaudeAPIError(
                    f"API error: {error_type} - {error_msg}",
                    details=json.dumps(error_info),
                )
            raise ClaudeAPIError(f"API error: {error_info}")

        # Extract content from response
        if "content" not in data:
            raise ClaudeAPIError("No 'content' in response")

        content_array = data.get("content", [])
        if not content_array:
            raise ClaudeAPIError("Empty 'content' array in response")

        # Find text content (Claude returns array of content blocks)
        for block in content_array:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    # Safety check: ensure content isn't too large
                    if len(text) > _MAX_RESPONSE_SIZE:
                        logger.warning("ClaudeClient: Response content exceeds %d bytes (%d)", _MAX_RESPONSE_SIZE, len(text))
                    return text

        raise ClaudeAPIError("No text content found in response")

    def _handle_http_error(self, exc: requests.RequestException, endpoint: str) -> None:
        """Categorize and raise appropriate exception for HTTP errors.

        Args:
            exc: The requests exception
            endpoint: The API endpoint being called

        Raises:
            ClaudeRateLimitError: For 429 responses
            ClaudeTransientError: For connection/timeout errors
            ClaudeAPIError: For other errors
        """
        self._error_count += 1

        # Handle response errors (status codes)
        if isinstance(exc, requests.HTTPError):
            if hasattr(exc.response, "status_code"):
                status_code = exc.response.status_code

                # Rate limit
                if status_code == 429:
                    retry_after = None
                    if hasattr(exc.response, "headers"):
                        retry_after_header = exc.response.headers.get("Retry-After")
                        if retry_after_header:
                            try:
                                retry_after = int(retry_after_header)
                            except ValueError:
                                pass

                    raise ClaudeRateLimitError(
                        f"Rate limited by API (endpoint={endpoint})",
                        retry_after=retry_after,
                    )

                # Auth errors
                if status_code in (401, 403):
                    raise ClaudeConfigError(f"Authentication failed (status={status_code})")

                # Server errors
                if status_code >= 500:
                    raise ClaudeTransientError(
                        f"Server error (endpoint={endpoint}, status={status_code})",
                        status_code=status_code,
                    )

                # Other 4xx errors
                if status_code >= 400:
                    raise ClaudeAPIError(
                        f"API request failed (endpoint={endpoint}, status={status_code})",
                        status_code=status_code,
                    )

            raise ClaudeAPIError(f"HTTP error: {str(exc)}")

        # Handle connection errors (timeout, connection refused, etc.)
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            raise ClaudeTransientError(f"Connection error (endpoint={endpoint}): {str(exc)}")

        # Handle request exceptions
        if isinstance(exc, requests.RequestException):
            raise ClaudeTransientError(f"Request error (endpoint={endpoint}): {str(exc)}")

        # Fallback
        raise ClaudeAPIError(f"Unexpected error: {str(exc)}")

    def _handle_json_error(self, exc: ValueError, response_text: str) -> None:
        """Handle JSON parsing errors.

        Args:
            exc: The JSON error
            response_text: The response text that failed to parse

        Raises:
            ClaudeAPIError: Always raises with details
        """
        preview = response_text[:200] if response_text else "(empty)"
        raise ClaudeAPIError(
            f"Failed to parse JSON response: {str(exc)}",
            details=f"Response preview: {preview}",
        )

    def chat_completion(
        self,
        user_content: str,
        system_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Call Anthropic Claude API.

        Args:
            user_content: User message content
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens in response

        Returns:
            Response content from the model

        Raises:
            ClaudeRateLimitError: If rate limited
            ClaudeTransientError: If transient error occurs
            ClaudeConfigError: If configuration is invalid
            ClaudeAPIError: If API returns error or response is malformed
        """
        endpoint = "/messages"
        url = f"{self._base_url}/v1{endpoint}"
        self._call_count += 1

        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_content},
            ],
        }

        logger.debug(
            "ClaudeClient: POST %s (call=%d, tokens=%d)",
            endpoint,
            self._call_count,
            max_tokens,
        )

        try:
            t0 = time.monotonic()
            resp = self._session.post(
                url,
                headers=self._build_headers(),
                json=payload,
                timeout=self._timeout,
            )
            elapsed = time.monotonic() - t0
            resp.raise_for_status()

            # Parse response
            try:
                data = resp.json()
            except ValueError as exc:
                self._handle_json_error(exc, resp.text)

            # Validate and extract content
            content = self._validate_response(data)

            logger.debug(
                "ClaudeClient: Success (call=%d, elapsed=%.2f s, content_len=%d)",
                self._call_count,
                elapsed,
                len(content),
            )

            return content

        except requests.RequestException as exc:
            self._handle_http_error(exc, endpoint)

    def close(self) -> None:
        """Close the session."""
        if self._session:
            self._session.close()
            logger.debug("ClaudeClient: Closed (calls=%d, errors=%d)", self._call_count, self._error_count)

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass


# ── Claude Assessor ────────────────────────────────────────────────────────


class ClaudeAssessor(BaseAssessor):
    """AI assessor using Anthropic's Claude 3.5 Haiku model (free tier friendly)."""

    def __init__(self) -> None:
        super().__init__("ANTHROPIC_API_KEY", _MODEL_NAME)
        self._client: ClaudeClient | None = None

        if self.enabled:
            try:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
                self._client = ClaudeClient(api_key=api_key, model=self._model_name)
                logger.info("ClaudeAssessor: Initialized with Claude Haiku API client (model=%s)", self._model_name)
            except ClaudeConfigError as exc:
                logger.error("ClaudeAssessor: Configuration error: %s", exc)
                self.enabled = False
            except Exception as exc:
                logger.error("ClaudeAssessor: Unexpected error during init: %s", exc, exc_info=True)
                self.enabled = False
        else:
            logger.info(
                "ClaudeAssessor: ANTHROPIC_API_KEY not set — AI assessment disabled; "
                "falling back to rules engine."
            )

    def assess_deal(self, deal: dict) -> dict | None:
        """Assess a single eBay deal using Claude."""
        if not self.enabled or not self.user_enabled or self.is_rate_limited:
            return None

        # Check deterministic rules first
        sr = _detect_sports_kinect_deal(deal)
        if sr:
            return self._build_deterministic_result("Avoid", 100, sr, is_scam=False)

        scam = _detect_bundle_individual_sale_scam(deal)
        if scam:
            return self._build_deterministic_result("Avoid", 100, scam, is_scam=True)

        # Use AI assessment
        try:
            prompt = self._build_contents(deal)
            response = self._call_api(prompt, _SYSTEM_PROMPT)
            if response is None:
                return None

            assessment = _parse_response(response)
            assessment = _apply_sports_kinect_override(deal, assessment)
            assessment = _apply_scam_override(deal, assessment)
            return assessment

        except Exception as exc:
            self._handle_api_error(exc)
            return None

    def assess_deals_batch(self, deals: list[dict]) -> list[dict | None]:
        """Assess a batch of deals."""
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
                    "ClaudeAssessor: Total time budget exhausted after batch %d",
                    batch_idx,
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

    def _call_api(self, user_content: str, system_prompt: str) -> str | None:
        """Call the Claude API with error handling.

        Returns:
            Response text or None if error occurred
        """
        if self._client is None:
            logger.error("ClaudeAssessor: Client not initialized")
            return None

        try:
            return self._client.chat_completion(user_content, system_prompt)
        except ClaudeRateLimitError as exc:
            retry_after = exc.retry_after or _DEFAULT_BACKOFF_SECONDS
            with _rate_limit_lock:
                import ai_providers.base

                ai_providers.base._rate_limited_until = time.monotonic() + retry_after
            logger.warning("ClaudeAssessor: Rate limited — backing off %.0f s", retry_after)
            raise
        except ClaudeTransientError as exc:
            logger.warning("ClaudeAssessor: Transient error: %s", exc)
            raise
        except ClaudeConfigError as exc:
            logger.error("ClaudeAssessor: Configuration error: %s", exc)
            self.enabled = False
            raise
        except ClaudeAPIError as exc:
            logger.error("ClaudeAssessor: API error: %s", exc)
            raise

    def _handle_api_error(self, exc: Exception) -> None:
        """Handle API errors and update rate limiting state."""
        if isinstance(exc, ClaudeRateLimitError):
            delay = exc.retry_after or _DEFAULT_BACKOFF_SECONDS
            with _rate_limit_lock:
                import ai_providers.base

                ai_providers.base._rate_limited_until = time.monotonic() + delay
            logger.warning("ClaudeAssessor: Rate limit — backing off %.0f s", delay)
        elif isinstance(exc, ClaudeTransientError):
            logger.warning("ClaudeAssessor: Transient error (will retry): %s", exc)
        elif isinstance(exc, ClaudeConfigError):
            logger.error("ClaudeAssessor: Configuration error (disabling): %s", exc)
            self.enabled = False
        elif isinstance(exc, ClaudeAPIError):
            logger.error("ClaudeAssessor: API error: %s", exc)
        else:
            logger.error("ClaudeAssessor: Unexpected error: %s", exc, exc_info=True)

    def _assess_batch_with_retry(self, deals: list[dict]) -> list[dict | None]:
        """Assess a batch of deals with retry logic."""
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                prompt = self._build_batch_contents(deals)
                response = self._call_api(prompt, _BATCH_SYSTEM_PROMPT)
                if response is None:
                    return [None] * len(deals)

                return _parse_batch_response(response, len(deals))

            except ClaudeRateLimitError:
                logger.warning(
                    "ClaudeAssessor: Rate limited (batch of %d) — not retrying",
                    len(deals),
                )
                return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)

            except ClaudeTransientError as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    retry_delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "ClaudeAssessor: Transient error (batch of %d) — "
                        "retrying in %.1f s (attempt %d/%d)",
                        len(deals),
                        retry_delay,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        "ClaudeAssessor: Transient error (batch of %d) — "
                        "max retries exhausted (attempt %d/%d)",
                        len(deals),
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    return [None] * len(deals)

            except (ClaudeConfigError, ClaudeAPIError) as exc:
                logger.error(
                    "ClaudeAssessor: Non-retryable error (batch of %d): %s",
                    len(deals),
                    exc,
                )
                return [None] * len(deals)

            except Exception as exc:
                logger.error(
                    "ClaudeAssessor: Unexpected error (batch of %d): %s",
                    len(deals),
                    exc,
                    exc_info=True,
                )
                return [None] * len(deals)

        logger.error("ClaudeAssessor: All retries exhausted (batch of %d): %s", len(deals), last_exc)
        return [None] * len(deals)

    def _build_contents(self, deal: dict) -> str:
        """Build text prompt for single deal (inherited prompt building logic)."""
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

        # Bundle price enrichment
        ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
        if not ebay_prices:
            single_price = self._fetch_ebay_price_for_single_listing(deal)
            if single_price is not None:
                lines.append(f"\nFetched eBay Market Price: €{single_price:.2f}")
        else:

            def _fmt(e):
                price = f"€{e['price_eur']:.2f}" if e.get("price_eur") is not None else "N/A"
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
        """Build text prompt for batch of deals."""
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
                    price = f"€{e['price_eur']:.2f}" if e.get("price_eur") is not None else "N/A"
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

    def _build_deterministic_result(self, rating: str, confidence: int, summary: str, is_scam: bool) -> dict:
        """Build deterministic assessment result (no AI call needed)."""
        return {
            "ai_deal_rating": rating,
            "ai_confidence_score": confidence,
            "ai_visual_findings": [],
            "ai_red_flags": [] if not is_scam else [summary],
            "ai_fair_market_estimate": "",
            "ai_itemized_resale_estimates": [],
            "ai_estimated_total_cost": 0,
            "ai_estimated_gross_profit": 0,
            "ai_verdict_summary": summary,
            "ai_assessed": True,
            "ai_potential_scam": is_scam,
            "ai_scam_warning": summary if is_scam else "",
        }

    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
