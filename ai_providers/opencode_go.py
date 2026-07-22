"""OpenCode Go AI provider — OpenAI-compatible chat API (Grok, DeepSeek, etc.)."""

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
    _is_rate_limit_error,
    _is_transient_error,
    _parse_retry_delay,
    _set_rate_limited_until,
    logger,
)

_REQUEST_TIMEOUT = 45
_DEFAULT_MODEL = "grok-4.5"
_DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"


class OpenCodeGoAssessor(BaseAssessor):
    """AI assessor using OpenCode Go (Grok 4.5 and other curated models)."""

    provider_id = "opencode-go"
    provider_label = "OpenCode Go"
    supports_images = False

    def __init__(self) -> None:
        # Accept either OPENCODE_GO_API_KEY or OPENCODE_API_KEY.
        api_key = os.environ.get("OPENCODE_GO_API_KEY", "").strip() or os.environ.get("OPENCODE_API_KEY", "").strip()
        default_model = os.environ.get("OPENCODE_GO_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
        # BaseAssessor reads a single env var; seed enabled from resolved key.
        super().__init__("OPENCODE_GO_API_KEY", default_model)
        self._api_key = api_key
        self.enabled = bool(api_key)
        self._base_url = (
            os.environ.get("OPENCODE_GO_BASE_URL", _DEFAULT_BASE_URL).strip().rstrip("/") or _DEFAULT_BASE_URL
        )
        self._session = requests.Session()
        if self.enabled:
            logger.info(
                "OpenCodeGoAssessor: initialised (model=%s, base=%s)",
                self._model_name,
                self._base_url,
            )
        else:
            logger.info("OpenCodeGoAssessor: OPENCODE_GO_API_KEY not set — provider disabled.")

    # ── Single-deal ───────────────────────────────────────────────────────

    def assess_deal(self, deal: dict) -> dict | None:
        if not self.enabled or not self.user_enabled or self.is_rate_limited:
            return None
        early = self._try_deterministic_assessment(deal)
        if early is not None:
            return early
        try:
            text = self._chat(
                system=_SYSTEM_PROMPT,
                user=self._build_deal_text_prompt(deal),
            )
            assessment = self._parse_response(text)
            return self._finalize_assessment(deal, assessment)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                _set_rate_limited_until(time.monotonic() + delay)
                logger.warning("OpenCodeGoAssessor: rate limited – backing off %.0f s.", delay)
            logger.error("OpenCodeGoAssessor: assess_deal failed: %s", exc, exc_info=True)
            return None

    # ── Batch ─────────────────────────────────────────────────────────────

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
                    "OpenCodeGoAssessor: budget exhausted after batch %d; returning %d unassessed deals.",
                    batch_idx,
                    len(deals) - len(results),
                )
                results.extend([None] * (len(deals) - len(results)))
                break
            batch_results = self._assess_batch_with_retry(batch)
            for deal, assessment in zip(batch, batch_results, strict=False):
                if isinstance(assessment, dict) and not assessment.get("ai_error_type"):
                    assessment = self._finalize_assessment(deal, assessment)
                results.append(assessment)
        return results

    def _assess_batch_with_retry(self, deals: list[dict]) -> list[dict | None]:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                user_prompt = self._build_batch_text_prompt(deals)
                t0 = time.monotonic()
                if self._timeout_executor is None:
                    self._timeout_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = self._timeout_executor.submit(
                    self._chat,
                    system=_BATCH_SYSTEM_PROMPT,
                    user=user_prompt,
                )
                try:
                    text = future.result(timeout=_REQUEST_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    elapsed = time.monotonic() - t0
                    logger.error(
                        "OpenCodeGoAssessor: batch of %d timed out after %.1f s (attempt %d/%d).",
                        len(deals),
                        elapsed,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    future.cancel()
                    return [{"ai_error_type": "timeout", "ai_assessed": False}] * len(deals)
                elapsed = time.monotonic() - t0
                logger.info(
                    "OpenCodeGoAssessor: batch of %d assessed in %.1f s (attempt %d/%d, model=%s)",
                    len(deals),
                    elapsed,
                    attempt + 1,
                    _MAX_RETRIES,
                    self._model_name,
                )
                return self._parse_batch_response(text, len(deals))
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                    _set_rate_limited_until(time.monotonic() + delay)
                    logger.warning(
                        "OpenCodeGoAssessor: rate limited (batch of %d) – backing off %.0f s.",
                        len(deals),
                        delay,
                    )
                    return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)
                last_exc = exc
                if _is_transient_error(exc) and attempt < _MAX_RETRIES - 1:
                    retry_delay = _RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "OpenCodeGoAssessor: transient error attempt %d/%d – retry in %.1f s: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        retry_delay,
                        exc,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        "OpenCodeGoAssessor: non-retryable error (batch of %d): %s",
                        len(deals),
                        exc,
                    )
                    return [None] * len(deals)
        logger.error(
            "OpenCodeGoAssessor: all retries exhausted (batch of %d): %s",
            len(deals),
            last_exc,
        )
        return [None] * len(deals)

    # ── HTTP ──────────────────────────────────────────────────────────────

    def _chat(self, *, system: str, user: str) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        resp = self._session.post(url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 429:
            raise RuntimeError(f"429 rate limit: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices in OpenCode Go response: {data!r}"[:400])
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("OpenCode Go response missing message content")
        if isinstance(content, list):
            # Some APIs return content parts; join text segments.
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("text"):
                    parts.append(str(part["text"]))
            content = "\n".join(parts)
        return str(content)
