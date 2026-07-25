"""Generic OpenAI-compatible provider — OpenRouter, GitHub Models, local, etc."""

from __future__ import annotations

import os
import time

import requests

from ai_providers.base import (
    _DEFAULT_BACKOFF_SECONDS,
    _SYSTEM_PROMPT,
    BaseAssessor,
    _is_rate_limit_error,
    _parse_retry_delay,
    _set_rate_limited_until,
    logger,
)

_REQUEST_TIMEOUT = 120
_TOTAL_BUDGET_S = 300
_DEFAULT_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "openai/gpt-4o"


class OpenAICompatAssessor(BaseAssessor):
    """Generic AI assessor for any OpenAI-compatible chat API (OpenRouter, etc.)."""

    def __init__(self) -> None:
        api_key = os.environ.get("AI_BACKEND_KEY", "").strip()
        model = os.environ.get("AI_BACKEND_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
        base = os.environ.get("AI_BACKEND_URL", _DEFAULT_BASE).strip().rstrip("/") or _DEFAULT_BASE

        super().__init__("AI_BACKEND_KEY", model)
        self._api_key = api_key
        self._base_url = base
        self.enabled = bool(api_key)
        self._session = requests.Session()
        if self.enabled:
            logger.info("OpenAICompatAssessor: initialised (model=%s, base=%s)", self._model_name, self._base_url)
        else:
            logger.info("OpenAICompatAssessor: AI_BACKEND_KEY not set — provider disabled.")

    def assess_deal(self, deal: dict) -> dict | None:
        if not self.enabled or not self.user_enabled or self.is_rate_limited:
            return None
        early = self._try_deterministic_assessment(deal)
        if early is not None:
            return early
        try:
            text = self._chat(system=_SYSTEM_PROMPT, user=self._build_deal_text_prompt(deal, description_limit=400))
            assessment = self._parse_response(text)
            return self._finalize_assessment(deal, assessment)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                _set_rate_limited_until(time.monotonic() + delay)
                logger.warning("OpenAICompatAssessor: rate limited – backing off %.0f s.", delay)
                return {"ai_error_type": "rate_limit", "ai_assessed": False}
            logger.error("OpenAICompatAssessor: assess_deal failed: %s", exc, exc_info=True)
            return None

    def assess_deals_batch(self, deals: list[dict]) -> list[dict | None]:
        if not self.enabled or not self.user_enabled or not deals or self.is_rate_limited:
            return [None] * len(deals) if deals else []
        self._prefetch_ebay_prices_parallel(deals)
        results: list[dict | None] = []
        t_start = time.monotonic()
        for idx, deal in enumerate(deals):
            elapsed = time.monotonic() - t_start
            if elapsed >= _TOTAL_BUDGET_S:
                logger.warning("OpenAICompatAssessor: budget exhausted after %d/%d deals.", idx, len(deals))
                results.extend([None] * (len(deals) - len(results)))
                break
            # Small delay to avoid hitting free-tier rate limits (1 req/s).
            if idx > 0:
                time.sleep(0.8)
            results.append(self.assess_deal(deal))
        return results

    def _chat(self, *, system: str, user: str) -> str:
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ebay-scrapper.local",
            "X-Title": "eBay Deal Finder",
        }
        resp = self._session.post(url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
        if resp.status_code == 429:
            raise RuntimeError(f"429 rate limit: {resp.text[:300]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices in response: {data!r}"[:400])
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("Response missing message content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("text"):
                    parts.append(str(part["text"]))
            content = "\n".join(parts)
        return str(content)
