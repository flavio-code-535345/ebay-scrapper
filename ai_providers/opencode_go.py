"""OpenAI-compatible chat provider — OpenCode Go + DeepSeek + OpenRouter."""

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
_GO_TOTAL_BUDGET_S = 600
_DEFAULT_MODEL = "grok-4.5"

_OP_BASE_URL = "https://opencode.ai/zen/go/v1"
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_COMPAT_PROVIDERS: dict[str, tuple[str, str, str]] = {
    "DEEPSEEK_API_KEY": (_DEEPSEEK_BASE_URL, "deepseek-chat", "DeepSeek"),
    "OPENROUTER_API_KEY": (_OPENROUTER_BASE_URL, "grok-4.5", "OpenRouter"),
}
_GO_KEY_ENVS = ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY")


def _resolve_credentials_and_base() -> tuple[str, str, str, str]:
    for env_name in _GO_KEY_ENVS:
        key = os.environ.get(env_name, "").strip()
        if key:
            return key, _OP_BASE_URL, "", env_name
    for env_name, (url, def_model, _label) in _COMPAT_PROVIDERS.items():
        key = os.environ.get(env_name, "").strip()
        if not key:
            continue
        explicit = os.environ.get("OPENCODE_GO_BASE_URL", "").strip()
        base = explicit if explicit else url
        return key, base.rstrip("/") or url, def_model, env_name
    return "", _OP_BASE_URL, "", ""


class OpenCodeGoAssessor(BaseAssessor):
    """AI assessor via OpenAI-compatible chat (OpenCode Go, DeepSeek, OpenRouter)."""

    provider_id = "opencode-go"
    provider_label = "OpenCode Go"
    supports_images = False

    def __init__(self) -> None:
        api_key, base_url, auto_model, key_source = _resolve_credentials_and_base()
        model = os.environ.get("OPENCODE_GO_MODEL", "").strip() or auto_model or _DEFAULT_MODEL
        super().__init__("OPENCODE_GO_API_KEY", model)
        self._api_key = api_key
        self.enabled = bool(api_key)
        self._base_url = base_url
        self._session = requests.Session()
        if self.enabled:
            logger.info(
                "OpenCodeGoAssessor: initialised (model=%s, base=%s, key=%s)",
                self._model_name,
                self._base_url,
                key_source,
            )
        else:
            logger.info(
                "OpenCodeGoAssessor: no API key found — provider disabled. Set OPENCODE_GO_API_KEY or DEEPSEEK_API_KEY."
            )

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
                return {"ai_error_type": "rate_limit", "ai_assessed": False}
            logger.error("OpenCodeGoAssessor: assess_deal failed: %s", exc, exc_info=True)
            return None

    # ── Batch (one-at-a-time — OpenCode Go is slow with large prompts) ────

    def assess_deals_batch(self, deals: list[dict]) -> list[dict | None]:
        if not self.enabled or not self.user_enabled or not deals or self.is_rate_limited:
            return [None] * len(deals) if deals else []
        self._prefetch_ebay_prices_parallel(deals)
        results: list[dict | None] = []
        t_start = time.monotonic()
        for idx, deal in enumerate(deals):
            elapsed = time.monotonic() - t_start
            if elapsed >= _GO_TOTAL_BUDGET_S:
                logger.warning(
                    "OpenCodeGoAssessor: budget exhausted after %d/%d deals; returning %d unassessed.",
                    idx,
                    len(deals),
                    len(deals) - len(results),
                )
                results.extend([None] * (len(deals) - len(results)))
                break
            results.append(self.assess_deal(deal))
        return results

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
