"""AI-assessment enrichment — Phase A of the three-phase assessment pipeline.

Phase A (this module): concurrent, deadline-bounded network I/O — every
eBay price lookup and every image fetch for a WHOLE filtered deal set at
once (not per Gemini sub-batch). Whatever isn't done when the deadline hits
is simply absent from the result; already-cached prices still populate,
partial image lists still get used — this never blocks past its budget.

Phase B (``ai_providers/gemini.py``'s content-builders): pure formatting
over an :class:`EnrichedDeal` — zero network calls of its own. That split is
what makes Phase C (the already-deadline-checked Gemini API call) actually
sufficient: previously, ``_fetch_ebay_prices_for_bundle``-style helpers ran
*inside* content-building with no deadline awareness at all, so a slow or
uncached price lookup for a bundle listing (up to 8 sequential live calls,
one per extracted game title) could alone burn far more time than a whole
search's deadline, for a single batch, before Gemini was ever called.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from ai_providers.base import (
    _BUNDLE_TITLE_KEYWORDS_RE,
    _IMAGE_FETCH_TIMEOUT,
    _MAX_IMAGES,
    _build_single_game_search_query,
    _extract_platform_name,
    _extract_potential_game_titles,
)

logger = logging.getLogger(__name__)

_EBAY_CACHE_TTL = 300.0
_ENRICH_MAX_WORKERS = 12
# Hard ceiling on Phase A's own wait, regardless of how much deadline remains
# — see the budget calculation in Enricher.enrich_deals for why this is
# additionally capped to a fraction of the remaining deadline, not just this
# fixed value.
_ENRICH_MAX_BUDGET_S = 15.0


@dataclass
class EnrichedDeal:
    """Phase-A output for one deal. Phase B reads only this — never a raw
    ``deal`` dict's ``image_urls``/title for fetching — which is what makes
    Phase B provably network-call-free."""

    bundle_prices: list[dict] = field(default_factory=list)
    single_price: float | None = None
    images: list[tuple[bytes, str]] = field(default_factory=list)  # (data, mime_type)


def _query_jobs_for_deal(deal: dict) -> tuple[bool, list[tuple[str, str]]]:
    """Pure computation (no I/O) — return ``(is_bundle, [(label, query), ...])``.

    For a bundle-keyword title with extractable individual game titles: one
    ``(game_name, query)`` pair per game. For anything else — including a
    bundle-keyword title with NO extractable game titles (e.g. "Xbox 360
    Spiele Konvolut Sammlung 11 Stück", which names no individual games) —
    a single ``(title, query)`` pair treating the whole title as one
    listing, matching the original single-deal fallback behavior exactly.
    """
    title = deal.get("title", "")
    if _BUNDLE_TITLE_KEYWORDS_RE.search(title):
        game_titles = _extract_potential_game_titles(title)
        if game_titles:
            platform = _extract_platform_name(title)
            jobs = []
            for game in game_titles:
                q = f"{game} ({platform})" if platform else game
                if len(q) >= 5:
                    jobs.append((game, q))
            if jobs:
                return True, jobs
    q = _build_single_game_search_query(title)
    return False, ([(title, q)] if q else [])


class Enricher:
    """Owns the eBay price cache and the shared enrichment thread pool.

    One instance lives for the lifetime of an assessor (like the old
    per-assessor prefetch executor it replaces) so the price cache and pool
    persist across searches, not just within one ``enrich_deals`` call.
    """

    def __init__(self) -> None:
        self.ebay_client: Any | None = None
        self._price_cache: dict[str, tuple[float | None, str, float]] = {}
        # Never used as a context manager / never shut down mid-lifecycle —
        # a `with` block's __exit__ calls shutdown(wait=True), which would
        # block on every straggler future finishing before returning,
        # defeating the whole point of a bounded concurrent.futures.wait()
        # below. A persistent pool lets stragglers keep running in the
        # background and still populate the price cache for next time.
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None

    # ── Price cache ────────────────────────────────────────────────────────

    def _cached_price(self, query: str) -> tuple[float | None, str] | None:
        entry = self._price_cache.get(query)
        if entry is None:
            return None
        price, source, expire_at = entry
        if time.monotonic() >= expire_at:
            del self._price_cache[query]
            return None
        return price, source

    def _store_price(self, query: str, price: float | None, source: str) -> None:
        self._price_cache[query] = (price, source, time.monotonic() + _EBAY_CACHE_TTL)

    # ── Fetchers — each is one boundable unit of network I/O ────────────────

    def _fetch_one_price(self, query: str) -> tuple[float | None, str]:
        try:
            price, source, _errors = self.ebay_client.get_lowest_market_price(query, max_results=10)
            return price, source
        except Exception as exc:
            logger.warning("Enricher: eBay price lookup failed for %r: %s", query, exc)
            return None, "no_result"

    @staticmethod
    def _fetch_one_image(url: str) -> tuple[bytes, str] | None:
        try:
            resp = requests.get(url, timeout=_IMAGE_FETCH_TIMEOUT)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            mime_type = content_type.split(";")[0].strip()
            return resp.content, mime_type
        except Exception as exc:
            logger.debug("Enricher: failed to fetch image %r: %s", url, exc)
            return None

    # ── Phase A entry point ──────────────────────────────────────────────────

    def enrich_deals(self, deals: list[dict], deadline: float, *, fetch_images: bool = True) -> list[EnrichedDeal]:
        """Fetch every eBay price lookup and every image for *deals* at once,
        bounded so this call never runs past *deadline* by more than a
        capped fraction of whatever time is left.

        ``fetch_images=False`` skips image fetching entirely — the caller
        (a provider whose active model doesn't support image input) already
        knows any fetched images would be discarded, so there's no reason to
        spend part of the deadline budget on them.

        Returns one :class:`EnrichedDeal` per input deal, same order.
        """
        results = [EnrichedDeal() for _ in deals]
        if not deals:
            return results

        if self.ebay_client is not None:
            jobs_by_deal = [_query_jobs_for_deal(deal) for deal in deals]
        else:
            jobs_by_deal = [(False, []) for _ in deals]

        unique_queries = {q for _, jobs in jobs_by_deal for _, q in jobs}
        uncached_queries = [q for q in unique_queries if self._cached_price(q) is None]
        image_jobs = (
            [(i, url) for i, deal in enumerate(deals) for url in (deal.get("image_urls") or [])[:_MAX_IMAGES]]
            if fetch_images
            else []
        )

        remaining = max(0.0, deadline - time.monotonic())
        # Never let enrichment alone consume more than half of whatever time
        # is actually left (capped at _ENRICH_MAX_BUDGET_S) — leaving room
        # for the Gemini calls that follow matters more than a perfectly
        # warm price cache. A partially-enriched batch still gets assessed;
        # one that never reaches Gemini because enrichment used the whole
        # deadline never does.
        budget = min(_ENRICH_MAX_BUDGET_S, remaining / 2)

        if (uncached_queries or image_jobs) and budget > 0:
            if self._pool is None:
                self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=_ENRICH_MAX_WORKERS)

            price_futures = {self._pool.submit(self._fetch_one_price, q): q for q in uncached_queries}
            image_futures = [(i, self._pool.submit(self._fetch_one_image, url)) for i, url in image_jobs]

            done, _not_done = concurrent.futures.wait(
                list(price_futures) + [f for _, f in image_futures], timeout=budget
            )

            for fut in done:
                query = price_futures.get(fut)
                if query is None:
                    continue
                try:
                    price, source = fut.result()
                    self._store_price(query, price, source)
                except Exception as exc:
                    logger.warning("Enricher: price future for %r raised: %s", query, exc)

            image_results: dict[int, list[tuple[bytes, str]]] = {}
            for i, fut in image_futures:
                if fut not in done:
                    continue
                try:
                    img = fut.result()
                except Exception as exc:
                    logger.debug("Enricher: image future raised: %s", exc)
                    img = None
                if img is not None:
                    image_results.setdefault(i, []).append(img)
            for i, imgs in image_results.items():
                results[i].images = imgs

        for i, (is_bundle, jobs) in enumerate(jobs_by_deal):
            if not jobs:
                continue
            if is_bundle:
                bundle_prices = []
                for game, q in jobs:
                    cached = self._cached_price(q)
                    price, source = cached if cached is not None else (None, "no_result")
                    if price is not None:
                        price_source = "ebay_sold" if source == "sold_listings" else "ebay_active"
                        bundle_prices.append({"game": game, "price_eur": round(price, 2), "price_source": price_source})
                    else:
                        bundle_prices.append({"game": game, "price_eur": None, "price_source": "no_result"})
                results[i].bundle_prices = bundle_prices
            else:
                _, q = jobs[0]
                cached = self._cached_price(q)
                if cached is not None:
                    results[i].single_price = cached[0]

        return results
