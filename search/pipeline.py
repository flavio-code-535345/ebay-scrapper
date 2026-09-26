"""The search pipeline: fetch every source concurrently under a time budget,
de-duplicate, filter, then rank and pick the deals that get AI budget.

Stages (each a small pure function, tested on its own):
  fetch_all   → concurrent source requests; a source that misses the budget
                becomes an error in its report instead of stalling the search
  dedupe      → one entry per listing (stable listing ID across URL shapes,
                then normalized title+price for cross-posted listings)
  apply_filters → skipped listings, non-German items, auctions that don't end
                within two days, single games posing as bundles (bundle
                searches only), sports-only bundles, listings for a
                different platform than the one searched
  check_prices → a price the listing text says is per game ("Stückpreis
                7 €") is marked as such, or replaced by the whole-lot price
                the text states ("komplett Paket 120 €")
  select      → score (query match, freshness, price per game) and fill the
                AI slots so every source that has relevant results is heard,
                instead of whichever source was merged first crowding it out
  complete_descriptions → fetch the full text of the few selected
                Kleinanzeigen bundles that look too cheap to be true (search
                results cut descriptions at ~100 characters) and re-check them
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ai_providers.base import (
    _BUNDLE_TITLE_KEYWORDS_RE,
    _PLATFORM_MAP,
    _SPORTS_KINECT_KEYWORDS_RE,
    _detect_sports_kinect_deal,
    analyze_price_scope,
    bundle_game_count,
    is_offer_placeholder,
    is_single_game_listing,
    looks_like_multi_item,
)
from models import canonical_listing_id
from search.query import SearchPlan, platforms_named

logger = logging.getLogger(__name__)

SearchFn = Callable[..., tuple[list[dict], list[str]]]
DescribeFn = Callable[..., tuple[str | None, list[str]]]  # (url, cached_only=False) → (text, errors)

# Auctions are only worth showing when bidding closes soon: the current bid of
# an auction with a week to go says little about what it will sell for, and
# nobody wants to wait that long. Auctions ending later — or whose end time
# no source reported — are dropped.
AUCTION_MAX_TIME_LEFT = timedelta(days=2)

# Shared by every search: a pool that is never torn down mid-request, so a
# source that overruns the budget keeps running in the background instead of
# blocking the response (a `with` block would wait for it on exit).
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="search")


@dataclass
class SourceJob:
    source: str  # "ebay", "ebay_auctions", "kleinanzeigen"
    fn: SearchFn
    query: str
    max_results: int


@dataclass
class SourceReport:
    source: str
    queries: list[str] = field(default_factory=list)
    count: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    timed_out: bool = False

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "queries": self.queries,
            "count": self.count,
            "errors": self.errors,
            "elapsed_s": round(self.elapsed_s, 2),
            "timed_out": self.timed_out,
        }


@dataclass
class SearchOutcome:
    selected: list[dict]
    candidates: int
    errors: list[str]
    sources: list[SourceReport]
    removed: dict[str, int]


# ── Fetch ────────────────────────────────────────────────────────────────────


def fetch_all(jobs: list[SourceJob], budget_s: float) -> tuple[list[tuple[SourceJob, list[dict]]], list[SourceReport]]:
    """Run every job concurrently; wait at most *budget_s* in total.

    Returns the per-job results in job order (deterministic merge order no
    matter which request finishes first) and one report per source.
    """
    reports: dict[str, SourceReport] = {}
    for job in jobs:
        reports.setdefault(job.source, SourceReport(job.source)).queries.append(job.query)

    def timed(job: SourceJob):
        t0 = time.monotonic()
        deals, errors = job.fn(job.query, max_results=job.max_results)
        return deals, errors, time.monotonic() - t0

    futures = [(job, _POOL.submit(timed, job)) for job in jobs]
    done, _ = concurrent.futures.wait([f for _, f in futures], timeout=max(0.0, budget_s))

    results: list[tuple[SourceJob, list[dict]]] = []
    for job, fut in futures:
        report = reports[job.source]
        if fut not in done:
            report.timed_out = True
            report.elapsed_s = max(report.elapsed_s, budget_s)
            report.errors.append(f"{job.source} search for {job.query!r} exceeded the {budget_s:.0f}s search budget")
            continue
        try:
            deals, errors, elapsed = fut.result()
        except Exception as exc:
            logger.warning("%s search failed for %r: %s", job.source, job.query, exc)
            report.errors.append(f"{job.source} search error: {exc}")
            continue
        report.errors.extend(errors)
        report.elapsed_s = max(report.elapsed_s, elapsed)
        report.count += len(deals)
        results.append((job, deals))
    return results, list(reports.values())


# ── Dedupe ───────────────────────────────────────────────────────────────────

_NON_ALNUM_RE = re.compile(r"[^0-9a-zäöüß]+")


def listing_key(deal: dict) -> str:
    """Stable identity: the listing ID when known, else the URL without its query string."""
    return deal.get("listing_id") or canonical_listing_id(deal.get("url")) or (deal.get("url") or "").split("?")[0]


def dedupe(results: list[tuple[SourceJob, list[dict]]]) -> list[dict]:
    by_key: dict[str, dict] = {}
    seen_title_price: set[str] = set()
    merged: list[dict] = []
    for job, deals in results:
        for deal in deals:
            key = listing_key(deal)
            if not key:
                continue
            if key in by_key:
                # e.g. an auction with a Buy-It-Now option: the newest-first
                # Buy-It-Now leg can't see its end time, the auction leg can.
                kept = by_key[key]
                if deal.get("auction_end") and not kept.get("auction_end"):
                    kept["auction_end"] = deal["auction_end"]
                continue
            price = deal.get("price") or 0
            norm_title = _NON_ALNUM_RE.sub(" ", (deal.get("title") or "").lower()).strip()
            title_price = f"{norm_title}|{price:.2f}" if price else ""
            if title_price and title_price in seen_title_price:
                continue
            by_key[key] = deal
            if title_price:
                seen_title_price.add(title_price)
            deal.setdefault("source", "kleinanzeigen" if job.source == "kleinanzeigen" else "ebay")
            if job.source == "ebay_auctions":
                deal["listing_type"] = "auction"
            deal.setdefault("listing_id", key)
            merged.append(deal)
    return merged


# ── Filters ──────────────────────────────────────────────────────────────────


def is_german_location(location: str | None) -> bool:
    """True when *location* is in Germany or unknown (benefit of the doubt)."""
    if not location:
        return True
    upper = location.strip().upper()
    return upper == "DE" or upper.endswith(", DE") or "DEUTSCHLAND" in upper or "GERMANY" in upper


_SPORTS_NOISE_RE = re.compile(r"\b(je\s+stk|stk|pro|jede|und|mit|für|oder|stück|wahl|aus)\b")
_QUANTITY_RE = re.compile(r"\d+\s*[€x×]|\b\d+\b")
_SPORTS_FILLER_WORDS = frozenset(
    {"die", "der", "das", "ein", "sie", "von", "spiele", "spiel", "games", "game", "pal", "ovp", "teile", "stk"}
)


def is_sports_only(deal: dict) -> bool:
    """A sports/Kinect listing with too little else in its title to be worth
    AI time. Mixed bundles (≥3 other meaningful words) are kept — the
    non-sports games may still make them profitable."""
    if not _detect_sports_kinect_deal(deal):
        return False
    cleaned = _SPORTS_KINECT_KEYWORDS_RE.sub(" ", (deal.get("title") or "").lower())
    # Bundle words and platform names say nothing about *which* games are in
    # it — "FIFA Sammlung Konvolut PS4" is still sports-only.
    cleaned = _BUNDLE_TITLE_KEYWORDS_RE.sub(" ", cleaned)
    for pattern, _ in _PLATFORM_MAP:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = _QUANTITY_RE.sub(" ", _SPORTS_NOISE_RE.sub(" ", cleaned))
    tokens = [t for t in cleaned.split() if len(t) > 2 and t not in _SPORTS_FILLER_WORDS]
    return len(tokens) < 3


def _parse_iso(raw) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def auction_ends_too_late(deal: dict, now: datetime) -> bool:
    """An auction that won't end within :data:`AUCTION_MAX_TIME_LEFT`, has
    already ended, or has no known end time. Fixed-price listings never are."""
    if deal.get("listing_type") != "auction":
        return False
    end = _parse_iso(deal.get("auction_end"))
    return end is None or not now < end <= now + AUCTION_MAX_TIME_LEFT


def is_other_platform(deal: dict, platform: str | None) -> bool:
    """The title names only platforms other than the one searched for."""
    if not platform:
        return False
    named = platforms_named(deal.get("title") or "")
    return bool(named) and platform not in named


def apply_filters(
    deals: list[dict], plan: SearchPlan, skipped: set[str], now: datetime | None = None
) -> tuple[list[dict], dict[str, int]]:
    now = now or datetime.now(UTC)
    removed = {
        "skipped": 0,
        "not_germany": 0,
        "auction_ends_late": 0,
        "not_a_bundle": 0,
        "sports_only": 0,
        "other_platform": 0,
    }
    kept = []
    for deal in deals:
        if deal.get("url") in skipped or listing_key(deal) in skipped:
            removed["skipped"] += 1
        elif deal.get("source") != "kleinanzeigen" and not is_german_location(deal.get("item_location")):
            removed["not_germany"] += 1
        elif auction_ends_too_late(deal, now):
            removed["auction_ends_late"] += 1
        elif plan.bundle_intent and is_single_game_listing(deal):
            removed["not_a_bundle"] += 1
        elif is_sports_only(deal):
            removed["sports_only"] += 1
        elif is_other_platform(deal, plan.platform):
            removed["other_platform"] += 1
        else:
            kept.append(deal)
    return kept, removed


# ── Ranking and selection ────────────────────────────────────────────────────

_FRESHNESS_HALF_LIFE_DAYS = 3.0


def game_count(title: str) -> int | None:
    """Number of games a bundle title claims ("26 Spiele", "10 PS3 Spiele / 6 PS4
    Spiele" → 16); platform numbers ("Xbox 360", "PS4") are not counts."""
    return bundle_game_count(title) or None


# ── Price checks ─────────────────────────────────────────────────────────────


def check_prices(deals: list[dict]) -> int:
    """Mark prices that don't buy the whole lot; returns how many were marked.

    - "1 € VB" on Kleinanzeigen is a make-an-offer placeholder, not a price.
    - A per-game price ("Stück preis 7euro") is replaced by the whole-lot
      price when the text states one ("oder komplett paket 120 euro inkl
      versand") — that is what the bundle costs — and the listed amount is
      kept in ``listed_price``.
    - Otherwise the price stays but is marked ``per_item``; the assessor
      rates such listings "Avoid" (analyze_price_scope sees the same text).

    Idempotent: deals that already carry a ``price_basis`` are skipped.
    """
    marked = 0
    for deal in deals:
        if deal.get("price_basis"):
            continue
        if is_offer_placeholder(deal):
            deal["price_basis"] = "offer"
            deal["price_note"] = (
                'Make-an-offer listing ("VB"): the amount shown is a placeholder, not the asking price.'
            )
            marked += 1
            continue
        scope = analyze_price_scope(deal)
        if scope is None:
            continue
        deal["price_note"] = scope.explain(deal)
        if scope.lot_price is not None:
            deal["listed_price"] = deal.get("price")
            deal["price"] = scope.lot_price
            deal["shipping_note"] = "VB" if scope.lot_is_negotiable else ""
            if scope.lot_includes_shipping:
                deal["shipping"], deal["shipping_cost"] = "inkl. Versand", 0.0
            deal["price_basis"] = "lot_from_text"
        else:
            deal["price_basis"] = "per_item"
        marked += 1
    return marked


# ── Full descriptions for suspiciously cheap bundles ─────────────────────────

# Each full description is one more request to a site that IP-bans at about
# six a minute, so only the most suspicious few are fetched per search; the
# scraper caches them, so repeated searches cost nothing.
_MAX_DESCRIPTION_FETCHES = 2
_DESCRIPTION_BUDGET_MAX_S = 12.0
_TOO_CHEAP_PER_GAME_EUR = 3.0
_TOO_CHEAP_BUNDLE_EUR = 15.0


def _is_truncated(deal: dict) -> bool:
    return deal.get("source") == "kleinanzeigen" and (deal.get("description") or "").rstrip().endswith("...")


def _price_per_game(deal: dict) -> float:
    total = (deal.get("price") or 0) + (deal.get("shipping_cost") or 0)
    return total / (game_count(deal.get("title") or "") or 1)


def needs_full_description(deal: dict) -> bool:
    """A Kleinanzeigen bundle, cut off in the search results, cheap enough that
    the listed price is likely per game — the ones the AI would call a steal."""
    if deal.get("price_basis") or not _is_truncated(deal) or not (deal.get("price") or 0) > 0:
        return False
    title = deal.get("title") or ""
    if not looks_like_multi_item(title):
        return False
    if game_count(title):
        return _price_per_game(deal) < _TOO_CHEAP_PER_GAME_EUR
    return (deal.get("price") or 0) <= _TOO_CHEAP_BUNDLE_EUR


def complete_descriptions(
    deals: list[dict], plan: SearchPlan, describe: DescribeFn, budget_s: float
) -> tuple[list[dict], list[str]]:
    """Swap in full descriptions (free when cached; fetched for at most
    ``_MAX_DESCRIPTION_FETCHES`` suspicious bundles), then re-check each updated
    deal: its price, and — on a bundle search — whether it is a bundle at all.
    Returns the deals to keep and any fetch errors."""
    updated: list[dict] = []
    for deal in (d for d in deals if _is_truncated(d)):
        text, _ = describe(deal["url"], cached_only=True)
        if text:
            deal["description"] = text
            updated.append(deal)

    suspicious = sorted((d for d in deals if needs_full_description(d)), key=_price_per_game)
    futures = [(deal, _POOL.submit(describe, deal["url"])) for deal in suspicious[:_MAX_DESCRIPTION_FETCHES]]
    errors: list[str] = []
    if futures:
        done, _ = concurrent.futures.wait([f for _, f in futures], timeout=max(0.0, budget_s))
        for deal, fut in futures:
            if fut not in done:
                continue  # still running: it lands in the scraper's cache for the next search
            try:
                text, fetch_errors = fut.result()
            except Exception as exc:
                errors.append(f"Kleinanzeigen description fetch failed: {exc}")
                continue
            errors.extend(fetch_errors)
            if text:
                deal["description"] = text
                updated.append(deal)

    check_prices(updated)
    dropped = {id(d) for d in updated if plan.bundle_intent and is_single_game_listing(d)}
    if updated:
        logger.info("Full descriptions: %d updated, %d turned out to be single games", len(updated), len(dropped))
    return [d for d in deals if id(d) not in dropped], errors


def _freshness(deal: dict, now: datetime) -> float:
    listed = _parse_iso(deal.get("listing_date"))
    if listed is None:
        return 0.4
    age_days = max(0.0, (now - listed).total_seconds() / 86400)
    return math.pow(0.5, age_days / _FRESHNESS_HALF_LIFE_DAYS)


def _relevance(deal: dict, plan: SearchPlan) -> float:
    title = (deal.get("title") or "").lower()
    compact = title.replace(" ", "")
    score = 0.0
    if plan.core_terms:
        matched = sum(1 for t in plan.core_terms if t in title or t in compact)
        score += 0.6 * matched / len(plan.core_terms)
    else:
        score += 0.6
    if plan.platform and plan.platform in platforms_named(deal.get("title") or ""):
        score += 0.2
    if not plan.bundle_intent or _BUNDLE_TITLE_KEYWORDS_RE.search(title) or game_count(title):
        score += 0.2
    return score


def _value(deal: dict) -> float:
    price = deal.get("price") or 0
    if deal.get("price_basis") == "offer":
        return 0.3  # no real asking price
    # A per-game price is the price of one game, whatever the title's count.
    count = 1 if deal.get("price_basis") == "per_item" else game_count(deal.get("title") or "")
    if not price or not count:
        return 0.3
    per_game = (price + (deal.get("shipping_cost") or 0)) / count
    return 1 / (1 + per_game / 3)  # €3 per game → 0.5, €1 → 0.75, €10 → ~0.23


def score(deal: dict, plan: SearchPlan, now: datetime) -> float:
    return 0.5 * _relevance(deal, plan) + 0.3 * _freshness(deal, now) + 0.2 * _value(deal)


def _bucket(deal: dict) -> str:
    if deal.get("source") == "kleinanzeigen":
        return "kleinanzeigen"
    return "ebay_auction" if deal.get("listing_type") == "auction" else "ebay_fixed"


def select(deals: list[dict], plan: SearchPlan, limit: int, now: datetime | None = None) -> list[dict]:
    """Pick up to *limit* deals: each source bucket (eBay Buy-It-Now, eBay
    auctions, Kleinanzeigen) first gets an equal share of its own best
    listings, then any remaining slots go to the best of the rest overall.

    Returned best-first. Deterministic: ties keep their merge order.
    """
    now = now or datetime.now(UTC)
    ranked = sorted(enumerate(deals), key=lambda iv: (-score(iv[1], plan, now), iv[0]))
    buckets: dict[str, list[tuple[int, dict]]] = {}
    for item in ranked:
        buckets.setdefault(_bucket(item[1]), []).append(item)

    share = limit // len(buckets) if buckets else 0
    chosen = {idx for items in buckets.values() for idx, _ in items[:share]}
    for idx, _ in ranked:
        if len(chosen) >= limit:
            break
        chosen.add(idx)
    return [deal for idx, deal in ranked if idx in chosen]


# ── Entry point ──────────────────────────────────────────────────────────────


def run_search(
    plan: SearchPlan,
    *,
    ebay_search: SearchFn,
    ebay_is_api: bool,
    auction_search: SearchFn | None,
    kleinanzeigen_search: SearchFn | None,
    skipped: set[str],
    budget_s: float,
    limit: int = 30,
    kleinanzeigen_describe: DescribeFn | None = None,
) -> SearchOutcome:
    t0 = time.monotonic()
    ebay_queries = plan.ebay_api if ebay_is_api else plan.ebay_web
    jobs = [SourceJob("ebay", ebay_search, q, 200 if ebay_is_api else 120) for q in ebay_queries]
    if auction_search:
        # Sources filter by end time themselves (fewer wasted result slots);
        # apply_filters re-checks every auction, whichever leg it came from.
        ending_soon = functools.partial(auction_search, ends_within=AUCTION_MAX_TIME_LEFT)
        jobs += [SourceJob("ebay_auctions", ending_soon, q, 100) for q in ebay_queries]
    if kleinanzeigen_search:
        jobs += [SourceJob("kleinanzeigen", kleinanzeigen_search, q, 25) for q in plan.kleinanzeigen]

    results, reports = fetch_all(jobs, budget_s)
    merged = dedupe(results)
    filtered, removed = apply_filters(merged, plan, skipped)
    repriced = check_prices(filtered)
    selected = select(filtered, plan, limit)
    if kleinanzeigen_describe:
        budget_left = min(_DESCRIPTION_BUDGET_MAX_S, budget_s - (time.monotonic() - t0))
        selected, describe_errors = complete_descriptions(selected, plan, kleinanzeigen_describe, budget_left)
        report = next((r for r in reports if r.source == "kleinanzeigen"), None)
        if report is not None:
            report.errors.extend(describe_errors)
    logger.info(
        "Search %r: %d requests, %d unique, %d after filters %s, %d price notes, %d selected",
        plan.label,
        len(jobs),
        len(merged),
        len(filtered),
        removed,
        repriced,
        len(selected),
    )
    errors = [e for r in reports for e in r.errors]
    return SearchOutcome(selected=selected, candidates=len(filtered), errors=errors, sources=reports, removed=removed)
