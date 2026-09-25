"""The search pipeline: fetch every source concurrently under a time budget,
de-duplicate, filter, then rank and pick the deals that get AI budget.

Stages (each a small pure function, tested on its own):
  fetch_all   → concurrent source requests; a source that misses the budget
                becomes an error in its report instead of stalling the search
  dedupe      → one entry per listing (stable listing ID across URL shapes,
                then normalized title+price for cross-posted listings)
  apply_filters → skipped listings, non-German items, sports-only bundles,
                listings for a different platform than the one searched
  select      → score (query match, freshness, price per game) and fill the
                AI slots so every source that has relevant results is heard,
                instead of whichever source was merged first crowding it out
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ai_providers.base import (
    _BUNDLE_TITLE_KEYWORDS_RE,
    _PLATFORM_MAP,
    _SPORTS_KINECT_KEYWORDS_RE,
    _detect_sports_kinect_deal,
)
from models import canonical_listing_id
from search.query import SearchPlan, platforms_named

logger = logging.getLogger(__name__)

SearchFn = Callable[..., tuple[list[dict], list[str]]]

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
    seen_keys: set[str] = set()
    seen_title_price: set[str] = set()
    merged: list[dict] = []
    for job, deals in results:
        for deal in deals:
            key = listing_key(deal)
            if not key or key in seen_keys:
                continue
            price = deal.get("price") or 0
            norm_title = _NON_ALNUM_RE.sub(" ", (deal.get("title") or "").lower()).strip()
            title_price = f"{norm_title}|{price:.2f}" if price else ""
            if title_price and title_price in seen_title_price:
                continue
            seen_keys.add(key)
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


def is_sports_only(deal: dict) -> bool:
    """A sports/Kinect listing with too little else in its title to be worth
    AI time. Mixed bundles (≥3 other meaningful words) are kept — the
    non-sports games may still make them profitable."""
    if not _detect_sports_kinect_deal(deal):
        return False
    cleaned = _SPORTS_KINECT_KEYWORDS_RE.sub(" ", (deal.get("title") or "").lower())
    cleaned = _QUANTITY_RE.sub(" ", _SPORTS_NOISE_RE.sub(" ", cleaned))
    tokens = [t for t in cleaned.split() if len(t) > 2 and t not in ("die", "der", "das", "ein", "sie", "von")]
    return len(tokens) < 3


def is_other_platform(deal: dict, platform: str | None) -> bool:
    """The title names only platforms other than the one searched for."""
    if not platform:
        return False
    named = platforms_named(deal.get("title") or "")
    return bool(named) and platform not in named


def apply_filters(deals: list[dict], plan: SearchPlan, skipped: set[str]) -> tuple[list[dict], dict[str, int]]:
    removed = {"skipped": 0, "not_germany": 0, "sports_only": 0, "other_platform": 0}
    kept = []
    for deal in deals:
        if deal.get("url") in skipped or listing_key(deal) in skipped:
            removed["skipped"] += 1
        elif deal.get("source") != "kleinanzeigen" and not is_german_location(deal.get("item_location")):
            removed["not_germany"] += 1
        elif is_sports_only(deal):
            removed["sports_only"] += 1
        elif is_other_platform(deal, plan.platform):
            removed["other_platform"] += 1
        else:
            kept.append(deal)
    return kept, removed


# ── Ranking and selection ────────────────────────────────────────────────────

_GAME_COUNT_RE = re.compile(
    r"(?<![\w.,])(\d{1,3})\s*(?:x\s*)?(?:spiele|games|titel|stück|stk\.?|videospiele)\b", re.IGNORECASE
)
_FRESHNESS_HALF_LIFE_DAYS = 3.0


def game_count(title: str) -> int | None:
    """Number of games a bundle title claims ("26 Spiele", "ca. 94 ... Spiele").

    Platform names are removed first — otherwise "Xbox 360 Spiele" reads as
    360 games and "PS4 Spiele" as four.
    """
    text = title or ""
    for pattern, _ in _PLATFORM_MAP:
        text = pattern.sub(" ", text)
    m = _GAME_COUNT_RE.search(text)
    if not m:
        return None
    n = int(m.group(1))
    return n if 2 <= n <= 500 else None


def _freshness(deal: dict, now: datetime) -> float:
    raw = deal.get("listing_date")
    if not raw:
        return 0.4
    try:
        listed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
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
    count = game_count(deal.get("title") or "")
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
) -> SearchOutcome:
    ebay_queries = plan.ebay_api if ebay_is_api else plan.ebay_web
    jobs = [SourceJob("ebay", ebay_search, q, 200 if ebay_is_api else 120) for q in ebay_queries]
    if auction_search:
        jobs += [SourceJob("ebay_auctions", auction_search, q, 50) for q in plan.ebay_api]
    if kleinanzeigen_search:
        jobs += [SourceJob("kleinanzeigen", kleinanzeigen_search, q, 25) for q in plan.kleinanzeigen]

    results, reports = fetch_all(jobs, budget_s)
    merged = dedupe(results)
    filtered, removed = apply_filters(merged, plan, skipped)
    selected = select(filtered, plan, limit)
    logger.info(
        "Search %r: %d requests, %d unique, %d after filters %s, %d selected",
        plan.label,
        len(jobs),
        len(merged),
        len(filtered),
        removed,
        len(selected),
    )
    errors = [e for r in reports for e in r.errors]
    return SearchOutcome(selected=selected, candidates=len(filtered), errors=errors, sources=reports, removed=removed)
