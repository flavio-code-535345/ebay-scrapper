"""Shared deal schema — canonical field reference, condition vocabulary, and
listing-date/sort helpers used by every deal-producing and deal-consuming
module (the three scrapers, the AI assessor, ``app.py``, ``database.py``).

Deals still flow through the app as plain ``dict`` objects (every existing
call site does ``deal.get(...)``/``{**deal, ...}``/``jsonify(deal)``), so
:class:`Deal` is a ``TypedDict`` — a documented, importable schema — rather
than a dataclass that would force every call site to switch from ``.get()``
to attribute access for no behavioral gain.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TypedDict
from zoneinfo import ZoneInfo


class Condition(StrEnum):
    """One shared condition vocabulary.

    Each deal source describes condition differently: the eBay Browse API
    uses a fixed 10-value English enum, the HTML scraper keeps whatever raw
    German text it found on the page, and Kleinanzeigen uses its own 5-value
    German enum. Anything that needs to *compare* conditions across sources
    needs one common scale — this is it. It is intentionally a *new,
    additional* field (``condition_normalized``); the original ``condition``
    string keeps flowing to the frontend/tests unchanged.
    """

    NEW = "new"
    NEW_OTHER = "new_other"
    NEW_WITH_DEFECTS = "new_with_defects"
    REFURBISHED = "refurbished"
    USED = "used"
    VERY_GOOD = "very_good"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    FOR_PARTS = "for_parts"
    UNKNOWN = "unknown"


# Exact (case-insensitive) label matches — covers the eBay API's fixed
# English labels (ai_providers/ebay_api_client._CONDITION_ID_MAP values) and
# Kleinanzeigen's fixed German labels precisely, with no ambiguity.
_EXACT_CONDITION_LABELS: dict[str, Condition] = {
    "new": Condition.NEW,
    "new – other": Condition.NEW_OTHER,
    "new - other": Condition.NEW_OTHER,
    "new with defects": Condition.NEW_WITH_DEFECTS,
    "manufacturer refurbished": Condition.REFURBISHED,
    "seller refurbished": Condition.REFURBISHED,
    "used": Condition.USED,
    "very good": Condition.VERY_GOOD,
    "good": Condition.GOOD,
    "acceptable": Condition.ACCEPTABLE,
    "for parts or not working": Condition.FOR_PARTS,
    "neu": Condition.NEW,
    "neu (sonstige)": Condition.NEW_OTHER,
    "sehr gut": Condition.VERY_GOOD,
    "gebraucht": Condition.USED,
    "defekt": Condition.FOR_PARTS,
}

# Substring fallback rules for the HTML scraper's free-text condition strings
# (e.g. "Gebraucht - Akzeptabler Zustand", "Sehr guter Zustand"), checked in
# order — most-specific first — since a string can contain more than one
# keyword.
_KEYWORD_CONDITION_RULES: list[tuple[tuple[str, ...], Condition]] = [
    (("for parts", "not working", "defekt", "kaputt", "ersatzteile", "bastler"), Condition.FOR_PARTS),
    (("new with defects",), Condition.NEW_WITH_DEFECTS),
    (("refurbished", "generalüberholt"), Condition.REFURBISHED),
    (("very good", "sehr guter", "sehr gut", "einwandfrei"), Condition.VERY_GOOD),
    # "akzeptab" (not "akzeptabel") because German adjective endings inflect
    # ("akzeptabler Zustand"), and "akzeptabler" is not a substring of
    # "akzeptabel" — the shared stem is.
    (("acceptable", "akzeptab"), Condition.ACCEPTABLE),
    (("neu", "new", "ovp", "originalverpackt", "unbenutzt", "sealed"), Condition.NEW),
    (("good", "guter zustand", "gut"), Condition.GOOD),
    (("used", "gebraucht"), Condition.USED),
]


def normalize_condition(raw: str | None) -> Condition:
    """Map any source's condition string into the shared :class:`Condition` scale."""
    if not raw:
        return Condition.UNKNOWN
    # eBay's result cards append the seller type: "Gebraucht | Privat".
    text = raw.split("|")[0].strip().lower()
    exact = _EXACT_CONDITION_LABELS.get(text)
    if exact is not None:
        return exact
    for keywords, condition in _KEYWORD_CONDITION_RULES:
        if any(kw in text for kw in keywords):
            return condition
    return Condition.UNKNOWN


# ── Listing-date parsing ────────────────────────────────────────────────────
# Each source module calls parse_listing_date() once, at scrape/ingestion
# time, and stores the result back as an ISO-8601 string (or None) — so every
# deal dict's "listing_date" field uses the SAME representation regardless of
# which source produced it, and nothing downstream (sorting, display) needs
# to know the original per-source format.

# Both German sources print wall-clock times in German local time, not UTC.
# Needs the IANA database — shipped by the `tzdata` package on platforms (like
# Windows and slim containers) that don't have one of their own.
_BERLIN = ZoneInfo("Europe/Berlin")

_KLEINANZEIGEN_TODAY_RE = re.compile(r"^heute,?\s*(\d{1,2}):(\d{2})$", re.IGNORECASE)
_KLEINANZEIGEN_YESTERDAY_RE = re.compile(r"^gestern,?\s*(\d{1,2}):(\d{2})$", re.IGNORECASE)
_KLEINANZEIGEN_ABS_DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")

# eBay's result cards (newest-first sort) say e.g. "Vor 30 Min. eingestellt",
# "Vor 5 Std. eingestellt", "Vor 1 T. eingestellt".
_EBAY_RELATIVE_AGE_RE = re.compile(r"vor\s+(\d+)\s*(min|std|t|tg|tag|tagen|w|wo|woche|wochen)\b\.?", re.IGNORECASE)
_EBAY_AGE_UNITS: dict[str, timedelta] = {
    "min": timedelta(minutes=1),
    "std": timedelta(hours=1),
    "t": timedelta(days=1),
    "tg": timedelta(days=1),
    "tag": timedelta(days=1),
    "tagen": timedelta(days=1),
    "w": timedelta(weeks=1),
    "wo": timedelta(weeks=1),
    "woche": timedelta(weeks=1),
    "wochen": timedelta(weeks=1),
}


def _parse_kleinanzeigen_date(raw: str, now: datetime) -> datetime | None:
    """Parse Kleinanzeigen's date text ("Heute, 19:30", "Gestern, 10:15",
    "05.09.2025") as German local time. Anything else returns None rather than
    a guess."""
    text = raw.strip()
    today = now.astimezone(_BERLIN)
    for pattern, days_back in ((_KLEINANZEIGEN_TODAY_RE, 0), (_KLEINANZEIGEN_YESTERDAY_RE, 1)):
        m = pattern.match(text)
        if m:
            day = today - timedelta(days=days_back)
            local = day.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
            return local.astimezone(UTC)
    m = _KLEINANZEIGEN_ABS_DATE_RE.match(text)
    if m:
        try:
            local = datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)), tzinfo=_BERLIN)
        except ValueError:
            return None
        return local.astimezone(UTC)
    return None


# Older listings show an absolute day instead: "Eingestellt am Sep 20"
# (eBay mixes English and German month abbreviations).
_EBAY_LISTED_ON_RE = re.compile(r"eingestellt\s+am\s+([a-zäöü]{3})[a-zäöü]*\.?\s+(\d{1,2})\b", re.IGNORECASE)
_ENGLISH_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_MONTHS = {
    **{m: i for i, m in enumerate(_ENGLISH_MONTHS, 1)},
    **{"mär": 3, "mrz": 3, "mai": 5, "okt": 10, "dez": 12},
}


def _parse_ebay_listing_age(raw: str, now: datetime) -> datetime | None:
    m = _EBAY_RELATIVE_AGE_RE.search(raw)
    if m:
        return now - int(m.group(1)) * _EBAY_AGE_UNITS[m.group(2).lower()]
    m = _EBAY_LISTED_ON_RE.search(raw)
    month = _MONTHS.get(m.group(1).lower()) if m else None
    if not month:
        return None
    today = now.astimezone(_BERLIN)
    try:
        listed = datetime(today.year, month, int(m.group(2)), tzinfo=_BERLIN)
    except ValueError:
        return None
    if listed > today:  # no year is shown; "Dez 30" seen in January means last year
        listed = listed.replace(year=today.year - 1)
    return listed.astimezone(UTC)


def parse_listing_date(raw: str | None, source: str, *, now: datetime | None = None) -> datetime | None:
    """Parse a source-specific listing-date string into a UTC ``datetime``.

    ``source`` is one of ``"api"`` (eBay Browse API — ISO-8601),
    ``"kleinanzeigen"`` (German local-time text), or ``"scraper"`` (eBay's
    result cards: a coarse relative age like "Vor 5 Std. eingestellt", or
    "Eingestellt am Sep 20" for older listings). Text
    that doesn't match the source's known format returns None — never a
    fabricated date. ``now`` exists for deterministic tests.
    """
    if not raw:
        return None
    now = now or datetime.now(UTC)
    if source == "kleinanzeigen":
        return _parse_kleinanzeigen_date(raw, now)
    if source == "scraper":
        return _parse_ebay_listing_age(raw, now)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# ── Canonical listing identity ──────────────────────────────────────────────
# The same eBay listing reaches us through different URL shapes (the Browse
# API's itemWebUrl vs. a search-results link with tracking parameters), so
# URL equality under-deduplicates. The numeric listing ID is stable.

_EBAY_ITEM_ID_RE = re.compile(r"/itm/(?:[^/?#]+/)?(\d{9,})")
_KLEINANZEIGEN_AD_ID_RE = re.compile(r"/s-anzeige/[^/]+/(\d+)-")


def canonical_listing_id(url: str | None) -> str | None:
    """Return ``"ebay:<id>"`` / ``"kleinanzeigen:<id>"`` for a listing URL, or None."""
    if not url:
        return None
    m = _EBAY_ITEM_ID_RE.search(url)
    if m and "ebay." in url:
        return f"ebay:{m.group(1)}"
    m = _KLEINANZEIGEN_AD_ID_RE.search(url)
    if m:
        return f"kleinanzeigen:{m.group(1)}"
    return None


def sort_key_for_deal(deal: dict) -> tuple[int, int, float]:
    """Sort key: "Must Have"/"Must Buy" first, then within each tier, deals
    with a known listing date sort newest-first, ahead of deals with no known
    date at all.

    The old sort collapsed every undated deal to ``datetime.min`` — which
    meant an undated-but-actually-newer deal always sank below every dated
    deal, even an old one, purely because its source (the HTML scraper, or
    unparsable Kleinanzeigen text) doesn't expose a date. Grouping "has a
    real date" ahead of "unknown date" within each tier fixes that without
    pretending to know a date nobody has.
    """
    rating = (deal.get("ai_deal_rating") or "").lower()
    not_must_have = int(rating not in ("must have", "must buy"))
    raw = deal.get("listing_date")
    if raw:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return (not_must_have, 0, -dt.timestamp())
        except ValueError:
            pass
    return (not_must_have, 1, 0.0)


class Deal(TypedDict, total=False):
    """Canonical deal schema. All fields optional — a source module may omit
    any field it has no data for; consumers should use ``.get(...)``.

    AI-assessment fields (``ai_deal_rating``, ``ai_verdict_summary``, etc.)
    are documented in ``ai_providers`` and merged into a deal dict after
    assessment — they are intentionally not listed here since this schema
    describes what a *search source* produces, not the AI layer's output.
    """

    title: str
    price: float
    condition: str
    condition_normalized: str
    seller_rating: float
    url: str
    listing_id: str  # "ebay:<id>" / "kleinanzeigen:<id>" — see canonical_listing_id
    shipping: str
    shipping_cost: float | None  # EUR; 0.0 = free, None = unknown / pickup only
    shipping_note: str
    is_trending: bool
    item_location: str
    description: str
    seller_count: str
    listing_date: str | None
    image_urls: list[str]
    image_issues: list[str]
    source: str
    listing_type: str  # "fixed" or "auction"
    auction_end: str | None  # ISO-8601 — when bidding ends; auctions only, None if unknown
    timestamp: float
    is_saved: bool
