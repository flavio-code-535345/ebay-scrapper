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
    text = raw.strip().lower()
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

_KLEINANZEIGEN_TODAY_RE = re.compile(r"^heute,?\s*(\d{1,2}):(\d{2})$", re.IGNORECASE)
_KLEINANZEIGEN_YESTERDAY_RE = re.compile(r"^gestern,?\s*(\d{1,2}):(\d{2})$", re.IGNORECASE)
_KLEINANZEIGEN_ABS_DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$")


def _parse_kleinanzeigen_date(raw: str) -> datetime | None:
    """Parse Kleinanzeigen's relative/absolute German date text.

    Handles the two formats the listing page actually shows: "Heute, 19:30" /
    "Gestern, 10:15" for recent ads, and "05.09.2025" for older ones. Anything
    else (a format Kleinanzeigen changed, or unparsable text) returns None
    rather than guessing.
    """
    text = raw.strip()
    m = _KLEINANZEIGEN_TODAY_RE.match(text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        return datetime.now(UTC).replace(hour=hour, minute=minute, second=0, microsecond=0)
    m = _KLEINANZEIGEN_YESTERDAY_RE.match(text)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        yesterday = datetime.now(UTC) - timedelta(days=1)
        return yesterday.replace(hour=hour, minute=minute, second=0, microsecond=0)
    m = _KLEINANZEIGEN_ABS_DATE_RE.match(text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day, tzinfo=UTC)
        except ValueError:
            return None
    return None


def parse_listing_date(raw: str | None, source: str) -> datetime | None:
    """Parse a source-specific listing-date string into a UTC ``datetime``.

    ``source`` is one of ``"api"`` (eBay Browse API — already ISO-8601),
    ``"kleinanzeigen"`` (relative/absolute German text), or ``"scraper"``
    (the HTML scraper exposes no listing date on its search-results page at
    all — always returns None for it, never a fabricated value).
    """
    if not raw:
        return None
    if source == "kleinanzeigen":
        return _parse_kleinanzeigen_date(raw)
    if source == "scraper":
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
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
    shipping: str
    shipping_note: str
    is_trending: bool
    item_location: str
    description: str
    seller_count: str
    listing_date: str | None
    image_urls: list[str]
    image_issues: list[str]
    source: str
    listing_type: str
    timestamp: float
    is_saved: bool
