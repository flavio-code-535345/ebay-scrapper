"""Query planning: turn what the user typed into as few, as broad requests as
each source can take.

eBay (web search and Browse API alike) supports OR groups — ``(a,b,c)`` — so
every German bundle synonym fits into ONE request, where the old design
fanned out up to 8 separate synonym variants per source (up to 24 requests a
search). Kleinanzeigen has no OR support and IP-bans quickly, so it gets at
most two plain variants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_providers.base import _PLATFORM_MAP
from ebay_api_client import MAX_QUERY_LENGTH

# Bundle vocabulary, most productive first — trimmed from the end when an
# OR group would push a query past the Browse API's 100-character cap.
# Compounds are listed separately: eBay matches whole words, so "sammlung"
# does not match a title that only says "Spielesammlung".
BUNDLE_TERMS: tuple[str, ...] = (
    "sammlung",
    "konvolut",
    "paket",
    "bundle",
    "spielesammlung",
    "lot",
    "spielepaket",
)
GAMES_TERMS: tuple[str, ...] = ("spiele", "games", "videospiele")
_BUNDLE_WORDS = frozenset(BUNDLE_TERMS) | {"set", "spielesammlung", "spielekonvolut", "spiele-sammlung"}
_GAMES_WORDS = frozenset(GAMES_TERMS) | {"titel", "spiel", "game"}

# Web-only exclusions (web search honors "-word"; the API documents no such
# syntax): toys-to-life, peripherals and karaoke that are never a game deal.
# Deliberately NOT fifa/kinect — mixed bundles containing one of those can
# still be worth it, and the pipeline's sports filter judges them instead.
WEB_EXCLUSIONS: tuple[str, ...] = ("skylanders", "amiibo", "lego", "disney", "singstar", "guitar", "rockband")

MAX_KLEINANZEIGEN_QUERIES = 2

_TOKEN_RE = re.compile(r"[\wäöüß]+(?:-[\wäöüß]+)*", re.IGNORECASE)
# Generic family names ("Xbox", "PlayStation") never conflict with a
# specific model of the same family.
_GENERIC_PLATFORMS = frozenset({"Microsoft Xbox", "Sony PlayStation"})


def detect_platform(text: str) -> str | None:
    """The most specific platform named in *text* (e.g. "Microsoft Xbox 360"), if any."""
    for pattern, name in _PLATFORM_MAP:
        if pattern.search(text):
            return name
    return None


def platforms_named(text: str) -> set[str]:
    """Every specific (non-generic) platform named in *text*."""
    return {name for pattern, name in _PLATFORM_MAP if name not in _GENERIC_PLATFORMS and pattern.search(text)}


@dataclass(frozen=True)
class SearchPlan:
    label: str
    core_terms: tuple[str, ...]
    platform: str | None
    bundle_intent: bool
    ebay_api: tuple[str, ...]
    ebay_web: tuple[str, ...]
    kleinanzeigen: tuple[str, ...]
    phrases: tuple[str, ...] = field(default_factory=tuple)


def _or_group(core: str, terms: tuple[str, ...], limit: int) -> str:
    """``core (t1,t2,...)``, dropping trailing terms until it fits *limit* chars."""
    for n in range(len(terms), 1, -1):
        query = f"{core} ({','.join(terms[:n])})".strip()
        if len(query) <= limit:
            return query
    return f"{core} {terms[0]}".strip()[:limit]


def _plan_phrase(phrase: str) -> tuple[list[str], bool, bool, str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(phrase)]
    bundle_intent = any(t in _BUNDLE_WORDS for t in tokens)
    games_intent = any(t in _GAMES_WORDS for t in tokens)
    core = [t for t in tokens if t not in _BUNDLE_WORDS and t not in _GAMES_WORDS]
    first_bundle_word = next((t for t in tokens if t in _BUNDLE_WORDS and t in BUNDLE_TERMS), "sammlung")
    return core, bundle_intent, games_intent, first_bundle_word


def plan_search(phrases: list[str]) -> SearchPlan:
    """Build one query per source from the user's phrase(s).

    A quick-search chip sends several near-identical phrases ("Xbox 360
    Spiele Sammlung Konvolut", "Xbox 360 Spielesammlung Konvolut", ...);
    they fold into the same plan instead of multiplying requests.
    """
    phrases = [p.strip() for p in phrases if p and p.strip()]
    if not phrases:
        raise ValueError("at least one non-empty search phrase is required")

    api_queries: list[str] = []
    web_queries: list[str] = []
    ka_queries: list[str] = []
    all_core: list[str] = []
    any_bundle = False
    for phrase in phrases:
        core, bundle_intent, games_intent, first_bundle_word = _plan_phrase(phrase)
        any_bundle = any_bundle or bundle_intent
        core_str = " ".join(core)
        for term in core:
            if term not in all_core:
                all_core.append(term)

        if bundle_intent:
            api_q = _or_group(core_str, BUNDLE_TERMS, MAX_QUERY_LENGTH)
            second_word = "sammlung" if first_bundle_word == "konvolut" else "konvolut"
            ka = [f"{core_str} {first_bundle_word}", f"{core_str} {second_word}"]
        elif games_intent:
            api_q = _or_group(core_str, GAMES_TERMS, MAX_QUERY_LENGTH)
            ka = [core_str or phrase.lower()]
        else:
            api_q = phrase.lower()[:MAX_QUERY_LENGTH]
            ka = [phrase.lower()]
        web_q = f"{api_q} " + " ".join(f"-{w}" for w in WEB_EXCLUSIONS)

        for q, bucket in ((api_q, api_queries), (web_q.strip(), web_queries)):
            if q and q not in bucket:
                bucket.append(q)
        for q in ka:
            q = " ".join(q.split())
            if q and q not in ka_queries:
                ka_queries.append(q)

    return SearchPlan(
        label=phrases[0],
        core_terms=tuple(all_core),
        platform=detect_platform(" ".join(phrases)),
        bundle_intent=any_bundle,
        # eBay results per request are generous (up to 200 via the API,
        # 120 via the web), so two distinct phrasings is plenty.
        ebay_api=tuple(api_queries[:2]),
        ebay_web=tuple(web_queries[:2]),
        kleinanzeigen=tuple(ka_queries[:MAX_KLEINANZEIGEN_QUERIES]),
        phrases=tuple(phrases),
    )
