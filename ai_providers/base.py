"""Base assessor — shared AI-agnostic logic, deterministic rules, eBay price helpers."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

with open(os.path.join(_PROMPT_DIR, "system_prompt.txt"), encoding="utf-8") as _f:
    _SYSTEM_PROMPT = _f.read()

with open(os.path.join(_PROMPT_DIR, "batch_system_prompt.txt"), encoding="utf-8") as _f:
    _BATCH_SYSTEM_PROMPT = _f.read()

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

_MAX_IMAGES = 3
_IMAGE_FETCH_TIMEOUT = 5
_DEFAULT_BACKOFF_SECONDS = 60
_BATCH_SIZE = 5
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0
_ASSESS_TOTAL_BUDGET_S = 145

# Shared rate-limit state (module-level so all assessors share the same
# gate when the user switches providers without restarting).
_rate_limit_lock = threading.Lock()
_rate_limited_until: float = 0.0


def _set_rate_limited_until(until: float) -> None:
    """Update the shared rate-limit clock (thread-safe)."""
    global _rate_limited_until
    with _rate_limit_lock:
        _rate_limited_until = until


_JSON_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── JSON / text helpers ───────────────────────────────────────────────────


def _sanitize_json_text(text: str) -> str:
    """Strip control characters that trip up ``json.loads``."""
    return _JSON_CONTROL_CHAR_RE.sub("", text)


def _extract_json_objects(text: str) -> list:
    """Try to find a JSON array or object in *text* via best-effort heuristics.

    1. Trim leading/trailing whitespace.
    2. If it starts with ``[`` try to parse the whole thing as a JSON array.
    3. Otherwise try to find a `````json`` fence and extract from there.
    4. Fall back to searching for ``[`` … ``]`` boundaries.
    """
    text = text.strip()

    # Direct parse attempt.
    if text.startswith("["):
        try:
            return json.loads(_sanitize_json_text(text))
        except json.JSONDecodeError:
            pass

    # Markdown code fence with json language tag.
    m = re.search(r"```json\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("["):
            try:
                return json.loads(_sanitize_json_text(candidate))
            except json.JSONDecodeError:
                pass
        elif candidate.startswith("{"):
            try:
                return [json.loads(_sanitize_json_text(candidate))]
            except json.JSONDecodeError:
                pass

    # Generic code fence.
    m = re.search(r"```\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("["):
            try:
                return json.loads(_sanitize_json_text(candidate))
            except json.JSONDecodeError:
                pass
        elif candidate.startswith("{"):
            try:
                return [json.loads(_sanitize_json_text(candidate))]
            except json.JSONDecodeError:
                pass

    # Fallback: find the outermost [ … ] bracket pair.
    start = text.find("[")
    if start == -1:
        return []
    depth = 0
    end = -1
    for i, ch in enumerate(text):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return []
    try:
        return json.loads(_sanitize_json_text(text[start:end]))
    except json.JSONDecodeError:
        pass
    return []


# ── Game-title helpers ────────────────────────────────────────────────────


_AGGREGATE_PLACEHOLDER_RE = re.compile(
    r"^(additional|remaining|other|more|weitere|restliche|sonstige)"
    r"[\s\-]*(titles?|games?|spiele?|titel|items?)"
    r"|^rest\s+(of\s+)?(titles?|games?|spiele?|titel|items?)",
    re.IGNORECASE,
)
_AGGREGATE_PLACEHOLDER_TOKENS = frozenset({"etc.", "etc", "...", "u.a.", "usw.", "and more", "und mehr"})


def _is_aggregate_placeholder(game_name: str) -> bool:
    """Return True if *game_name* is an aggregate/grouping placeholder string."""
    if not isinstance(game_name, str):
        return False
    name_lower = game_name.strip().lower()
    if _AGGREGATE_PLACEHOLDER_RE.match(name_lower):
        return True
    return name_lower in _AGGREGATE_PLACEHOLDER_TOKENS


# ── Error classification ──────────────────────────────────────────────────


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate_limit" in msg


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("timeout", "connection", "reset", "unavailable", "503", "500"))


def _parse_retry_delay(exc: Exception) -> float | None:
    """Try to extract retryDelay from the error payload."""
    m = re.search(r'retry_delay["\']?\s*:\s*["\']?(\d+\.?\d*)', str(exc))
    if m:
        return float(m.group(1))
    m = re.search(r"retry\s+after\s+(\d+)", str(exc), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ── Platform & title helpers ──────────────────────────────────────────────


# Ordered from most specific to least specific.
_PLATFORM_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bxbox\s*360\b", re.IGNORECASE), "Microsoft Xbox 360"),
    (re.compile(r"\bxbox\s*one\b", re.IGNORECASE), "Microsoft Xbox One"),
    (re.compile(r"\bxbox\s*series\b", re.IGNORECASE), "Microsoft Xbox Series"),
    (re.compile(r"\bxbox\b", re.IGNORECASE), "Microsoft Xbox"),
    (re.compile(r"\bps\s*5\b|\bplaystation\s*5\b", re.IGNORECASE), "Sony PlayStation 5"),
    (re.compile(r"\bps\s*4\b|\bplaystation\s*4\b", re.IGNORECASE), "Sony PlayStation 4"),
    (re.compile(r"\bps\s*3\b|\bplaystation\s*3\b", re.IGNORECASE), "Sony PlayStation 3"),
    (re.compile(r"\bps\s*2\b|\bplaystation\s*2\b", re.IGNORECASE), "Sony PlayStation 2"),
    (re.compile(r"\bps\s*vita\b|\bpsvita\b", re.IGNORECASE), "Sony PS Vita"),
    (re.compile(r"\bpsp\b", re.IGNORECASE), "Sony PSP"),
    (re.compile(r"\bplaystation\b", re.IGNORECASE), "Sony PlayStation"),
    (re.compile(r"\bnintendo\s*switch\b", re.IGNORECASE), "Nintendo Switch"),
    (re.compile(r"\bwii\s*u\b", re.IGNORECASE), "Nintendo Wii U"),
    (re.compile(r"\bwii\b", re.IGNORECASE), "Nintendo Wii"),
    (re.compile(r"\bn64\b|\bnintendo\s*64\b", re.IGNORECASE), "Nintendo 64"),
    (re.compile(r"\bsnes\b|\bsuper\s*nintendo\b", re.IGNORECASE), "Super Nintendo"),
    (re.compile(r"\bnes\b|\bnintendo\s*entertainment\b", re.IGNORECASE), "Nintendo Entertainment System"),
    (re.compile(r"\bgba\b|\bgame\s*boy\s*advance\b", re.IGNORECASE), "Game Boy Advance"),
    (re.compile(r"\b3ds\b", re.IGNORECASE), "Nintendo 3DS"),
    (re.compile(r"\bnds\b|\bnintendo\s*ds\b|\bnintendogs\b", re.IGNORECASE), "Nintendo DS"),
]

# Common words that indicate condition or bundling, not game titles.
_SINGLE_GAME_NOISE_RE = re.compile(
    r"\b(neu|ovp|sealed|version|edition|complete|included|mit|ohne|"
    r"plus|exklusive|inkl|spiele|spiel|game|games|konsole|zubehör|"
    r"zubehoer|controller|kabel|netzteil|anleitung|verpackung|"
    r"originalverpackung|gebraucht|sehr\s*gut|gut|akzeptabel|"
    r"defekt|neuwertig|wie\s*neu)\b",
    re.IGNORECASE,
)

# Bundle-specific noise: words that appear in bundle listings but are NOT game titles.
_BUNDLE_TITLE_KEYWORDS_RE = re.compile(
    r"\b(spielesammlung|spielepaket|spieleset|spiele[- ]set|spiele[- ]paket"
    r"|sammlung|konvolut|paket|lot|bundle|collection|spielekonvolut"
    r"|spiele[- ]sammlung|spiele[- ]konvolut)\b",
    re.IGNORECASE,
)

_NON_TITLE_WORDS_RE = re.compile(
    r"^\s*(\d+|spiele?|games?|stück|pieces?|neu|used|gebraucht|like\s+new"
    r"|nintendo|playstation|ps[1-5]|xbox|sega|atari|pc|psp|ds|3ds|wii"
    r"|switch|gamecube|gameboy|game\s+boy|mega\s+drive"
    r"|sehr\s+gut|gut|akzeptabel|neuwertig|top|set|bundle"
    r"|sammlung|konvolut|paket|lot|collection|inklusive?|inkl|mit|und|and"
    r"|plus|\+|für|fuer|for|the|der|die|das|ein|eine)\s*$",
    re.IGNORECASE,
)

_TITLE_SEPARATOR_RE = re.compile(r"\s*[+;,/&\n•·–—|]\s*")
_QUANTITY_PREFIX_RE = re.compile(r"^\d+\s*x\s*", re.IGNORECASE)

_BUNDLE_PART_NOISE_RE = re.compile(
    r"\b(komplett|complete|ovp|cib|sealed|ungetestet|defekt|gebraucht"
    r"|neuwertig|wie\s+neu|like\s+new|sehr\s+gut|top\s+zustand"
    r"|pal|ntsc|deutsch|german)\b",
    re.IGNORECASE,
)

_MAX_GAMES_PER_BUNDLE = 8


def _extract_platform_name(title: str) -> str:
    """Return the console/platform name from a listing title, or empty string."""
    for pattern, name in _PLATFORM_MAP:
        if pattern.search(title):
            return name
    return ""


def _build_single_game_search_query(title: str) -> str:
    """Build an eBay search query for a single-game listing.

    Strips platform keywords and condition/noise words from *title*, then
    appends the canonical platform name in the required format::

        "GAME NAME (PLATFORM NAME)"

    If no platform can be detected the cleaned title is returned as-is.
    """
    platform = _extract_platform_name(title)
    cleaned = title.strip()
    for pattern, _ in _PLATFORM_MAP:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = _SINGLE_GAME_NOISE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n:()-")
    if len(cleaned) < 3:
        cleaned = title.strip()
    if platform:
        return f"{cleaned} ({platform})"
    return cleaned


def _extract_potential_game_titles(title: str) -> list[str]:
    """Extract individual game titles from a bundle listing title."""
    if not title:
        return []
    # Strip quantity patterns ("10 Spiele", "5 Games") from the start.
    cleaned = re.sub(r"^\d+\s+(spiele?|games?)\s*", "", title.strip(), flags=re.IGNORECASE)
    # Remove platform names via _PLATFORM_MAP (most specific first).
    for _pat, _ in _PLATFORM_MAP:
        cleaned = _pat.sub(" ", cleaned)
    # Remove any remaining standalone platform/manufacturer keywords and
    # bundle/collection keywords that weren't caught by compound patterns.
    cleaned = re.sub(
        r"\b(microsoft|nintendo|sony|sega|atari|pc"
        r"|switch|wii|gameboy|gamecube|n64|snes|nes|psp|vita|3ds|nds|gba"
        r"|bundle|lot|paket|sammlung|konvolut|spielesammlung|spielepaket"
        r"|spieleset|spiele[- ]set|spiele[- ]paket|collection)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    separators = re.split(r"[,+;/\n•·–—|&]|(?<!\d)\s*x\s(?!\d)", cleaned)
    candidates: list[str] = []
    for part in separators:
        part = part.strip(" \t\n:()-")
        if not part:
            continue
        part = _QUANTITY_PREFIX_RE.sub("", part).strip()
        if not part:
            continue
        part = _BUNDLE_PART_NOISE_RE.sub(" ", part)
        # Strip platform tokens from the part
        for _pat, _ in _PLATFORM_MAP:
            part = _pat.sub(" ", part)
        part = re.sub(r"\s+", " ", part).strip()
        if not part or len(part) < 3:
            continue
        if _NON_TITLE_WORDS_RE.match(part):
            continue
        if _is_aggregate_placeholder(part):
            continue
        if re.match(r"^[\d.,€\s]+$", part):
            continue
        candidates.append(part)
    seen: set = set()
    unique: list[str] = []
    for c in candidates:
        normalized = c.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(c)
    return unique[:_MAX_GAMES_PER_BUNDLE]


# ── Garbage & trash detection (deterministic) ────────────────────────────
# These are common-sense rules that identify listings with near-zero or
# negative resale value, overriding any AI-generated rating.
#
# Three tiers:
#   Tier 0 — Scam: bait-and-switch, per-piece pricing, fake bundles
#   Tier 1 — Garbage: broken, untested, empty cases, demos, shovelware
#   Tier 2 — Avoid:  sports/Kinect bundles, low-demand categories, suspicious pricing


# ── Tier 1: Trash title keywords (zero-value garbage) ──────────────────────

_TRASH_TITLE_KEYWORDS_RE = re.compile(
    r"\b("
    # Demo discs / promotional items (worthless)
    r"demo\s+disc|demo\s+volume|playstation\s+underground"
    r"|official\s+xbox\s+magazine"
    r"|playstation\s+magazine"
    # Empty cases / manuals only / no game
    r"|empty\s+case|leerhülle|ohne\s+(spiel|disk|disc|game)"
    r"|nur\s+(hülle|case|anleitung|manual)"
    # Common low-value shovelware & kids licensed trash
    r"|imagine\s+(babys|fashion|teacher|doctor|mermaid)"
    r"|carnival\s+(games|king)|family\s+feast"
    r"|party\s+(superstars|megamix|mania)"
    r"|game\s+party"
    r"|rabbids|ray(mans?\s*)?rabbids"
    r"|barbie|hannah\s+montana|high\s+school\s+musical"
    r"|disney\s+(channel|infinity|lions|princess|sing(it)?)"
    r"|hello\s+kitty|my\s+little\s+pony|spongebob"
    r"|dora\s+(the\s+)?explorer"
    r"|skylanders|disney\s+infinity|amiibo"
    r"|lego\s+(movie|batman|pirates|rock\s+band|city|friends)(\s+game)?"
    r"|nintendogs|nintendocats"
    r"|big\s+game\s+hunter|cabela"
    r"|fitness\s+(evolution|academy|game|circuit)"
    r"|your\s+shape|ea\s+sports\s+active"
    r"|zumba\s+(fitness|party)"
    r")\b",
    re.IGNORECASE,
)


# ── Broken/defective/untested detection (negative-value garbage) ───────────

_BROKEN_KEYWORDS_RE = re.compile(
    r"\b("
    r"defekt|kaputt|beschädigt|schaden|reparatur"
    r"|for\s+parts|as\s*[-]\s*is|not\s+working|untested"
    r"|ungetestet|ohne\s+gewähr|ohne\s+garantie"
    r"|verkaufe\s+ohne|keine\s+rücknahme"
    r"|bastler|nicht\s+getestet"
    r"|fehlerhaft|zerstört|nur\s+als\s+ersatzteil"
    r")\b",
    re.IGNORECASE,
)


# ── Tier 2: Sports & Kinect detection (low-value, but not quite garbage) ──

_SPORTS_KINECT_KEYWORDS_RE = re.compile(
    r"\b("
    r"kinect"
    r"|fifa"
    r"|topspin|top[\s\-]spin"
    r"|forza"
    r"|nba\s*2k|nba\s*live|nba\b"
    r"|nhl\b"
    r"|madden"
    r"|pes\b|pro\s+evolution\s+soccer"
    r"|wwe\b"
    r"|ufc\b"
    r"|motogp"
    r"|tour\s+de\s+france"
    r"|just\s+dance"
    r"|dance\s+central"
    r"|wii\s+(sports|play|party|music)"
    r"|wrc\b|f1\s+\d{4}"
    r"|wii\s+fit(\s+plus|\s+board)?"
    r"|singstar|rock\s+band|guitar\s+hero"
    r")\b",
    re.IGNORECASE,
)

_SPORTS_KINECT_AVOID_PREFIX = (
    "⛔ **SPORTS/KINECT — AVOID**: This listing contains sports or "
    "Kinect game titles (FIFA, Forza, TopSpin, Kinect, etc.) that "
    "have minimal resale value in the current German eBay market. "
    "These titles rarely generate profit and are best avoided unless "
    "the bundle also contains clearly high-value non-sports games."
)

_GARBAGE_VERDICT_PREFIX = (
    "🗑️ **GARBAGE — AVOID**: This listing has near-zero or negative resale "
    "value. These items are effectively worthless on the German secondhand "
    "market – do not purchase even at a low price, as you will not be able "
    "to resell them."
)


# ── Detection functions ────────────────────────────────────────────────────


def _detect_broken_deal(deal: dict) -> str | None:
    """Check for defective/broken/untested listings.

    Returns a warning string, or ``None`` if the deal passes.
    """
    title = (deal.get("title") or "").strip()
    description = (deal.get("description") or "").strip()
    match_title = _BROKEN_KEYWORDS_RE.search(title) if title else None
    match_desc = _BROKEN_KEYWORDS_RE.search(description) if description else None
    if not match_title and not match_desc:
        return None
    short_title = title[:80] + ("..." if len(title) > 80 else "")
    return (
        f"DEFECTIVE/UNTESTED: Item '{short_title}' is listed as defective, "
        f"untested, or for parts. Such items have zero or negative resale "
        f"value (cost of disposal). AVOID."
    )


def _detect_trash_title(deal: dict) -> str | None:
    """Check for title keywords that indicate worthless/garbage items.

    Returns a warning string, or ``None`` if the deal passes.
    """
    title = (deal.get("title") or "").strip()
    if not title:
        return None
    match = _TRASH_TITLE_KEYWORDS_RE.search(title)
    if not match:
        return None
    keyword = match.group(0)
    short_title = title[:80] + ("..." if len(title) > 80 else "")
    return (
        f"TRASH CONTENT DETECTED: Title '{short_title}' contains "
        f"low/zero-value keyword '{keyword}'. This item has near-zero "
        f"resale value on the German eBay market and should be avoided."
    )


def _detect_sports_kinect_deal(deal: dict) -> str | None:
    """Deterministic check for sports-franchise or Kinect-themed listings.

    Returns a warning string, or ``None`` if the deal passes.
    """
    title = (deal.get("title") or "").strip()
    if not title:
        return None
    match = _SPORTS_KINECT_KEYWORDS_RE.search(title)
    if not match:
        return None
    keyword = match.group(0)
    short_title = title[:80] + ("..." if len(title) > 80 else "")
    return (
        f"SPORTS/KINECT CONTENT DETECTED: Title '{short_title}' contains "
        f"sports or Kinect keyword '{keyword}'. Sports game franchises "
        f"(FIFA, Forza, TopSpin, etc.) and Kinect titles have very low "
        f"resale value in the German eBay market and rarely generate "
        f"meaningful profit."
    )


# ── Override functions ──────────────────────────────────────────────────────


def _apply_garbage_overrides(deal: dict, assessment: dict) -> dict:
    """Apply deterministic garbage/trash overrides (Tier 1).

    Checks for broken/defective items and trash title keywords first,
    before sports/Kinect overrides. Sets rating to "Garbage".

    Always returns *assessment* (mutated in-place if overridden).
    """
    broken_warning = _detect_broken_deal(deal)
    if broken_warning:
        assessment["ai_deal_rating"] = "Garbage"
        _existing_flags = assessment.get("ai_red_flags")
        if not isinstance(_existing_flags, list):
            _existing_flags = []
        if "Defective/untested: no resale value" not in _existing_flags:
            assessment["ai_red_flags"] = _existing_flags + ["Defective/untested: no resale value"]
        _existing_summary = assessment.get("ai_verdict_summary", "")
        assessment["ai_verdict_summary"] = (
            f"{_GARBAGE_VERDICT_PREFIX}\n\n{_existing_summary}" if _existing_summary else _GARBAGE_VERDICT_PREFIX
        )
        return assessment

    trash_warning = _detect_trash_title(deal)
    if trash_warning:
        assessment["ai_deal_rating"] = "Garbage"
        _existing_flags = assessment.get("ai_red_flags")
        if not isinstance(_existing_flags, list):
            _existing_flags = []
        if "Low/zero-value content: no resale demand" not in _existing_flags:
            assessment["ai_red_flags"] = _existing_flags + ["Low/zero-value content: no resale demand"]
        _existing_summary = assessment.get("ai_verdict_summary", "")
        assessment["ai_verdict_summary"] = (
            f"{_GARBAGE_VERDICT_PREFIX}\n\n{_existing_summary}" if _existing_summary else _GARBAGE_VERDICT_PREFIX
        )
        return assessment

    return assessment


def _apply_sports_kinect_override(deal: dict, assessment: dict) -> dict:
    """Apply a deterministic 'Avoid' override for sports/Kinect themed deals (Tier 2).

    Always returns *assessment* (mutated in-place if overridden).
    """
    warning = _detect_sports_kinect_deal(deal)
    if warning is None:
        return assessment
    assessment["ai_deal_rating"] = "Avoid"
    existing_flags = assessment.get("ai_red_flags")
    if not isinstance(existing_flags, list):
        existing_flags = []
    if "Sports/Kinect content: low resale value" not in existing_flags:
        assessment["ai_red_flags"] = existing_flags + ["Sports/Kinect content: low resale value"]
    existing_summary = assessment.get("ai_verdict_summary", "")
    if existing_summary:
        assessment["ai_verdict_summary"] = f"{_SPORTS_KINECT_AVOID_PREFIX}\n\n{existing_summary}"
    else:
        assessment["ai_verdict_summary"] = _SPORTS_KINECT_AVOID_PREFIX
    return assessment


# ── Listings whose price doesn't buy the bundle they show (deterministic) ──
#
# The commonest way a listing looks like a steal and isn't: a bundle title and
# photo, but the price buys ONE game. Sellers do say so — often only deep in
# the description ("Stück preis 7euro VB oder komplett paket 120 euro inkl
# versand"), which is why the search pipeline fetches full Kleinanzeigen
# descriptions for bundles that look too cheap. The signals:
#   - per-piece pricing: "Stückpreis", "Einzelpreis", "5 € pro Spiel", "je 3 €"
#   - prices vary per game: "Spiele ab 2 €", "Preise je nach Spiel", "Preisliste"
#   - the buyer picks one: "Spiel nach Wahl", "welches Spiel möchtest du"
#   - sold one by one: "werden einzeln verkauft", "Einzelverkauf"
#   - many units of one "bundle" listing (the lazy Stückzahl trick, further down)
# When the text also states what the whole lot costs, that is the bundle's
# real price (PriceScope.lot_price) and the pipeline re-prices the deal with it.
#
# Every text signal is ignored when negated ("nicht einzeln", "kein
# Stückpreis") or when it is about shipping ("Versand pro Stück 1 €").

# "7euro", "120 €", "5,50€", "€ 15", "15,-", "1.200 €"
_AMOUNT = r"(?:\d{1,3}(?:\.\d{3})+|\d{1,5})(?:[.,]\d{1,2})?"
_MONEY = rf"(?:€\s*{_AMOUNT}|{_AMOUNT}\s*(?:€|euros?\b|eur\b|,-))"
_MONEY_RE = re.compile(rf"€\s*({_AMOUNT})|({_AMOUNT})\s*(?:€|euros?\b|eur\b|,-)", re.IGNORECASE)
_PIECE = r"(?:st(?:ü|ue)ck|stk\.?|spiel|game|titel|teil|exemplar|disc|artikel)"

_PER_PIECE_RE = re.compile(
    r"\b(?:st(?:ü|ue)ck|stk)\.?\s*-?\s*preis(?:e)?\b"  # Stückpreis, Stück preis, Stk.-Preis
    r"|\beinzel\s*-?\s*preis(?:e)?\b"  # Einzelpreis
    rf"|\b(?:preis\s*)?(?:pro|je|per)\s+{_PIECE}(?!\w)|\bpreis\s*/\s*{_PIECE}(?!\w)"  # pro Stück, Preis/Stück
    r"|\bpreis\s+(?:gilt\s+|ist\s+|bezieht\s+sich\s+)?(?:nur\s+)?(?:für|auf)\s+(?:ein(?:e[ns]?)?|1|jedes)\s+"
    r"(?:einzelne[sn]?\s+)?(?:spiel|stück|titel|game)\b"  # Preis gilt für ein Spiel
    r"|\bprice\s+(?:per|for\s+each|each)\b|\bper\s+(?:piece|item|game)\b|\beach\s+game\b"
    rf"|{_MONEY}\s*(?:das\s+|/\s*){_PIECE}(?!\w)"  # 5 € das Stück, 5€/Stk
    rf"|\b(?:je|jeweils|à)\s*{_MONEY}|{_MONEY}\s*jeweils\b",  # je 5 €, à 5€, 5 € jeweils
    re.IGNORECASE,
)
_PRICE_VARIES_RE = re.compile(
    rf"\bab\s*{_MONEY}"  # Spiele ab 2 €
    r"|\bpreis(?:e)?\s+(?:je\s+nach|variier\w*|unterschiedlich\w*|siehe|auf\s+anfrage|in\s+der\s+beschreibung"
    r"|stehen\s+(?:bei|neben|in|auf|unter))"
    r"|\bpreisliste\b|\bverschiedene\s+preise\b|\bpreise\s+einzeln\b",
    re.IGNORECASE,
)
_BUYER_PICKS_RE = re.compile(
    r"\b(?:ihrer|deiner|eurer|nach|zur)\s+wahl\b|\bwunsch(?:spiel|titel)\w*"
    r"|\bspiel\w*\s+(?:bitte\s+)?(?:aus)?(?:suchen|wählen)\b"  # Spiel aussuchen
    r"|\b(?:sie|du|ihr)\s+(?:können|kannst|könnt|dürfen|darfst)\s+(?:sich\s+|dir\s+|euch\s+)?"
    r"(?:ein|1|eines|einen)\s+(?:spiel|titel|game)\w*\s+(?:aus)?(?:suchen|wählen)"
    r"|\b(?:such|wähl)\w*\s+(?:dir|euch|sich)\s+(?:ein|1|eines|einen)\s+(?:spiel|titel|game)"  # such dir ein Spiel aus
    r"|\bwelche[sn]?\s+(?:spiel|titel|game)\w*\s+(?:(?:du|sie|ihr)\s+(?:möcht|will|wollt|woll|hab|brauch)"
    r"|(?:möcht|will|wollt|woll|hätt|brauch)\w*\s+(?:du|sie|ihr)\b)"  # welches Spiel möchtest du
    r"|\bgewünschte[sn]?\s+(?:spiel|titel|game)|\bauswahl\s+treffen\b"
    r"|\b1\s+(?:spiel|stück|titel)\s+(?:nach\s+wahl|wählen|auswählen|aussuchen)|\b1\s+aus\s+\d+"
    r"|\bbitte\s+(?:gewünschte\w*\s+)?(?:variante|spiel|titel)?\s*auswählen\b"
    r"|\byou\s+pick\b|\b(?:choose|pick)\s+(?:1|one|your)\b",
    re.IGNORECASE,
)
# Only in a title: "PS4 Spiele Auswahl", "nur 1 Spiel", "1 aus" — in a
# description "eine Auswahl an Spielen" / "nur ein Spiel hat Kratzer" are innocent.
_TITLE_ONLY_PICKS_RE = re.compile(r"\bauswahl\b|\bnur\s+(?:ein|1)\s+spiel\b|\b1\s+aus\b", re.IGNORECASE)
_SOLD_SINGLY_RE = re.compile(
    r"\beinzelverk(?:auf|äufe)\b|\beinzeln\s+(?:zu\s+)?(?:verkauf\w*|abzugeben|erhältlich|kaufbar|zu\s+haben)"
    r"|\bnur\s+einzeln\b|\b(?:spiele|titel)\s+(?:werden\s+|sind\s+)?einzeln\b",
    re.IGNORECASE,
)
# "Einzelverkauf oder komplett", "auch einzeln", "Einzelverkauf möglich" offer
# both — the listed price may well be the lot's.
_BOTH_OPTIONS_RE = re.compile(r"\b(?:oder|auch|möglich|komplett\w*|zusammen|gesamt\w*)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"nicht|kein\w*|ohne", re.IGNORECASE)
_SHIPPING_WORD_RE = re.compile(
    r"versand|porto|lieferung|verpackung|verschick|päckchen|warensendung|\bdhl\b|\bhermes\b|\bdpd\b|\bgls\b",
    re.IGNORECASE,
)
_SHIPPING_NEXT_WORD_RE = re.compile(r"\s*(?:versand|porto|lieferung)", re.IGNORECASE)
# A sentence, or a comma-separated part of one (decimal commas are not breaks).
_CLAUSE_BREAK_RE = re.compile(r"[.!?;\n]\s|\n|(?<!\d),|,(?!\d)")

# What the whole lot costs, when the text says: "komplett paket 120 euro",
# "Gesamtpreis 120€", "alle 16 Spiele für 120 €", "120 € für alle".
_LOT_PRICE_RE = re.compile(
    r"\b(?:komplett\w*|gesamt\w*|pauschal\w*|insgesamt|zusammen|alle(?:s)?|paket\s*-?\s*preis"
    r"|bundle\s*-?\s*preis|im\s+paket|als\s+(?:paket|set|konvolut))"
    r"(?:\s+(?:paket|preis|konvolut|set|sammlung|zusammen|\d{1,3}\s+spiele\w*|spiele\w*|für|zu|zum\s+preis\s+von"
    rf"|nur|abzugeben|verkauf\w*))*\s*[:=]?\s*{_MONEY}"
    rf"|{_MONEY}\s*(?:für\s+)?(?:alle(?:s)?\b|komplett\b|zusammen\b|(?:das|den|die)\s+(?:ganze|gesamte|komplette)\w*)",
    re.IGNORECASE,
)
_LOT_INCLUDES_SHIPPING_RE = re.compile(
    r"^\W{0,3}(?:vb\W+)?(?:inkl\w*\.?|incl\w*\.?|mit)\s*(?:versand|porto|vers\b)|^\W{0,3}(?:versandkostenfrei|portofrei)",
    re.IGNORECASE,
)
_MONEY_BEFORE_RE = re.compile(rf"{_MONEY}\s*$", re.IGNORECASE)
_PER_PIECE_NEXT_RE = re.compile(rf"^\s*(?:pro|je|das|/)\s*{_PIECE}(?!\w)|^\s*jeweils\b", re.IGNORECASE)

# Titles that describe more than one item — per-piece wording on a single
# controller or game ("Preis pro Stück, 3 vorhanden") is perfectly honest.
_MULTI_ITEM_TITLE_RE = re.compile(
    r"\b(?:spiele|spielen|games|videospiele|titel)\b|\b\d{1,3}\s*(?:x\b|stück\b|stk\b)|\bx\s*\d{1,3}\b",
    re.IGNORECASE,
)

# One game dressed up as a bundle for search: "Battlefield 1 PS4 Spiel
# Sammlung PS2 PS3 PS5 Konvolut Bundle Top", "Dragon Ball PS4 Spiel aus Sammlung".
_SINGLE_GAME_TITLE_RE = re.compile(
    r"\bspiel\s+(?:aus\s+(?:der\s+|meiner\s+|einer\s+)?)?(?:sammlung|konvolut|bundle|paket|lot|collection)\b",
    re.IGNORECASE,
)
_SINGLE_GAME_DESC_RE = re.compile(
    r"\b(?:verkaufe|biete)\s+(?:ich\s+)?(?:hier\s+)?(?:das|dieses|diesen|den|mein|meinen|ein|einen)\s+"
    r"(?:(?:top|tolle[ns]?|super|seltene[ns]?)\s+)?(?:spiel|game|titel)\b",  # "Biete hier diesen Top Titel an"
    re.IGNORECASE,
)
_MULTI_GAME_DESC_RE = re.compile(
    r"\b\d{1,3}\s+(?:\w+\s+){0,3}(?:spiele|spielen|games|titel)\b|\bsammlung\s+(?:von|aus|mit)\s+\d", re.IGNORECASE
)


def _parse_money(text: str) -> float | None:
    m = _MONEY_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?", raw):
        raw = raw.replace(".", "")
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return None
    return value if value > 0 else None


def _clause_before(text: str, start: int, span: int = 40) -> str:
    """The part of *text*'s current clause that precedes *start* (at most *span* chars)."""
    before = text[max(0, start - span) : start]
    breaks = list(_CLAUSE_BREAK_RE.finditer(before))
    return before[breaks[-1].end() :] if breaks else before


def _is_about_shipping_or_negated(text: str, m: re.Match) -> bool:
    """ "nicht einzeln", "kein Stückpreis"; "Versand pro Stück", "je 1,50 € Versand"."""
    words = _clause_before(text, m.start()).split()
    if any(_NEGATION_RE.fullmatch(w) for w in words[-3:]):
        return True
    if "preis" in m.group(0).lower():
        return False  # "Stückpreis", "Einzelpreis" are never about shipping
    return bool(_SHIPPING_WORD_RE.search(" ".join(words[-4:])) or _SHIPPING_NEXT_WORD_RE.match(text, m.end()))


def _nearby_amount(text: str, m: re.Match) -> float | None:
    """The per-piece amount stated with a per-piece phrase: in it ("je 5 €"),
    right before it ("7 € pro Stück") or right after it ("Stückpreis 7euro")."""
    before = _MONEY_BEFORE_RE.search(_clause_before(text, m.start(), span=15))
    return (
        _parse_money(m.group(0))
        or (_parse_money(before.group(0)) if before else None)
        or _parse_money(text[m.end() : m.end() + 15].split("\n")[0])
    )


def _find_price_signal(text: str, *, in_title: bool = False) -> tuple[str, str, float | None] | None:
    """First sign in *text* that the price is not for the whole lot → (kind, evidence, per-piece amount)."""
    if not text:
        return None
    checks = [("per_item", _PER_PIECE_RE), ("price_varies", _PRICE_VARIES_RE), ("buyer_picks", _BUYER_PICKS_RE)]
    if in_title:
        checks.append(("buyer_picks", _TITLE_ONLY_PICKS_RE))
    for kind, pattern in checks:
        for m in pattern.finditer(text):
            if _is_about_shipping_or_negated(text, m):
                continue
            amount = _nearby_amount(text, m) if kind == "per_item" else None
            return kind, m.group(0).strip(), amount
    for m in _SOLD_SINGLY_RE.finditer(text):
        clause_end = _CLAUSE_BREAK_RE.search(text, m.end())
        clause = _clause_before(text, m.start(), span=60) + text[m.start() : clause_end.start() if clause_end else None]
        if _is_about_shipping_or_negated(text, m) or _BOTH_OPTIONS_RE.search(clause):
            continue
        return "sold_singly", m.group(0).strip(), None
    return None


def _find_lot_price(text: str) -> tuple[float | None, bool, bool]:
    """Whole-lot price stated in *text* → (amount, includes shipping, negotiable)."""
    for m in _LOT_PRICE_RE.finditer(text or ""):
        after = text[m.end() : m.end() + 30]
        if _PER_PIECE_NEXT_RE.match(after) or _SHIPPING_WORD_RE.search(_clause_before(text, m.start())):
            continue  # "alle 20 Spiele 5 € pro Stück", "Versand zusammen 5 €"
        amount = _parse_money(m.group(0))
        if amount:
            is_vb = bool(re.match(r"^\W{0,3}vb\b", after, re.IGNORECASE))
            return amount, bool(_LOT_INCLUDES_SHIPPING_RE.search(after)), is_vb
    return None, False, False


def looks_like_multi_item(title: str) -> bool:
    """Does *title* offer more than one item (a bundle, "Spiele", "10x", "5 Stück")?"""
    return bool(_BUNDLE_TITLE_KEYWORDS_RE.search(title or "") or _MULTI_ITEM_TITLE_RE.search(title or ""))


def is_single_game_listing(deal: dict) -> bool:
    """One game whose title is padded with bundle words for search — "Battlefield
    1 PS4 Spiel Sammlung PS2 PS3 PS5 Konvolut Bundle" or "… Spiel aus Sammlung" —
    or whose description says "Ich verkaufe das Spiel …"."""
    title = deal.get("title") or ""
    if _MULTI_ITEM_TITLE_RE.search(title):
        return False
    description = deal.get("description") or ""
    if _MULTI_GAME_DESC_RE.search(description):
        return False
    if _SINGLE_GAME_TITLE_RE.search(title):
        return True
    return bool(_BUNDLE_TITLE_KEYWORDS_RE.search(title) and _SINGLE_GAME_DESC_RE.search(description))


def is_offer_placeholder(deal: dict) -> bool:
    """Kleinanzeigen "1 € VB" / "VB": the amount is a placeholder for "make me an offer"."""
    return deal.get("shipping_note") == "VB" and (deal.get("price") or 0) <= 1.0


@dataclass(frozen=True)
class PriceScope:
    """Why a listing's price doesn't buy the whole lot its title shows."""

    kind: str  # "per_item" | "price_varies" | "buyer_picks" | "sold_singly"
    evidence: str  # the wording found, as the seller wrote it
    item_price: float | None = None  # per-piece amount, when stated
    lot_price: float | None = None  # whole-lot price, when stated
    lot_includes_shipping: bool = False
    lot_is_negotiable: bool = False

    def explain(self, deal: dict) -> str:
        price = float(deal.get("price") or 0)
        quoted = f"'{self.evidence}'"
        if self.lot_price is not None:
            shipping = " incl. shipping" if self.lot_includes_shipping else ""
            return (
                f"Listed €{price:.2f} is the price per game ({quoted}); "
                f"the whole lot costs €{self.lot_price:.2f}{shipping} according to the description."
            )
        # "Sammlung von 12 verschiedenen PlayStation 3 Spielen … Pro Spiel 10€"
        count = bundle_game_count(deal.get("title") or "") or bundle_game_count(
            (deal.get("description") or "")[:300], first_only=True
        )
        unit = self.item_price or price
        total = f" — all {count} would cost about €{count * unit:.0f}" if count and unit else ""
        return {
            "per_item": f"Price is per game, not for the bundle ({quoted}){total}.",
            "price_varies": f"Price varies per game ({quoted}) — €{price:.2f} is not the price of the whole lot.",
            "buyer_picks": f"You get ONE game of your choice ({quoted}), not the pictured bundle.",
            "sold_singly": f"The games are sold individually ({quoted}) — the price is for one game.",
        }[self.kind]


_GAME_COUNT_IN_TITLE_RE = re.compile(
    r"\b(\d{1,3})\s*(?:x\s*)?(?:spiele|spielen|games|titel|stück|stk)\b", re.IGNORECASE
)


def bundle_game_count(title: str, *, first_only: bool = False) -> int:
    """Games a title claims in total ("10 PS3 Spiele / 6 PS4 Spiele" → 16); 0 if none stated.
    Platform names are removed first so "Xbox 360 Spiele" isn't read as 360 games.
    *first_only* reads just the first count — for prose, which repeats itself."""
    text = title
    for pattern, _ in _PLATFORM_MAP:
        text = pattern.sub(" ", text)
    counts = [int(n) for n in re.findall(r"\b(\d{1,3})\s+(?:[a-zäöü]+\s+){0,3}?spielen?\b", text, re.IGNORECASE)]
    if not counts:
        counts = [int(n) for n in _GAME_COUNT_IN_TITLE_RE.findall(text)]
    counts = [n for n in counts if 2 <= n <= 500][: 1 if first_only else None]
    return sum(counts)


def analyze_price_scope(deal: dict) -> PriceScope | None:
    """Does the listed price buy the whole lot the title shows? ``None`` if it
    does (or nothing says otherwise); else why not, with the whole-lot price
    when the text states one."""
    title = deal.get("title") or ""
    if not looks_like_multi_item(title):
        return None
    description = deal.get("description") or ""
    signal = _find_price_signal(title, in_title=True) or _find_price_signal(description)
    if signal is None:
        return None
    kind, evidence, item_price = signal
    lot_price, includes_shipping, lot_vb = _find_lot_price(f"{title}\n{description}")
    listed = float(deal.get("price") or 0)
    if listed and lot_price and listed >= 0.5 * lot_price:
        return None  # the listed price already is (roughly) the whole lot's
    if listed and item_price and not lot_price and listed > 1.5 * item_price:
        return None  # "Einzelpreis 5 €" on a 60 € listing: 60 € is the lot price
    return PriceScope(kind, evidence, item_price, lot_price, includes_shipping, lot_vb)


# Bait-and-switch: a bundle title sold with a plain quantity selector.
def _multi_unit_warning(deal: dict) -> str | None:
    title = deal.get("title") or ""
    seller_count = deal.get("seller_count") or ""
    if not seller_count or not _BUNDLE_TITLE_KEYWORDS_RE.search(title):
        return None
    numbers = [int(n) for n in re.findall(r"\d+", seller_count)]
    if not numbers or max(numbers) <= 1:
        return None
    return (
        f"BAIT-AND-SWITCH DETECTED: Title advertises a bundle/collection "
        f"('{title[:80]}{'...' if len(title) > 80 else ''}') but seller_count "
        f"is '{seller_count}', meaning multiple units are available or have "
        f"already been sold. A genuine one-of-a-kind bundle would have exactly "
        f"1 unit available and 0 sold. This listing almost certainly sells "
        f"individual items from the collection one by one — the seller uses a "
        f"plain 'Stückzahl' quantity selector instead of a variant/game-picker "
        f"dropdown (classic 'lazy Stückzahl trick' on German eBay). Buyer "
        f"likely receives only ONE game despite bundle appearance. AVOID."
    )


def _misleading_listing(deal: dict) -> tuple[str, str] | None:
    """(headline, warning) when the listing isn't the bundle deal it appears to be."""
    if not deal.get("title"):
        return None
    scope = analyze_price_scope(deal)
    if scope is not None:
        return "NOT A BUNDLE PRICE — AVOID", f"{scope.explain(deal)} AVOID."
    warning = _multi_unit_warning(deal)
    if warning:
        return "SCAM RISK — AVOID", warning
    return None


def _detect_bundle_individual_sale_scam(deal: dict) -> str | None:
    """Warning text when the price doesn't buy the bundle the listing shows, else ``None``.

    Covers per-piece / pick-one / sold-singly wording (see analyze_price_scope)
    and the multi-unit bait-and-switch. A listing the search pipeline already
    re-priced with its stated whole-lot price is not flagged: its price is real.
    """
    found = _misleading_listing(deal)
    return found[1] if found else None


def _apply_scam_override(deal: dict, assessment: dict) -> dict:
    """Apply the deterministic scam override to *assessment* if warranted.

    Always returns *assessment* (mutated in-place if overridden, then returned).
    """
    found = _misleading_listing(deal)
    if found is None:
        return assessment
    headline, warning = found
    assessment["ai_potential_scam"] = True
    assessment["ai_deal_rating"] = "Avoid"
    existing_warning = assessment.get("ai_scam_warning", "")
    if existing_warning:
        assessment["ai_scam_warning"] = f"{warning} | {existing_warning}"
    else:
        assessment["ai_scam_warning"] = warning
    existing_summary = assessment.get("ai_verdict_summary", "")
    prefix = f"⚠️ **{headline}**: {warning}"
    if existing_summary:
        assessment["ai_verdict_summary"] = f"{prefix}\n\n{existing_summary}"
    else:
        assessment["ai_verdict_summary"] = prefix
    return assessment


# ── Deterministic garbage result builder ────────────────────────────────────


def _build_deterministic_garbage(rating: str, confidence: int, summary: str) -> dict:
    """Build deterministic garbage/trash assessment result (no AI call needed)."""
    return {
        "ai_deal_rating": rating,
        "ai_confidence_score": confidence,
        "ai_visual_findings": [],
        "ai_red_flags": [summary],
        "ai_fair_market_estimate": "",
        "ai_itemized_resale_estimates": [],
        "ai_estimated_total_cost": 0,
        "ai_estimated_gross_profit": 0,
        "ai_verdict_summary": _GARBAGE_VERDICT_PREFIX + "\n\n" + summary,
        "ai_assessed": True,
        "ai_potential_scam": False,
        "ai_scam_warning": "",
    }


# ── Response parsing (shared) ─────────────────────────────────────────────


_DEFAULT_PARSE_ERROR: dict = {
    "ai_deal_rating": "Unknown",
    "ai_confidence_score": 0,
    "ai_visual_findings": [],
    "ai_red_flags": ["AI response could not be parsed"],
    "ai_fair_market_estimate": "",
    "ai_itemized_resale_estimates": [],
    "ai_estimated_total_cost": 0.0,
    "ai_estimated_gross_profit": 0.0,
    "ai_verdict_summary": "AI assessment parsing failed.",
    "ai_assessed": False,
    "ai_potential_scam": False,
    "ai_scam_warning": "",
}


def _parse_response(text: str) -> dict:
    """Extract the JSON payload from a single-deal AI response."""
    original_text = text
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = _sanitize_json_text(text)
    if text.startswith("["):
        text = text[1:]
    if text.endswith("]"):
        text = text[:-1]
    if text.endswith(","):
        text = text[:-1].rstrip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        found = _extract_json_objects(original_text)
        if found:
            data = found[0] if isinstance(found[0], dict) else None
    if not isinstance(data, dict):
        return dict(_DEFAULT_PARSE_ERROR)
    err = _DEFAULT_PARSE_ERROR
    return {
        "ai_deal_rating": data.get("deal_rating", "Unknown"),
        "ai_confidence_score": data.get("confidence_score", err["ai_confidence_score"]),
        "ai_visual_findings": data.get("visual_findings", err["ai_visual_findings"]),
        "ai_red_flags": data.get("red_flags", err["ai_red_flags"]),
        "ai_fair_market_estimate": data.get("fair_market_estimate", err["ai_fair_market_estimate"]),
        "ai_itemized_resale_estimates": data.get("itemized_resale_estimates", err["ai_itemized_resale_estimates"]),
        "ai_estimated_total_cost": data.get("estimated_total_cost", err["ai_estimated_total_cost"]),
        "ai_estimated_gross_profit": data.get("estimated_gross_profit", err["ai_estimated_gross_profit"]),
        "ai_verdict_summary": data.get("verdict_summary", err["ai_verdict_summary"]),
        "ai_assessed": True,
        "ai_potential_scam": data.get("potential_scam", err["ai_potential_scam"]),
        "ai_scam_warning": data.get("scam_warning", err["ai_scam_warning"]),
    }


def _parse_batch_response(text: str, expected_count: int) -> list[dict]:
    """Parse a batch AI response as a JSON array."""
    text = _sanitize_json_text(text.strip())
    lines = text.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    text = "\n".join(lines).strip()
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_json_objects(text)
        if extracted:
            items: list = []
            for obj in extracted:
                if isinstance(obj, list):
                    items.extend(obj)
                elif isinstance(obj, dict):
                    items.append(obj)
            data = items if items else None
    if not isinstance(data, list):
        return [dict(_DEFAULT_PARSE_ERROR)] * expected_count
    results: list[dict] = []
    for item_data in data:
        if not isinstance(item_data, dict):
            results.append(dict(_DEFAULT_PARSE_ERROR))
            continue
        try:
            confidence = int(float(item_data.get("confidence_score", 0)))
        except (TypeError, ValueError):
            confidence = 0
        potential_scam = bool(item_data.get("potential_scam", False))
        try:
            total_cost = float(item_data.get("estimated_total_cost", 0) or 0)
        except (TypeError, ValueError):
            total_cost = 0.0
        try:
            gross_profit = float(item_data.get("estimated_gross_profit", 0) or 0)
        except (TypeError, ValueError):
            gross_profit = 0.0
        itemized = item_data.get("itemized_resale_estimates", [])
        if not isinstance(itemized, list):
            itemized = []
        filtered_itemized = []
        for entry in itemized:
            if isinstance(entry, dict):
                game_name = str(entry.get("game") or "").strip()
                if not game_name:
                    continue
                if _is_aggregate_placeholder(game_name):
                    continue
                try:
                    price_eur = float(entry.get("price_eur") or 0)
                except (TypeError, ValueError):
                    price_eur = 0.0
                price_source = str(entry.get("price_source") or "ai_estimate")
                is_exceptional = bool(entry.get("is_exceptional", False))
                filtered_itemized.append(
                    {
                        "game": game_name,
                        "price_eur": round(price_eur, 2),
                        "price_source": price_source,
                        "is_exceptional": is_exceptional,
                    }
                )
        err = _DEFAULT_PARSE_ERROR
        results.append(
            {
                "ai_deal_rating": item_data.get("deal_rating", "Unknown"),
                "ai_confidence_score": confidence,
                "ai_visual_findings": item_data.get("visual_findings", err["ai_visual_findings"]),
                "ai_red_flags": item_data.get("red_flags", err["ai_red_flags"]),
                "ai_fair_market_estimate": item_data.get("fair_market_estimate", err["ai_fair_market_estimate"]),
                "ai_itemized_resale_estimates": filtered_itemized,
                "ai_estimated_total_cost": total_cost,
                "ai_estimated_gross_profit": gross_profit,
                "ai_verdict_summary": item_data.get("verdict_summary", err["ai_verdict_summary"]),
                "ai_assessed": True,
                "ai_potential_scam": potential_scam,
                "ai_scam_warning": item_data.get("scam_warning", err["ai_scam_warning"]),
            }
        )
    while len(results) < expected_count:
        results.append(dict(_DEFAULT_PARSE_ERROR))
    return results


# ── Base assessor class ───────────────────────────────────────────────────


class BaseAssessor:
    """Shared AI-assessor logic: deterministic rules, eBay price helpers, response parsing.

    Subclasses must implement the AI-specific methods marked with ``NotImplementedError``.
    """

    def __init__(self, api_key_env: str, default_model: str) -> None:
        """Subclasses call ``super().__init__("GEMINI_API_KEY", "gemini-model-name")``."""
        api_key = os.environ.get(api_key_env, "").strip()
        self.enabled = bool(api_key)
        self.user_enabled: bool = True
        self._model_name: str = default_model
        # Owns the eBay price cache and runs Phase A of assessment (see
        # ai_providers/enrichment.py) — concurrent, deadline-bounded price
        # lookups and image fetches, so Phase B (this class's content
        # builders) never makes a network call of its own. Imported here
        # (not at module level) because enrichment.py itself imports several
        # pure text-helpers back out of this module — a module-level import
        # in either direction would be circular.
        from ai_providers.enrichment import Enricher

        self._enricher = Enricher()
        self._timeout_executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Subclass sets provider-specific client objects
        self._client = None
        self._types = None

    # ── eBay client registry ──────────────────────────────────────────────

    def set_ebay_client(self, client: Any) -> None:
        """Register an :class:`EbayApiClient` for per-game price lookups."""
        self._enricher.ebay_client = client
        logger.info("%s: eBay client registered.", type(self).__name__)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def rate_limited_until(self) -> float:
        with _rate_limit_lock:
            return _rate_limited_until

    @property
    def is_rate_limited(self) -> bool:
        return time.monotonic() < self.rate_limited_until

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        value = value.strip()
        if not value:
            raise ValueError("model_name must not be empty")
        if value != self._model_name:
            logger.info("%s: model changed from %s to %s", type(self).__name__, self._model_name, value)
            self._model_name = value

    # ── Abstract methods (subclasses must implement) ──────────────────────

    def _try_deterministic_assessment(self, deal: dict) -> dict | None:
        broken = _detect_broken_deal(deal)
        if broken:
            return _build_deterministic_garbage("Garbage", 100, broken)
        trash = _detect_trash_title(deal)
        if trash:
            return _build_deterministic_garbage("Garbage", 100, trash)
        sports = _detect_sports_kinect_deal(deal)
        if sports:
            return {
                "ai_deal_rating": "Avoid",
                "ai_confidence_score": 100,
                "ai_visual_findings": [],
                "ai_red_flags": ["Automatically flagged — sports/Kinect content"],
                "ai_fair_market_estimate": "",
                "ai_itemized_resale_estimates": [],
                "ai_estimated_total_cost": deal.get("price", 0) or 0,
                "ai_estimated_gross_profit": 0,
                "ai_verdict_summary": sports,
                "ai_assessed": True,
                "ai_potential_scam": False,
                "ai_scam_warning": "",
            }
        scam = _detect_bundle_individual_sale_scam(deal)
        if scam:
            return {
                "ai_deal_rating": "Avoid",
                "ai_confidence_score": 100,
                "ai_visual_findings": [],
                "ai_red_flags": [],
                "ai_fair_market_estimate": "",
                "ai_itemized_resale_estimates": [],
                "ai_estimated_total_cost": deal.get("price", 0) or 0,
                "ai_estimated_gross_profit": 0,
                "ai_verdict_summary": scam,
                "ai_assessed": True,
                "ai_potential_scam": True,
                "ai_scam_warning": scam,
            }
        return None

    def _finalize_assessment(self, deal: dict, assessment: dict) -> dict:
        assessment = _apply_garbage_overrides(deal, assessment)
        assessment = _apply_sports_kinect_override(deal, assessment)
        assessment = _apply_scam_override(deal, assessment)
        return assessment

    def assess_deal(self, deal: dict) -> dict | None:
        raise NotImplementedError

    def assess_deals_batch(self, deals: list[dict], deadline: float | None = None) -> list[dict | None]:
        """Assess *deals* in batches.

        ``deadline`` is an absolute ``time.monotonic()`` timestamp by which
        assessment must stop (whatever hasn't been assessed yet comes back
        as ``None``). Callers on a request/response cycle with an external
        time budget — e.g. a reverse proxy that will kill the connection
        after N seconds regardless of what the app is still doing — should
        pass one; ``None`` falls back to a provider-chosen standalone
        default (``_ASSESS_TOTAL_BUDGET_S`` in this module) for callers
        with no such constraint.
        """
        raise NotImplementedError

    @staticmethod
    def _format_image_issues_line(deal: dict) -> str:
        issues: list[str] = deal.get("image_issues", [])
        return f"Image Issues: {', '.join(issues)}\n" if issues else ""

    # Static methods that delegate to module-level functions so subclasses
    # and external callers can use ``cls._parse_batch_response(...)``.
    @staticmethod
    def _parse_response(text: str) -> dict:
        return _parse_response(text)

    @staticmethod
    def _parse_batch_response(text: str, expected_count: int) -> list[dict]:
        return _parse_batch_response(text, expected_count)


def extract_listed_game_prices(text: str) -> list[tuple[str, float]]:
    """Parse 'Game Name – 15 €' or 'Game Name = 15 €' patterns from descriptions."""
    if not text:
        return []
    results: list[tuple[str, float]] = []
    for line in text.split("\n"):
        line = line.strip()
        m = re.match(r"(.+?)\s*[–\-–—=]\s*(\d+[\.,]?\d*)\s*€?\s*$", line)
        if m:
            name = m.group(1).strip()
            if len(name) < 3 or len(name) > 80:
                continue
            skip = ("versand", "paket", "gesamt", "alle", "preis", "porto", "ab", "nur", "zahlung")
            if any(kw in name.lower() for kw in skip):
                continue
            price_str = m.group(2).replace(",", ".")
            try:
                pr = float(price_str)
            except ValueError:
                pr = 0.0
            results.append((name, pr))
    return results[:20]
