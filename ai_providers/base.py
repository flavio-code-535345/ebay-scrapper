"""Base assessor — shared AI-agnostic logic, deterministic rules, eBay price helpers."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from typing import Any

_PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

with open(os.path.join(_PROMPT_DIR, "system_prompt.txt")) as _f:
    _SYSTEM_PROMPT = _f.read()

with open(os.path.join(_PROMPT_DIR, "batch_system_prompt.txt")) as _f:
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
_EBAY_CACHE_TTL = 300.0
_EBAY_PREFETCH_BUDGET_S = 15
_EBAY_MAX_WORKERS = 5

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


# ── Bait-and-switch scam detection (deterministic) ────────────────────────

# Description-level scam phrases — when the description matches these,
# it's almost certainly a bait-and-switch even if seller_count is clean.
_DESC_SCAM_RE = re.compile(
    r"\b("
    r"sie\s+wählen|bitte\s+(teilen|mitteilen|nennen|angeben|auswählen)"
    r"|ein\s+spiel\s+ihrer\s+wahl|ihrer\s+wahl|nach\s+wahl"
    r"|wunschspiel|nur\s+(ein|1)\s+spiel"
    r"|bitte\s+im\s+nachrichtenfenster|bitte\s+per\s+nachricht"
    r"|pro\s+stück|je\s+stück"
    r"|spiel\s+aussuchen|spiel\s+auswählen"
    r"|einzeln\s+(verkauf|verkauft|erhältlich|kaufbar)"
    r"|auswahl\s+(aus|von|treffen)"
    r")\b",
    re.IGNORECASE,
)

# Title-level scam patterns — these are definitive.
_TITLE_SCAM_RE = re.compile(
    r"\b("
    r"you\s+pick|choose\s+1|auswahl|nur\s+1\s+spiel|1\s+spiel\s+nach\s+wahl"
    r"|1\s+aus|1\s+stück\s+wählen|bitte\s+auswählen|ihre\s+wahl|nach\s+wahl"
    r")\b",
    re.IGNORECASE,
)


def _detect_bundle_individual_sale_scam(deal: dict) -> str | None:
    """Check for the 'bundle title + individual-unit sale' scam.

    Returns a warning string, or ``None`` if no scam detected.
    """
    title = deal.get("title", "")
    description = deal.get("description", "")
    seller_count = deal.get("seller_count", "")

    if not title:
        return None

    # Check 1: title-level scam keywords (deterministic — any match = scam).
    if _BUNDLE_TITLE_KEYWORDS_RE.search(title) and _TITLE_SCAM_RE.search(title):
        short_title = title[:80] + ("..." if len(title) > 80 else "")
        return (
            f"BAIT-AND-SWITCH DETECTED (title keyword): '{short_title}' "
            f"contains bundle terms AND 'you pick' / 'Auswahl' / 'choose' "
            f"wording — the listing shows a collection but sells only one game. AVOID."
        )

    # Check 2: description scam phrases — any match = scam.
    if description and _BUNDLE_TITLE_KEYWORDS_RE.search(title) and _DESC_SCAM_RE.search(description):
        match = _DESC_SCAM_RE.search(description)
        short_title = title[:80] + ("..." if len(title) > 80 else "")
        return (
            f"BAIT-AND-SWITCH DETECTED (description): Title advertises a bundle "
            f"('{short_title}') but description contains '{match.group(0)}' which "
            f"indicates buyer selects/chooses individual items. AVOID."
        )

    # Check 3: seller_count > 1 + bundle title = canonical scam.
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


def _apply_scam_override(deal: dict, assessment: dict) -> dict:
    """Apply the deterministic scam override to *assessment* if warranted.

    Always returns *assessment* (mutated in-place if overridden, then returned).
    """
    warning = _detect_bundle_individual_sale_scam(deal)
    if warning is None:
        return assessment
    assessment["ai_potential_scam"] = True
    assessment["ai_deal_rating"] = "Avoid"
    existing_warning = assessment.get("ai_scam_warning", "")
    if existing_warning:
        assessment["ai_scam_warning"] = f"{warning} | {existing_warning}"
    else:
        assessment["ai_scam_warning"] = warning
    existing_summary = assessment.get("ai_verdict_summary", "")
    scam_prefix = (
        "⚠️ **SCAM RISK — AVOID**: This listing shows the classic 'bundle "
        "title + multiple units available' bait-and-switch pattern. The seller "
        "almost certainly sends only one game despite the bundle appearance. "
        "Do NOT purchase unless the seller explicitly confirms you receive the "
        "full collection."
    )
    if existing_summary:
        assessment["ai_verdict_summary"] = f"{scam_prefix}\n\n{existing_summary}"
    else:
        assessment["ai_verdict_summary"] = scam_prefix
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
        self._ebay_client: Any | None = None
        self._ebay_price_cache: dict[str, tuple[float | None, str, float]] = {}
        self._prefetch_executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._timeout_executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Subclass sets provider-specific client objects
        self._client = None
        self._types = None

    # ── eBay client registry ──────────────────────────────────────────────

    def set_ebay_client(self, client: Any) -> None:
        """Register an :class:`EbayApiClient` for per-game price lookups."""
        self._ebay_client = client
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

    def _build_deal_text_prompt(self, deal: dict, *, description_limit: int = 1500, include_header: bool = True) -> str:
        title = deal.get("title", "Unknown")
        price = deal.get("price", "?")
        condition = deal.get("condition", "?")
        seller_rating = deal.get("seller_rating", "?")
        shipping = deal.get("shipping", "?")
        description = deal.get("description", "")
        seller_count = deal.get("seller_count", "")
        item_location = deal.get("item_location", "")
        listing_date = deal.get("listing_date", "")
        lines: list[str] = []
        if include_header:
            lines.append("Analyze this eBay listing:\n")
        lines.extend(
            [
                f"Title: {title}",
                f"Price: €{price}",
                f"Shipping: {shipping}",
                f"Condition: {condition}",
                f"Seller rating: {seller_rating}%",
                f"Seller Count: {seller_count}",
                f"Item Location: {item_location}",
                f"Listing Date: {listing_date}",
            ]
        )
        ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
        if not ebay_prices:
            single_price = self._fetch_ebay_price_for_single_listing(deal)
            if single_price is not None:
                lines.append(f"\nFetched eBay Market Price: €{single_price:.2f}")
        else:
            lines.append("\n" + self._format_ebay_prices_section(ebay_prices).rstrip())
        image_issues_line = self._format_image_issues_line(deal)
        if image_issues_line:
            lines.append(image_issues_line.rstrip())
        if description:
            lines.append(f"\nDescription:\n{description[:description_limit]}")
        return "\n".join(lines)

    def _finalize_assessment(self, deal: dict, assessment: dict) -> dict:
        assessment = _apply_garbage_overrides(deal, assessment)
        assessment = _apply_sports_kinect_override(deal, assessment)
        assessment = _apply_scam_override(deal, assessment)
        return assessment

    def assess_deal(self, deal: dict) -> dict | None:
        raise NotImplementedError

    def assess_deals_batch(self, deals: list[dict]) -> list[dict | None]:
        raise NotImplementedError

    # ── eBay price cache ──────────────────────────────────────────────────

    def _cached_ebay_price(self, query: str) -> tuple[float | None, str] | None:
        entry = self._ebay_price_cache.get(query)
        if entry is None:
            return None
        price, source, expire_at = entry
        if time.monotonic() >= expire_at:
            del self._ebay_price_cache[query]
            return None
        return price, source

    def _store_ebay_price_in_cache(self, query: str, price: float | None, source: str) -> None:
        self._ebay_price_cache[query] = (price, source, time.monotonic() + _EBAY_CACHE_TTL)

    def _collect_ebay_queries_for_deal(self, deal: dict) -> list[str]:
        if self._ebay_client is None:
            return []
        title = deal.get("title", "")
        if not _BUNDLE_TITLE_KEYWORDS_RE.search(title):
            q = _build_single_game_search_query(title)
            return [q] if q else []
        game_titles = _extract_potential_game_titles(title)
        platform = _extract_platform_name(title)
        queries: list[str] = []
        for game in game_titles:
            q = f"{game} ({platform})" if platform else game
            if len(q) >= 5:
                queries.append(q)
        return queries

    def _prefetch_ebay_prices_parallel(self, deals: list[dict]) -> None:
        if self._ebay_client is None:
            return
        if not deals:
            return
        all_queries: list[str] = []
        seen: set = set()
        for deal in deals:
            for q in self._collect_ebay_queries_for_deal(deal):
                if q not in seen:
                    seen.add(q)
                    all_queries.append(q)
        if not all_queries:
            return
        uncached = [q for q in all_queries if self._cached_ebay_price(q) is None]
        if not uncached:
            return
        logger.info(
            "GeminiAssessor: eBay prefetch: %d unique queries (%d cached, %d to fetch, ≤%ds budget).",
            len(all_queries),
            len(all_queries) - len(uncached),
            len(uncached),
            _EBAY_PREFETCH_BUDGET_S,
        )

        def _fetch_one(query: str) -> tuple[str, float | None, str]:
            try:
                price, source, _ = self._ebay_client.get_median_sold_price(query, max_results=10)
                return query, price, source
            except Exception as exc:
                logger.warning("GeminiAssessor: eBay prefetch failed for %r: %s", query, exc)
                return query, None, "no_result"

        t0 = time.monotonic()
        if self._prefetch_executor is None:
            self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_EBAY_MAX_WORKERS)
        future_to_query = {self._prefetch_executor.submit(_fetch_one, q): q for q in uncached}
        done, not_done = concurrent.futures.wait(future_to_query, timeout=_EBAY_PREFETCH_BUDGET_S)
        for fut in done:
            try:
                query, price, source = fut.result()
                self._store_ebay_price_in_cache(query, price, source)
            except Exception as exc:
                q = future_to_query[fut]
                logger.warning("GeminiAssessor: eBay prefetch result error for %r: %s", q, exc)
        if not_done:
            logger.warning(
                "GeminiAssessor: eBay prefetch budget (%.0fs) exhausted; %d/%d queries did not complete.",
                _EBAY_PREFETCH_BUDGET_S,
                len(not_done),
                len(uncached),
            )
        elapsed = time.monotonic() - t0
        found = sum(1 for q in all_queries if (self._cached_ebay_price(q) or (None,))[0] is not None)
        logger.info(
            "GeminiAssessor: eBay prefetch done in %.1fs: %d/%d prices found.",
            elapsed,
            found,
            len(all_queries),
        )

    def _fetch_ebay_prices_for_bundle(self, deal: dict) -> list[dict]:
        if self._ebay_client is None:
            return []
        title = deal.get("title", "")
        if not _BUNDLE_TITLE_KEYWORDS_RE.search(title):
            return []
        game_titles = _extract_potential_game_titles(title)
        if not game_titles:
            return []
        platform = _extract_platform_name(title)
        results: list[dict] = []
        for game in game_titles:
            search_query = f"{game} ({platform})" if platform else game
            cached = self._cached_ebay_price(search_query)
            if cached is not None:
                price, source = cached
                errs: list[str] = []
            else:
                try:
                    price, source, errs = self._ebay_client.get_median_sold_price(search_query, max_results=10)
                except Exception as exc:
                    logger.warning("GeminiAssessor: bundle price fetch failed for %r: %s", search_query, exc)
                    price, source = None, "no_result"
            if price is not None:
                price_source = "ebay_sold" if source == "sold_listings" else "ebay_active"
                results.append(
                    {
                        "game": game,
                        "price_eur": round(price, 2),
                        "price_source": price_source,
                    }
                )
            else:
                results.append({"game": game, "price_eur": None, "price_source": "no_result"})
        return results

    def _fetch_ebay_price_for_single_listing(self, deal: dict) -> float | None:
        if self._ebay_client is None:
            return None
        title = deal.get("title", "")
        query = _build_single_game_search_query(title)
        if not query:
            return None
        cached = self._cached_ebay_price(query)
        if cached is not None:
            return cached[0]
        try:
            price, source, _ = self._ebay_client.get_median_sold_price(query, max_results=10)
            self._store_ebay_price_in_cache(query, price, source)
            return price
        except Exception as exc:
            logger.warning("GeminiAssessor: single-listing price fetch failed for %r: %s", query, exc)
            return None

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

    @staticmethod
    def _format_ebay_prices_section(ebay_prices: list[dict]) -> str:
        if not ebay_prices:
            return ""
        lines = ["Fetched eBay Prices:"]
        for entry in ebay_prices:
            game = entry.get("game", "?")
            price = entry.get("price_eur")
            src = entry.get("price_source", "?")
            price_str = f"€{price:.2f}" if price is not None else "N/A"
            lines.append(f"  - {game}: {price_str} ({src})")
        return "\n".join(lines) + "\n"
