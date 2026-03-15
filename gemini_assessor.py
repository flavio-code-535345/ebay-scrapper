#!/usr/bin/env python3
"""
Gemini AI Deal Assessor
Analyzes eBay listings using Google Gemini multimodal API (text + images).
Falls back gracefully when the API key is absent or a request fails.
"""

import json
import logging
import os
import re
import threading
import time
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Professional eBay Deal Examiner system instructions
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
### ROLE
You are a **Professional eBay Deal Examiner** specialising in the German \
secondhand gaming market. Your job is to give the buyer a thorough, \
actionable verdict on whether this deal is worth taking. Think like an \
experienced reseller who buys purely for **resale profit** and who \
understands the German eBay market deeply. Do NOT factor in personal \
enjoyment, nostalgia, or collector value — only resale profit matters.

### ABSOLUTE RULE — THE 2 € THRESHOLD
> If a game (or bundle) is listed as **working/functional** and the total \
price (including shipping) is **≤ 2 €**, it is ALWAYS rated **"Must Buy"** \
regardless of market value, popularity, or condition.  
> State this rule explicitly in your verdict when it applies.

### SPORTS & KINECT BUNDLES — INSTANT AVOID
Sports game franchises and Kinect titles have **minimal resale value** on \
the German secondhand market. Apply these rules immediately, before any \
other analysis:
- Any listing dominated by **FIFA**, **Forza**, **TopSpin**, **NBA 2K**, \
**PES** (Pro Evolution Soccer), **Madden**, **NHL**, **WRC**, **MotoGP**, \
**Just Dance**, **Dance Central**, **Wii Sports**, or similar annual sports \
franchise titles → rate **"Avoid"** immediately. These titles flood the \
secondhand market and rarely fetch more than €1–3 each, making profit \
impossible on a bundle.
- Any listing featuring **Kinect** hardware or games (Kinect Adventures, \
Kinect Sports, Kinect Star Wars, Just Dance for Kinect, Dance Central, \
etc.) → rate **"Avoid"**. Kinect accessories and Kinect-only games have \
near-zero resale value today.
- Exception: only bump above "Avoid" if the listing also contains clearly \
identified **rare or high-value non-sports titles** that outweigh the \
sports/Kinect content in resale value and provide meaningful net profit.

### RATING DECISION — RESALE PROFIT ONLY
Your `deal_rating` is determined **solely by resale profit potential**:
- **"Must Buy"**: Estimated resale clearly exceeds (asking price + \
shipping) by at least 30–40 % gross margin **or** ≥ €8 net profit. A \
genuinely profitable flip that a reseller should act on immediately.
- **"Fair"**: Small positive margin (< 30 % gross margin or < €8 net \
profit), break-even, or uncertain — worth considering only if the price \
drops further or you already have a buyer lined up.
- **"Avoid"**: Estimated resale ≤ cost (no profit), high risk, scam/fraud \
detected, or the listing is dominated by low-demand categories (sports/ \
Kinect bundles, common shovelware).

### ANALYSIS PROTOCOL
1. **SCAM / BAIT-AND-SWITCH DETECTION (CHECK THIS FIRST)**
   This is a gating check — a scam suspicion overrides all other advice.
   Shady sellers often list a multi-game **bundle or lot** but actually sell \
only a single game chosen from a dropdown or variant selector, baiting buyers \
with the bundle image/price.

   **Check the title for red-flag keywords:**
   - "you pick", "choose 1", "Auswahl", "nur 1 Spiel", "1 Spiel nach Wahl", \
"1 aus", "1 Stück wählen", "bitte auswählen", "Ihre Wahl", "nach Wahl"

   **Check the description (CRITICAL — read every sentence):**
   Descriptions are the primary vehicle for bait-and-switch deception. Flag \
immediately if the description contains ANY of the following patterns:
   - "Sie wählen" / "Sie wählen ein Spiel" / "Sie wählen sich" — buyer must \
pick one item
   - "bitte teilen Sie mir mit" / "bitte mitteilen" / "bitte nennen Sie" / \
"bitte angeben" — buyer must tell the seller which game they want
   - "Auswahl" / "aus der Auswahl" / "aus dem Angebot wählen" — choose from \
a selection
   - "ein Spiel Ihrer Wahl" / "ein Titel Ihrer Wahl" / "Wunschspiel" — one \
game of your choice
   - "nur ein Spiel" / "nur 1 Spiel" / "ein Spiel pro Kauf" — only one game \
per purchase
   - "bitte im Nachrichtenfenster" / "bitte per Nachricht" — buyer must send \
a message to specify
   - "pro Stück" / "je Stück" / "einzeln" (when the title implies a bundle) \
— per-piece pricing on a bundle-titled listing
   - Any phrase asking the buyer to specify, choose, or message which item \
they want from a displayed collection

   **Check the seller count (quantity available / sold) — CANONICAL SCAM \
PATTERN:**
   - **DEFINITIVE BAIT-AND-SWITCH RULE**: If the title contains bundle/lot \
keywords (Spielesammlung, Sammlung, Konvolut, Paket, Lot, Bundle, Spieleset, \
Spielepaket, Set, or multiple titles listed) AND `Seller Count` shows any \
number **greater than 1** (e.g. "4 verfügbar", "4 verfügbar, 1 verkauft", \
"2 verkauft"), you MUST set `potential_scam: true` and `deal_rating: "Avoid"`. \
This is non-negotiable — no other evidence is required.
   - A genuine one-of-a-kind bundle has quantity **exactly 1** and sold \
count **0**. Multiple available/sold units + a bundle title is a \
near-certain bait-and-switch: the seller is listing individual games from \
the collection one by one, NOT selling the whole lot.
   - The absence of a dropdown/variant selector (only a plain "Stückzahl" / \
quantity field visible in the listing) further confirms the seller has no \
mechanism to let the buyer choose from the collection — they just send one \
random or cheapest game. This "lazy Stückzahl trick" is a classic scam on \
German eBay.
   - Example canonical scam: title contains "Spielesammlung", images show a \
stack of games, but `Seller Count` is "4 verfügbar, 1 verkauft" — this MUST \
be rated `"Avoid"` with `potential_scam: true` regardless of price or profit.

   **Check images:**
   - Does the photo show a whole stack/pile of games while the description \
only mentions one?
   - Look for a plain **"Stückzahl"** quantity box (not a variant/game-selector \
dropdown) — this confirms the buyer chooses quantity but NOT which game, \
making it impossible to guarantee the full bundle.
   - Look for dropdown/variant selectors or phrases like "see drop-down", \
"see options", "Variante wählen", or item specifics that list multiple titles \
as variants.

   - If the listing is genuinely a complete lot (buyer receives every game \
shown), state this explicitly: "Bundle verified: buyer receives all items."
   - If there is ANY credible sign that the buyer might receive only one game \
(not the whole lot), set `"potential_scam": true` and explain in \
`"scam_warning"` exactly what raised suspicion (quote the specific phrase or \
data point that triggered the flag).
   - When `potential_scam` is true, also set `deal_rating` to `"Avoid"` \
regardless of price or resale value. The scam warning OVERRIDES all other \
advice — even a profitable resale estimate does not rescue this verdict.

2. **IMAGE SCAN**
   - Condition Check: Look for scratches, cracks, yellowing, missing labels, \
heavy controller-stick drift wear, disc rot, broken hinges, etc.
   - Completeness: Are all expected items present? (OVP/box, manual, cables, \
power supply, memory cards, controllers, disc/cartridge)
   - Authenticity: Check labels, holograms, disc printing, font/logo details \
for signs of counterfeits or bootlegs.
   - Placeholder/Stock Photo (CRITICAL Red Flag): Manufacturer renders or \
watermarked stock images instead of real seller photos mean the actual \
condition is unknown. Flag immediately and reduce confidence.
   - No/Low-Res Images: Treat `no_images` or `low_res_only` in image_issues \
as a significant risk factor.

3. **TEXTUAL DATA SCAN**
   - Flag risky phrases: "Ungetestet" / "Untested", "Defekt" / "For parts", \
"As-is", "Verkaufe ohne Gewähr".
   - Cross-check title vs. item specifics (e.g., "Neu" in title but "Gebraucht" \
in specifics).
   - Seller feedback: ≥ 99 % = trustworthy; < 95 % = risky; new seller = \
higher caution.
   - Location (Germany-based seller expected).

4. **MARKET & RESELL ANALYSIS**
   - Estimate fair market value for this item **in the condition shown** on \
German eBay (ebay.de sold listings benchmark).
   - Assess real-world resell-ability: Is this game/console in demand right \
now? Is it rare or common on Kleinanzeigen/eBay.de?
   - Calculate estimated gross profit: resale value − asking price − \
shipping. Use this number to determine the rating per the RATING DECISION \
thresholds above. Do NOT let nostalgia, collector interest, or personal \
preference influence the rating — only profit counts.

### OUTPUT FORMAT
Return **only** a JSON object (no markdown fences, no commentary) with \
exactly these keys:
- `"deal_rating"`: `"Must Buy"` / `"Fair"` / `"Avoid"`
- `"confidence_score"`: integer 1–100
- `"potential_scam"`: boolean — `true` if this listing shows signs of \
bundle-bait or bait-and-switch (buyer likely receives only one game despite \
bundle appearance), `false` otherwise
- `"scam_warning"`: string — if `potential_scam` is true, a concise \
human-readable explanation of why (e.g. "Title says 'Spielesammlung' but \
seller_count shows '4 verfügbar, 1 verkauft' — multiple units available \
means seller is selling games individually, NOT the whole bundle"); empty \
string otherwise
- `"visual_findings"`: list of strings — physical condition observations from \
images (empty list if no images)
- `"red_flags"`: list of strings — risks from text, photos, or seller profile
- `"fair_market_estimate"`: string — estimated market value in current \
condition, e.g. `"~€12–18"`
- `"verdict_summary"`: markdown string — 3–5 sentences covering price vs. \
market value, condition, resell-ability, and a clear recommendation with \
reasoning; invoke the 2 € rule explicitly when applicable; if \
`potential_scam` is true, lead with the scam warning and make clear this \
overrides all other advice
"""

_BATCH_SYSTEM_PROMPT = """\
### ROLE
You are a **Professional eBay Deal Examiner** specialising in the German \
secondhand gaming market. Your job is to give the buyer a thorough, \
actionable verdict on each deal. Think like an experienced reseller who \
buys purely for **resale profit** and who understands the German eBay \
market deeply. Do NOT factor in personal enjoyment, nostalgia, or \
collector value — only resale profit matters.

### ABSOLUTE RULE — THE 2 € THRESHOLD
> If a game (or bundle) is listed as **working/functional** and the total \
price (including shipping) is **≤ 2 €**, it is ALWAYS rated **"Must Buy"** \
regardless of market value, popularity, or condition.  
> State this rule explicitly in the verdict when it applies.

### SPORTS & KINECT BUNDLES — INSTANT AVOID
Sports game franchises and Kinect titles have **minimal resale value** on \
the German secondhand market. Apply these rules immediately, before any \
other analysis:
- Any listing dominated by **FIFA**, **Forza**, **TopSpin**, **NBA 2K**, \
**PES** (Pro Evolution Soccer), **Madden**, **NHL**, **WRC**, **MotoGP**, \
**Just Dance**, **Dance Central**, **Wii Sports**, or similar annual sports \
franchise titles → rate **"Avoid"** immediately. These titles flood the \
secondhand market and rarely fetch more than €1–3 each, making profit \
impossible on a bundle.
- Any listing featuring **Kinect** hardware or games (Kinect Adventures, \
Kinect Sports, Kinect Star Wars, Just Dance for Kinect, Dance Central, \
etc.) → rate **"Avoid"**. Kinect accessories and Kinect-only games have \
near-zero resale value today.
- Exception: only bump above "Avoid" if the listing also contains clearly \
identified **rare or high-value non-sports titles** that outweigh the \
sports/Kinect content in resale value and provide meaningful net profit.

### RATING DECISION — RESALE PROFIT ONLY
Your `deal_rating` is determined **solely by resale profit potential**:
- **"Must Buy"**: Estimated resale clearly exceeds (asking price + \
shipping) by at least 30–40 % gross margin **or** ≥ €8 net profit. A \
genuinely profitable flip.
- **"Fair"**: Small positive margin (< 30 % gross margin or < €8 net \
profit), break-even, or uncertain — worth considering only if the price \
drops or you have a buyer.
- **"Avoid"**: Estimated resale ≤ cost (no profit), scam/fraud detected, \
or dominated by low-demand categories (sports/Kinect, common shovelware).

### BUNDLE RESALE RULE
> Whenever a listing appears to be a **multi-game lot or bundle** \
(keywords: Lot, Bundle, Sammlung, Konvolut, Paket, or multiple titles \
listed), you **MUST** include a resale breakdown in `verdict_summary`:
> - Estimate what each individual game would fetch if sold separately on \
ebay.de (or Kleinanzeigen).
> - Estimate the total potential resale revenue.
> - Calculate approximate gross profit (resale total − asking price − \
shipping) and flag any particularly valuable or worthless titles in the lot.
> - Advise whether to flip the lot whole or split it.

### ANALYSIS PROTOCOL (apply to EACH listing)
1. **SCAM / BAIT-AND-SWITCH DETECTION (CHECK THIS FIRST)**
   This is a gating check — a scam suspicion overrides all other advice.
   Shady sellers often list a multi-game **bundle or lot** but actually sell \
only a single game chosen from a dropdown or variant selector, baiting buyers \
with the bundle image/price.

   **Check the title for red-flag keywords:**
   - "you pick", "choose 1", "Auswahl", "nur 1 Spiel", "1 Spiel nach Wahl", \
"1 aus", "1 Stück wählen", "bitte auswählen", "Ihre Wahl", "nach Wahl"

   **Check the description (CRITICAL — read every sentence):**
   Descriptions are the primary vehicle for bait-and-switch deception. Flag \
immediately if the description contains ANY of the following patterns:
   - "Sie wählen" / "Sie wählen ein Spiel" / "Sie wählen sich" — buyer must \
pick one item
   - "bitte teilen Sie mir mit" / "bitte mitteilen" / "bitte nennen Sie" / \
"bitte angeben" — buyer must tell the seller which game they want
   - "Auswahl" / "aus der Auswahl" / "aus dem Angebot wählen" — choose from \
a selection
   - "ein Spiel Ihrer Wahl" / "ein Titel Ihrer Wahl" / "Wunschspiel" — one \
game of your choice
   - "nur ein Spiel" / "nur 1 Spiel" / "ein Spiel pro Kauf" — only one game \
per purchase
   - "bitte im Nachrichtenfenster" / "bitte per Nachricht" — buyer must send \
a message to specify
   - "pro Stück" / "je Stück" / "einzeln" (when the title implies a bundle) \
— per-piece pricing on a bundle-titled listing
   - Any phrase asking the buyer to specify, choose, or message which item \
they want from a displayed collection

   **Check the seller count (quantity available / sold) — CANONICAL SCAM \
PATTERN:**
   - **DEFINITIVE BAIT-AND-SWITCH RULE**: If the title contains bundle/lot \
keywords (Spielesammlung, Sammlung, Konvolut, Paket, Lot, Bundle, Spieleset, \
Spielepaket, Set, or multiple titles listed) AND `Seller Count` shows any \
number **greater than 1** (e.g. "4 verfügbar", "4 verfügbar, 1 verkauft", \
"2 verkauft"), you MUST set `potential_scam: true` and `deal_rating: "Avoid"`. \
This is non-negotiable — no other evidence is required.
   - A genuine one-of-a-kind bundle has quantity **exactly 1** and sold \
count **0**. Multiple available/sold units + a bundle title is a \
near-certain bait-and-switch: the seller is listing individual games from \
the collection one by one, NOT selling the whole lot.
   - The absence of a dropdown/variant selector (only a plain "Stückzahl" / \
quantity field visible in the listing) further confirms the seller has no \
mechanism to let the buyer choose from the collection — they just send one \
random or cheapest game. This "lazy Stückzahl trick" is a classic scam on \
German eBay.
   - Example canonical scam: title contains "Spielesammlung", images show a \
stack of games, but `Seller Count` is "4 verfügbar, 1 verkauft" — this MUST \
be rated `"Avoid"` with `potential_scam: true` regardless of price or profit.

   **Check images:**
   - Does the photo show a whole stack/pile of games while the description \
only mentions one?
   - Look for a plain **"Stückzahl"** quantity box (not a variant/game-selector \
dropdown) — this confirms the buyer chooses quantity but NOT which game, \
making it impossible to guarantee the full bundle.
   - Look for dropdown/variant selectors or phrases like "see drop-down", \
"see options", "Variante wählen", or item specifics that list multiple titles \
as variants.

   - If the listing is genuinely a complete lot (buyer receives every game \
shown), state this explicitly in the verdict: "Bundle verified: buyer \
receives all items."
   - If there is ANY credible sign that the buyer might receive only one game \
(not the whole lot), set `"potential_scam": true` and explain in \
`"scam_warning"` exactly what raised suspicion (quote the specific phrase or \
data point that triggered the flag).
   - When `potential_scam` is true, also set `deal_rating` to `"Avoid"` \
regardless of price or resale value. The scam warning OVERRIDES all other \
advice — even a profitable resale estimate does not rescue this verdict.

2. **IMAGE SCAN**
   - Condition Check: Look for scratches, cracks, yellowing, missing labels, \
heavy controller-stick drift wear, disc rot, broken hinges, etc.
   - Completeness: Are all expected items present? (OVP/box, manual, cables, \
power supply, memory cards, controllers, disc/cartridge)
   - Authenticity: Check labels, holograms, disc printing, font/logo details \
for signs of counterfeits or bootlegs.
   - Placeholder/Stock Photo (CRITICAL Red Flag): Manufacturer renders or \
watermarked stock images instead of real seller photos mean the actual \
condition is unknown. Flag immediately and reduce confidence.
   - No/Low-Res Images: Treat `no_images` or `low_res_only` in image_issues \
as a significant risk factor.

3. **TEXTUAL DATA SCAN**
   - Flag risky phrases: "Ungetestet" / "Untested", "Defekt" / "For parts", \
"As-is", "Verkaufe ohne Gewähr".
   - Cross-check title vs. item specifics (e.g., "Neu" in title but "Gebraucht" \
in specifics).
   - Seller feedback: ≥ 99 % = trustworthy; < 95 % = risky; new seller = \
higher caution.
   - Location (Germany-based seller expected).

4. **MARKET & RESELL ANALYSIS**
   - Estimate fair market value for the item **in the condition shown** on \
German eBay (ebay.de sold listings benchmark).
   - Assess real-world resell-ability: Is this game/console in demand right \
now? Is it rare or common on Kleinanzeigen/eBay.de?
   - For **bundles**: identify the most and least valuable games in the lot; \
estimate per-game and total resale value; compute profit margin.
   - Calculate estimated gross profit: resale value − asking price − \
shipping. Use this number to determine the rating per the RATING DECISION \
thresholds above. Do NOT let nostalgia, collector interest, or personal \
preference influence the rating — only profit counts.

### OUTPUT FORMAT
You MUST return a **single JSON array** where each element corresponds to one \
listing in the order they were presented. Each element must have exactly \
these keys:
- `"deal_rating"`: `"Must Buy"` / `"Fair"` / `"Avoid"`
- `"confidence_score"`: integer 1–100
- `"potential_scam"`: boolean — `true` if this listing shows signs of \
bundle-bait or bait-and-switch (buyer likely receives only one game despite \
bundle appearance), `false` otherwise
- `"scam_warning"`: string — if `potential_scam` is true, a concise \
human-readable explanation of why (e.g. "Title says 'Spielesammlung' but \
seller_count shows '4 verfügbar, 1 verkauft' — multiple units available \
means seller is selling games individually, NOT the whole bundle"); empty \
string otherwise
- `"visual_findings"`: list of strings — physical condition observations from \
images (empty list if no images)
- `"red_flags"`: list of strings — risks from text, photos, or seller profile
- `"fair_market_estimate"`: string — estimated market value in current \
condition, e.g. `"~€12–18"`
- `"verdict_summary"`: markdown string — 3–5 sentences covering price vs. \
market value, condition, resell-ability, and a clear recommendation; for \
bundles include the resale breakdown described above; invoke the 2 € rule \
explicitly when applicable; if `potential_scam` is true, lead with the scam \
warning and make clear this overrides all other advice

CRITICAL: Output ONLY the JSON array — no markdown fences, no explanation \
text, no concatenated separate objects. The entire response must be parseable \
as a single `json.loads()` call that returns a list.
"""

# Gemini model to use – gemini-2.0-flash-lite supports multimodal (text + images).
_MODEL_NAME = "gemini-2.0-flash-lite"

# Maximum number of listing images sent per item per request (keeps latency reasonable).
_MAX_IMAGES = 3

# Request timeout when downloading listing images (seconds).
_IMAGE_FETCH_TIMEOUT = 5

# Default back-off (seconds) when no retryDelay is provided by the API.
_DEFAULT_BACKOFF_SECONDS = 60

# Maximum number of deals bundled into a single Gemini generateContent call.
_BATCH_SIZE = 50

# Retry configuration for transient (non-rate-limit) API errors.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled on each retry (exponential back-off)

# ---------------------------------------------------------------------------
# Rate-limit / back-off state (shared across all GeminiAssessor instances).
# ---------------------------------------------------------------------------
_rate_limit_lock = threading.Lock()
_rate_limited_until: float = 0.0  # monotonic timestamp


def _extract_json_objects(text: str) -> list:
    """Extract all top-level JSON values (objects or arrays) from *text*.

    Handles Gemini responses that return concatenated JSON objects instead of a
    single JSON array, e.g. ``{"a":1}{"b":2}`` instead of ``[{"a":1},{"b":2}]``.
    Returns a (possibly empty) list of decoded Python values.
    """
    decoder = json.JSONDecoder()
    results = []
    pos = 0
    while pos < len(text):
        # Skip whitespace and stray commas between objects.
        while pos < len(text) and text[pos] in " \t\n\r,":
            pos += 1
        if pos >= len(text):
            break
        if text[pos] in "{[":
            try:
                obj, end_pos = decoder.raw_decode(text, pos)
                results.append(obj)
                pos = end_pos
            except json.JSONDecodeError:
                pos += 1
        else:
            pos += 1
    return results


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if *exc* looks like a 429 / RESOURCE_EXHAUSTED error."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* is a transient error that warrants a retry.

    Rate-limit errors are NOT transient (they are handled separately with a
    long back-off window).  Network timeouts and 5xx server errors are.
    """
    if _is_rate_limit_error(exc):
        return False
    msg = str(exc).lower()
    return any(x in msg for x in ("timeout", "connection", "503", "500", "502", "504"))


def _parse_retry_delay(exc: Exception) -> Optional[float]:
    """Try to extract retryDelay (seconds) from the Gemini API error payload."""
    try:
        msg = str(exc)
        # Match patterns like "retryDelay": "30s" or "retry_delay": "30s"
        match = re.search(r'"retry[_\s]?[Dd]elay"\s*:\s*"(\d+(?:\.\d+)?)s?"', msg)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Deterministic bundle bait-and-switch scam detector
# ---------------------------------------------------------------------------

# Bundle/collection keywords in German and English that indicate a multi-item lot.
_BUNDLE_TITLE_KEYWORDS_RE = re.compile(
    r"\b(spielesammlung|spielepaket|spieleset|spiele[- ]set|spiele[- ]paket"
    r"|sammlung|konvolut|paket|lot|bundle|collection|spielekonvolut"
    r"|spiele[- ]sammlung|spiele[- ]konvolut)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Deterministic sports / Kinect deal detector
# ---------------------------------------------------------------------------

# Sports franchises and Kinect titles that have minimal resale value.
# Checked against the deal title (case-insensitive).
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
    r"|wii\s+sports"
    r"|wrc\b"
    r")\b",
    re.IGNORECASE,
)


# Human-readable prefix prepended to verdict_summary when a sports/Kinect
# override fires, so the user sees a clear, consistent explanation.
_SPORTS_KINECT_AVOID_PREFIX = (
    "⛔ **SPORTS/KINECT — AVOID**: This listing contains sports or "
    "Kinect game titles (FIFA, Forza, TopSpin, Kinect, etc.) that "
    "have minimal resale value in the current German eBay market. "
    "These titles rarely generate profit and are best avoided unless "
    "the bundle also contains clearly high-value non-sports games."
)


def _detect_sports_kinect_deal(deal: Dict) -> Optional[str]:
    """Deterministic check for sports-franchise or Kinect-themed listings.

    Sports game franchises (FIFA, Forza, TopSpin, NBA, PES, Madden, etc.)
    and Kinect titles have very low resale value in the German eBay market
    and should be rated "Avoid" by default.

    Returns a warning string if sports/Kinect content is detected in the
    title, ``None`` otherwise.
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


def _apply_sports_kinect_override(deal: Dict, assessment: Dict) -> Dict:
    """Apply a deterministic 'Avoid' override for sports/Kinect themed deals.

    If ``_detect_sports_kinect_deal`` fires, this function forces
    ``ai_deal_rating`` to ``"Avoid"`` and prepends a clear explanation to
    the verdict summary and red flags.

    Always returns *assessment* (mutated in-place if overridden).
    """
    warning = _detect_sports_kinect_deal(deal)
    if warning is None:
        return assessment

    logger.info(
        "GeminiAssessor: Sports/Kinect override applied for listing %r",
        deal.get("title", "?"),
    )

    assessment["ai_deal_rating"] = "Avoid"

    existing_flags = assessment.get("ai_red_flags")
    if not isinstance(existing_flags, list):
        existing_flags = []
    if "Sports/Kinect content: low resale value" not in existing_flags:
        assessment["ai_red_flags"] = existing_flags + [
            "Sports/Kinect content: low resale value"
        ]

    existing_summary = assessment.get("ai_verdict_summary", "")
    if existing_summary:
        assessment["ai_verdict_summary"] = (
            f"{_SPORTS_KINECT_AVOID_PREFIX}\n\n{existing_summary}"
        )
    else:
        assessment["ai_verdict_summary"] = _SPORTS_KINECT_AVOID_PREFIX

    return assessment

def _detect_bundle_individual_sale_scam(deal: Dict) -> Optional[str]:
    """Deterministic check for the 'bundle title + individual-unit sale' scam.

    Pattern (canonical case reported by user):
    - Title contains bundle/collection keywords (e.g. 'Spielesammlung').
    - ``seller_count`` shows multiple units available or sold (any number > 1),
      e.g. "4 verfügbar, 1 verkauft" or "3 verfügbar".

    When this pattern matches the seller is almost certainly offering individual
    games from the collection, NOT the whole lot.  The plain "Stückzahl" quantity
    selector (instead of a game-picker dropdown) is the tell-tale UI sign that
    confirms the individual-sale intent — this is the "lazy Stückzahl trick".

    Returns a warning string if the scam pattern is detected, ``None`` otherwise.
    """
    title = deal.get("title", "")
    seller_count = deal.get("seller_count", "")

    if not title or not seller_count:
        return None

    # Step 1 — title must contain at least one bundle/collection keyword.
    if not _BUNDLE_TITLE_KEYWORDS_RE.search(title):
        return None

    # Step 2 — seller_count must contain a number greater than 1.
    numbers = [int(n) for n in re.findall(r"\d+", seller_count)]
    if not numbers or max(numbers) <= 1:
        return None

    # Both conditions met: canonical bait-and-switch pattern detected.
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


def _apply_scam_override(deal: Dict, assessment: Dict) -> Dict:
    """Apply the deterministic scam override to *assessment* if warranted.

    If ``_detect_bundle_individual_sale_scam`` fires, this function forces:
    - ``ai_potential_scam = True``
    - ``ai_deal_rating = "Avoid"``
    - ``ai_scam_warning`` is set (or prepended) with the deterministic warning.
    - ``ai_verdict_summary`` is prepended with a prominent scam notice.

    Always returns *assessment* (mutated in-place if overridden, then returned).
    """
    warning = _detect_bundle_individual_sale_scam(deal)
    if warning is None:
        return assessment

    logger.info(
        "GeminiAssessor: Deterministic scam override applied for listing %r",
        deal.get("title", "?"),
    )

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


class GeminiAssessor:
    """Wraps the Gemini API for multimodal eBay deal assessment."""

    def __init__(self) -> None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.enabled = bool(api_key)
        self.user_enabled: bool = True  # Can be toggled via the UI; persisted in settings.
        self._client = None
        self._types = None
        self._model_name: str = _MODEL_NAME

        if self.enabled:
            try:
                from google import genai  # lazy import
                from google.genai import types

                self._client = genai.Client(api_key=api_key)
                self._types = types
                logger.info("GeminiAssessor: Gemini API initialised (model=%s)", self._model_name)
            except Exception as exc:
                logger.error("GeminiAssessor: Failed to initialise Gemini client: %s", exc)
                self.enabled = False
        else:
            logger.info(
                "GeminiAssessor: GEMINI_API_KEY not set — AI assessment disabled; "
                "falling back to rules engine."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def rate_limited_until(self) -> float:
        """Return the monotonic timestamp until which Gemini is rate-limited (0 if none)."""
        with _rate_limit_lock:
            return _rate_limited_until

    @property
    def is_rate_limited(self) -> bool:
        """Return True if the Gemini quota is currently exhausted."""
        return time.monotonic() < self.rate_limited_until

    @property
    def model_name(self) -> str:
        """Return the Gemini model string currently in use."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Update the Gemini model used for all future requests (no restart needed)."""
        value = value.strip()
        if not value:
            raise ValueError("model_name must not be empty (e.g., gemini-2.0-flash-lite)")
        if value != self._model_name:
            logger.info(
                "GeminiAssessor: model changed from %s to %s",
                self._model_name,
                value,
            )
            self._model_name = value

    def assess_deal(self, deal: Dict) -> Optional[Dict]:
        """Analyse *deal* with Gemini and return an AI-assessment dict.

        Returns ``None`` when:
        - The API key is not configured.
        - The Gemini API call fails (network error, etc.).
        Returns a dict with ``ai_error_type`` set when:
        - A 429 rate-limit is hit (``ai_error_type="rate_limit"``).
        - The response cannot be parsed as JSON (``ai_error_type="parse_error"``).
        Callers should fall back to the rules-based engine in these cases.
        """
        global _rate_limited_until

        if not self.enabled or self._client is None or self._types is None:
            return None

        # If we are currently in a rate-limit back-off window, skip the call.
        pause_remaining = self.rate_limited_until - time.monotonic()
        if pause_remaining > 0:
            logger.warning(
                "GeminiAssessor: rate-limited – skipping AI assessment for %r "
                "(%.0f s remaining in back-off).",
                deal.get("title", "?"),
                pause_remaining,
            )
            return {"ai_error_type": "rate_limit", "ai_assessed": False}

        try:
            contents = self._build_contents(deal)
            # The new google.genai SDK requires per-request config; there is no
            # global model object that holds a system instruction.
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                ),
            )
            result = self._parse_response(response.text)
            result = _apply_sports_kinect_override(deal, result)
            return _apply_scam_override(deal, result)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                with _rate_limit_lock:
                    _rate_limited_until = time.monotonic() + delay
                logger.warning(
                    "GeminiAssessor: 429 RESOURCE_EXHAUSTED for %r – "
                    "backing off %.0f s. Gemini AI temporarily paused.",
                    deal.get("title", "?"),
                    delay,
                )
                return {"ai_error_type": "rate_limit", "ai_assessed": False}

            logger.error(
                "GeminiAssessor: API error for listing %r: %s",
                deal.get("title", "?"),
                exc,
            )
            return None

    def assess_deals_batch(self, deals: List[Dict]) -> List[Optional[Dict]]:
        """Assess a list of *deals* in as few Gemini requests as possible.

        Deals are grouped into batches of up to ``_BATCH_SIZE`` items and sent
        together in a single ``generateContent`` call, dramatically reducing
        API quota consumption compared to one call per deal.

        Returns a list of the same length as *deals*.  Each element is either:
        - A dict with AI assessment fields (``ai_assessed=True``).
        - A dict with ``ai_assessed=False`` and ``ai_error_type`` set.
        - ``None`` if the assessor is disabled or an unrecoverable error occurred.
        """
        if not self.enabled or self._client is None or self._types is None:
            return [None] * len(deals)

        if not deals:
            return []

        # If we are currently in a rate-limit back-off window, skip everything.
        pause_remaining = self.rate_limited_until - time.monotonic()
        if pause_remaining > 0:
            logger.warning(
                "GeminiAssessor: rate-limited – skipping batch of %d items "
                "(%.0f s remaining in back-off).",
                len(deals),
                pause_remaining,
            )
            return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)

        results: List[Optional[Dict]] = []
        for batch_start in range(0, len(deals), _BATCH_SIZE):
            batch = deals[batch_start : batch_start + _BATCH_SIZE]
            batch_results = self._assess_batch_with_retry(batch)
            # Apply deterministic overrides for each deal in the batch.
            # Sports/Kinect override runs first, then scam override (scam takes
            # priority and can further modify the already-overridden assessment).
            for i, (deal, assessment) in enumerate(zip(batch, batch_results)):
                if assessment is not None:
                    assessment = _apply_sports_kinect_override(deal, assessment)
                    batch_results[i] = _apply_scam_override(deal, assessment)
            results.extend(batch_results)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_image_issues_line(deal: Dict) -> str:
        """Return a formatted 'Image Issues: …\n' line for a deal, or empty string."""
        issues: List[str] = deal.get("image_issues", [])
        return f"Image Issues: {', '.join(issues)}\n" if issues else ""

    def _build_contents(self, deal: Dict) -> List:
        """Construct the Gemini contents list (text + image parts)."""
        title = deal.get("title", "Unknown")
        price = deal.get("price", 0)
        condition = deal.get("condition", "Unknown")
        shipping = deal.get("shipping", "Unknown")
        seller_rating = deal.get("seller_rating", 0)
        description = deal.get("description", "")
        seller_count = deal.get("seller_count", "")

        prompt_lines = [
            "Analyze this eBay listing:\n",
            f"Title: {title}",
            f"Price: €{price:.2f}",
            f"Condition: {condition}",
            f"Shipping: {shipping}",
            f"Seller Rating: {seller_rating}%",
        ]
        if seller_count:
            prompt_lines.append(f"Seller Count (available/sold): {seller_count}")
        if description:
            prompt_lines.append(f"Description: {description}")
        image_issues_line = self._format_image_issues_line(deal)
        if image_issues_line:
            prompt_lines.append(image_issues_line.rstrip())
        prompt_lines.append("\nReturn your analysis in the required JSON format.")
        text_prompt = "\n".join(prompt_lines)

        parts: List = [self._types.Part.from_text(text=text_prompt)]

        image_urls: List[str] = deal.get("image_urls", [])
        for url in image_urls[:_MAX_IMAGES]:
            image_part = self._fetch_image_part(url)
            if image_part is not None:
                parts.append(image_part)

        return parts

    def _build_batch_contents(self, deals: List[Dict]) -> List:
        """Construct the Gemini contents list for a batch of *deals*.

        Each deal is introduced with a numbered separator so that Gemini can
        unambiguously map its array response back to the original items.
        """
        parts: List = []

        intro = (
            f"Below are {len(deals)} eBay listings to analyze. "
            f"Return a JSON array of exactly {len(deals)} objects in the same "
            "order. Each object must contain: deal_rating, confidence_score, "
            "visual_findings, red_flags, fair_market_estimate, verdict_summary."
        )
        parts.append(self._types.Part.from_text(text=intro))

        for idx, deal in enumerate(deals, 1):
            title = deal.get("title", "Unknown")
            price = deal.get("price", 0)
            condition = deal.get("condition", "Unknown")
            shipping = deal.get("shipping", "Unknown")
            seller_rating = deal.get("seller_rating", 0)
            description = deal.get("description", "")
            seller_count = deal.get("seller_count", "")

            item_lines = [
                f"\n--- ITEM {idx} ---",
                f"Title: {title}",
                f"Price: €{price:.2f}",
                f"Condition: {condition}",
                f"Shipping: {shipping}",
                f"Seller Rating: {seller_rating}%",
            ]
            if seller_count:
                item_lines.append(f"Seller Count (available/sold): {seller_count}")
            if description:
                item_lines.append(f"Description: {description}")
            image_issues_line = self._format_image_issues_line(deal)
            if image_issues_line:
                item_lines.append(image_issues_line.rstrip())
            item_text = "\n".join(item_lines)
            parts.append(self._types.Part.from_text(text=item_text))

            for url in deal.get("image_urls", [])[:_MAX_IMAGES]:
                image_part = self._fetch_image_part(url)
                if image_part is not None:
                    parts.append(image_part)

        parts.append(
            self._types.Part.from_text(
                text=(
                    f"\nNow return a JSON array of exactly {len(deals)} analysis "
                    "objects, one per item in order, with no other text."
                )
            )
        )
        return parts

    def _assess_batch_with_retry(self, deals: List[Dict]) -> List[Optional[Dict]]:
        """Send *deals* as a single batch request, retrying on transient errors."""
        global _rate_limited_until

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                contents = self._build_batch_contents(deals)
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=self._types.GenerateContentConfig(
                        system_instruction=_BATCH_SYSTEM_PROMPT,
                    ),
                )
                return self._parse_batch_response(response.text, len(deals))
            except Exception as exc:
                if _is_rate_limit_error(exc):
                    delay = _parse_retry_delay(exc) or _DEFAULT_BACKOFF_SECONDS
                    with _rate_limit_lock:
                        _rate_limited_until = time.monotonic() + delay
                    logger.warning(
                        "GeminiAssessor: 429 RESOURCE_EXHAUSTED (batch of %d) – "
                        "backing off %.0f s. Gemini AI temporarily paused.",
                        len(deals),
                        delay,
                    )
                    return [{"ai_error_type": "rate_limit", "ai_assessed": False}] * len(deals)

                last_exc = exc
                if _is_transient_error(exc) and attempt < _MAX_RETRIES - 1:
                    retry_delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "GeminiAssessor: Transient error on attempt %d/%d "
                        "(batch of %d) – retrying in %.1f s: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        len(deals),
                        retry_delay,
                        exc,
                    )
                    time.sleep(retry_delay)
                    continue

                # Non-transient error or retries exhausted.
                logger.error(
                    "GeminiAssessor: Batch API error (attempt %d/%d, batch of %d): %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    len(deals),
                    exc,
                )
                return [None] * len(deals)

        logger.error(
            "GeminiAssessor: All %d retries exhausted for batch of %d: %s",
            _MAX_RETRIES,
            len(deals),
            last_exc,
        )
        return [None] * len(deals)

    def _fetch_image_part(self, url: str):
        """Download *url* and return a Gemini-compatible image Part, or None."""
        try:
            resp = requests.get(url, timeout=_IMAGE_FETCH_TIMEOUT)
            resp.raise_for_status()
            mime_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            if not mime_type.startswith("image/"):
                mime_type = "image/jpeg"

            return self._types.Part.from_bytes(data=resp.content, mime_type=mime_type)
        except Exception as exc:
            logger.warning("GeminiAssessor: Could not fetch image %s: %s", url, exc)
            return None

    @staticmethod
    def _parse_response(text: str) -> Dict:
        """Extract the JSON payload from Gemini's response text."""
        original_text = text
        text = text.strip()

        # Strip optional markdown code fences.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Last-ditch: try to find a JSON object anywhere in the text.
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                try:
                    data = json.loads(obj_match.group())
                except json.JSONDecodeError as inner_exc:
                    logger.error(
                        "GeminiAssessor: JSON parse failed after extraction – %s. "
                        "Raw response (first 500 chars): %r",
                        inner_exc,
                        original_text[:500],
                    )
            else:
                logger.error(
                    "GeminiAssessor: No JSON object found in Gemini response. "
                    "Raw response (first 500 chars): %r",
                    original_text[:500],
                )

        if data is None:
            return {
                "ai_deal_rating": "Unknown",
                "ai_confidence_score": 0,
                "ai_potential_scam": False,
                "ai_scam_warning": "",
                "ai_visual_findings": [],
                "ai_red_flags": [],
                "ai_fair_market_estimate": "",
                "ai_verdict_summary": "AI response could not be parsed.",
                "ai_assessed": False,
                "ai_error_type": "parse_error",
            }

        potential_scam = bool(data.get("potential_scam", False))
        return {
            "ai_deal_rating": str(data.get("deal_rating", "Unknown")),
            "ai_confidence_score": int(data.get("confidence_score", 0)),
            "ai_potential_scam": potential_scam,
            "ai_scam_warning": str(data.get("scam_warning", "")),
            "ai_visual_findings": data.get("visual_findings", []),
            "ai_red_flags": data.get("red_flags", []),
            "ai_fair_market_estimate": str(data.get("fair_market_estimate", "")),
            "ai_verdict_summary": str(data.get("verdict_summary", "")),
            "ai_assessed": True,
        }

    @staticmethod
    def _parse_batch_response(text: str, expected_count: int) -> List[Dict]:
        """Parse a batch Gemini response as a JSON array.

        Returns a list of exactly *expected_count* assessment dicts.  Missing
        or unparseable items are filled with a parse-error sentinel so the
        caller always gets a list of the right length.
        """
        _parse_error: Dict = {
            "ai_deal_rating": "Unknown",
            "ai_confidence_score": 0,
            "ai_potential_scam": False,
            "ai_scam_warning": "",
            "ai_visual_findings": [],
            "ai_red_flags": [],
            "ai_fair_market_estimate": "",
            "ai_verdict_summary": "AI response could not be parsed.",
            "ai_assessed": False,
            "ai_error_type": "parse_error",
        }

        original_text = text
        text = text.strip()

        # Strip optional markdown code fences.
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("GeminiAssessor: Direct JSON parse failed: %s", exc)

            # Fallback 1: find a JSON array anywhere in the text.
            arr_match = re.search(r"\[[\s\S]*\]", text)
            if arr_match:
                try:
                    data = json.loads(arr_match.group())
                except json.JSONDecodeError as inner_exc:
                    logger.warning(
                        "GeminiAssessor: Batch JSON array parse failed – %s. "
                        "Trying concatenated-object extraction.",
                        inner_exc,
                    )

            # Fallback 2: extract concatenated JSON objects (e.g. {...}{...}).
            if data is None:
                extracted = _extract_json_objects(text)
                if extracted:
                    items: List = []
                    for obj in extracted:
                        if isinstance(obj, list):
                            items.extend(obj)
                        elif isinstance(obj, dict):
                            items.append(obj)
                    if items:
                        data = items
                        logger.info(
                            "GeminiAssessor: Extracted %d items via "
                            "concatenated-object fallback.",
                            len(items),
                        )

            if data is None:
                logger.error(
                    "GeminiAssessor: All JSON parse strategies failed for batch. "
                    "Raw response (first 500 chars): %r",
                    original_text[:500],
                )

        # Normalise: if Gemini returned a single object, wrap it.
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            return [dict(_parse_error)] * expected_count

        results: List[Dict] = []
        for item_data in data:
            if not isinstance(item_data, dict):
                results.append(dict(_parse_error))
            else:
                try:
                    confidence = int(float(item_data.get("confidence_score", 0)))
                except (TypeError, ValueError):
                    confidence = 0
                potential_scam = bool(item_data.get("potential_scam", False))
                results.append(
                    {
                        "ai_deal_rating": str(item_data.get("deal_rating", "Unknown")),
                        "ai_confidence_score": confidence,
                        "ai_potential_scam": potential_scam,
                        "ai_scam_warning": str(item_data.get("scam_warning", "")),
                        "ai_visual_findings": item_data.get("visual_findings", []),
                        "ai_red_flags": item_data.get("red_flags", []),
                        "ai_fair_market_estimate": str(
                            item_data.get("fair_market_estimate", "")
                        ),
                        "ai_verdict_summary": str(item_data.get("verdict_summary", "")),
                        "ai_assessed": True,
                    }
                )

        # Pad or truncate to match the expected count.
        if len(results) < expected_count:
            missing = expected_count - len(results)
            logger.warning(
                "GeminiAssessor: Batch response had %d items but expected %d; "
                "padding %d with parse-error sentinels.",
                len(results),
                expected_count,
                missing,
            )
            results.extend([dict(_parse_error)] * missing)
        elif len(results) > expected_count:
            results = results[:expected_count]

        return results
