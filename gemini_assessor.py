#!/usr/bin/env python3
"""
Gemini AI Deal Assessor
Analyzes eBay listings using Google Gemini multimodal API (text + images).
Falls back gracefully when the API key is absent or a request fails.
"""

import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import requests

if TYPE_CHECKING:
    from ebay_api_client import EbayApiClient

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
price (including shipping) is **≤ 2 €**, it is ALWAYS rated **"Must Have"** \
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
Your `deal_rating` is determined **solely by resale profit potential** \
(gross profit = estimated resale − total cost; total cost = asking price + \
shipping):
- **"Must Have"**: Gross profit ≥ total cost (i.e. ≥ 100 % return on cost). \
An amazing deal — act immediately.
- **"Good"**: Gross profit ≥ 50 % of total cost but < 100 % of total cost. \
Good profits — strong flip worth acting on.
- **"Okay"**: Gross profit > 0 but < 50 % of total cost. \
Decent profits — worth considering.
- **"Avoid"**: Gross profit ≤ 0 (unprofitable — break-even or loss), \
scam/fraud detected, or the listing is dominated by low-demand \
categories (sports/Kinect bundles, common shovelware).

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
- `"deal_rating"`: `"Must Have"` / `"Good"` / `"Okay"` / `"Avoid"`
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
- `"itemized_resale_estimates"`: list of objects — one entry per item to be \
resold; for bundles, one entry per identified game; for **single-item \
listings**, include exactly one entry for the item itself with its estimated \
resale value — **never use an empty list for a single-item listing**; each \
entry must have keys `"game"` (string title), `"price_eur"` (number, \
estimated resale price in EUR), `"price_source"` (`"ebay_sold"`, \
`"ebay_active"`, or `"ai_estimate"`), and `"is_exceptional"` (boolean — \
`true` only for standout high-value games in the bundle whose resale price \
is notably above the bundle average or which are rare/high-demand titles; \
always `false` for single-item listings and for bundles rated below "Good"); \
use provided `Fetched eBay Market Price` or `Fetched eBay Prices` data when \
available, otherwise estimate
- `"estimated_total_cost"`: number — asking price + shipping in EUR (0 if unknown)
- `"estimated_gross_profit"`: number — sum of all itemized_resale_estimates \
price_eur values minus estimated_total_cost; **always compute this value** — \
it is the sole basis for deal_rating; a value of 0 is only correct if you \
genuinely have no resale estimate at all
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
price (including shipping) is **≤ 2 €**, it is ALWAYS rated **"Must Have"** \
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
Your `deal_rating` is determined **solely by resale profit potential** \
(gross profit = estimated resale − total cost; total cost = asking price + \
shipping):
- **"Must Have"**: Gross profit ≥ total cost (i.e. ≥ 100 % return on cost). \
An amazing deal — act immediately.
- **"Good"**: Gross profit ≥ 50 % of total cost but < 100 % of total cost. \
Good profits — strong flip worth acting on.
- **"Okay"**: Gross profit > 0 but < 50 % of total cost. \
Decent profits — worth considering.
- **"Avoid"**: Gross profit ≤ 0 (unprofitable — break-even or loss), \
scam/fraud detected, or dominated by low-demand categories \
(sports/Kinect, common shovelware).

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
> - For any game that is a standout value-driver in the bundle (notably \
higher resale price than the bundle average, rare or high-demand title), \
set `"is_exceptional": true` in its `itemized_resale_estimates` entry. \
Only mark a game exceptional when the bundle rating is "Good" or better.

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
- `"deal_rating"`: `"Must Have"` / `"Good"` / `"Okay"` / `"Avoid"`
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
- `"itemized_resale_estimates"`: list of objects — one entry per item to be \
resold; for bundles, one entry per identified game; for **single-item \
listings**, include exactly one entry for the item itself with its estimated \
resale value — **never use an empty list for a single-item listing**; each \
entry must have keys `"game"` (string title), `"price_eur"` (number, \
estimated resale price in EUR), `"price_source"` (`"ebay_sold"`, \
`"ebay_active"`, or `"ai_estimate"`), and `"is_exceptional"` (boolean — \
`true` only for standout high-value games in the bundle whose resale price \
is notably above the bundle average or which are rare/high-demand titles; \
always `false` for single-item listings and for bundles rated below "Good"); \
use provided `Fetched eBay Market Price` or `Fetched eBay Prices` data when \
available, otherwise estimate
- `"estimated_total_cost"`: number — asking price + shipping in EUR (0 if unknown)
- `"estimated_gross_profit"`: number — sum of all itemized_resale_estimates \
price_eur values minus estimated_total_cost; **always compute this value** — \
it is the sole basis for deal_rating; a value of 0 is only correct if you \
genuinely have no resale estimate at all
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
# Capped at 10 to match the Gemini AFC (Adaptive Flow Control) limit of 10
# max remote calls per request.  Smaller batches also reduce per-request
# latency and memory usage.
_BATCH_SIZE = 10

# Retry configuration for transient (non-rate-limit) API errors.
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubled on each retry (exponential back-off)

# Hard timeout (seconds) applied around a single Gemini generateContent call,
# including all internal SDK retries.  With _BATCH_SIZE=10 each call is small
# enough to complete well within this window under normal network conditions.
# If the call stalls, the ThreadPoolExecutor future times out here and returns
# graceful timeout errors rather than blocking the Gunicorn worker.
_GEMINI_REQUEST_TIMEOUT = 25

# Total wall-clock budget (seconds) for the entire assess_deals_batch() call
# (all sub-batches combined).  This is the last line of defence against a
# Gunicorn worker timeout: if the cumulative Gemini time exceeds this limit,
# remaining batches are skipped and their deals are returned as timeout errors.
# Set well below the Gunicorn worker timeout (180 s) to leave room for eBay
# API calls and other per-request work.
_ASSESS_TOTAL_BUDGET_S = 90

# ---------------------------------------------------------------------------
# eBay price pre-fetch / caching settings
# ---------------------------------------------------------------------------

# Time-to-live (seconds) for in-memory eBay price cache entries.
_EBAY_CACHE_TTL = 300.0

# Total wall-clock budget (seconds) for the parallel eBay price pre-fetch step
# at the start of assess_deals_batch().  All pre-fetch threads are abandoned
# after this limit so the Gunicorn worker is never blocked waiting for slow eBay
# API responses.  Must be < (_ASSESS_TOTAL_BUDGET_S - _GEMINI_REQUEST_TIMEOUT)
# so at least one Gemini batch call can still complete within the overall budget.
_EBAY_PREFETCH_BUDGET_S = 20

# Maximum number of concurrent eBay lookup threads during the pre-fetch step.
_EBAY_MAX_WORKERS = 5

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
# eBay per-game price enrichment helpers
# ---------------------------------------------------------------------------

# Mapping from compiled platform patterns (case-insensitive) to canonical names.
# Ordered from most specific to least specific so that "Xbox 360" matches
# before a bare "Xbox" pattern.
_PLATFORM_MAP: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bxbox\s*360\b",                    re.IGNORECASE), "Microsoft Xbox 360"),
    (re.compile(r"\bxbox\s*one\b",                    re.IGNORECASE), "Microsoft Xbox One"),
    (re.compile(r"\bxbox\s*series\s*[xs]\b",          re.IGNORECASE), "Microsoft Xbox Series X"),
    (re.compile(r"\bxbox\b",                           re.IGNORECASE), "Microsoft Xbox"),
    (re.compile(r"\bps\s*5\b|\bplaystation\s*5\b",    re.IGNORECASE), "Sony PlayStation 5"),
    (re.compile(r"\bps\s*4\b|\bplaystation\s*4\b",    re.IGNORECASE), "Sony PlayStation 4"),
    (re.compile(r"\bps\s*3\b|\bplaystation\s*3\b",    re.IGNORECASE), "Sony PlayStation 3"),
    (re.compile(r"\bps\s*2\b|\bplaystation\s*2\b",    re.IGNORECASE), "Sony PlayStation 2"),
    (re.compile(r"\bpsx\b|\bps\s*1\b|\bplaystation\s*1\b", re.IGNORECASE), "Sony PlayStation"),
    (re.compile(r"\bplaystation\b",                    re.IGNORECASE), "Sony PlayStation"),
    (re.compile(r"\bnintendo\s*switch\b",              re.IGNORECASE), "Nintendo Switch"),
    (re.compile(r"\bwii\s*u\b",                        re.IGNORECASE), "Nintendo Wii U"),
    (re.compile(r"\bwii\b",                            re.IGNORECASE), "Nintendo Wii"),
    (re.compile(r"\bgamecube\b|\bgame\s*cube\b",       re.IGNORECASE), "Nintendo GameCube"),
    (re.compile(r"\bn64\b|\bnintendo\s*64\b",          re.IGNORECASE), "Nintendo 64"),
    (re.compile(r"\bsnes\b|\bsuper\s*nintendo\b",      re.IGNORECASE), "Super Nintendo"),
    (re.compile(r"\bnes\b|\bnintendo\s*entertainment\b", re.IGNORECASE), "Nintendo Entertainment System"),
    (re.compile(r"\bgba\b|\bgame\s*boy\s*advance\b",   re.IGNORECASE), "Game Boy Advance"),
    (re.compile(r"\b3ds\b",                            re.IGNORECASE), "Nintendo 3DS"),
    (re.compile(r"\bnds\b|\bnintendo\s*ds\b",          re.IGNORECASE), "Nintendo DS"),
    (re.compile(r"\bpsp\b",                            re.IGNORECASE), "PlayStation Portable"),
    (re.compile(r"\bvita\b|\bps\s*vita\b",             re.IGNORECASE), "PlayStation Vita"),
    (re.compile(r"\bdreamcast\b",                      re.IGNORECASE), "Sega Dreamcast"),
    (re.compile(r"\bsaturn\b",                         re.IGNORECASE), "Sega Saturn"),
    (re.compile(r"\bmega\s*drive\b|\bgenesis\b",       re.IGNORECASE), "Sega Mega Drive"),
]


def _extract_platform_name(title: str) -> str:
    """Return the canonical platform name detected in *title*, or empty string."""
    for pattern, canonical in _PLATFORM_MAP:
        if pattern.search(title):
            return canonical
    return ""


# Condition / noise words commonly found in single-game eBay listing titles
# that should be stripped when building a clean search query.
_SINGLE_GAME_NOISE_RE = re.compile(
    r"\b(gebraucht|neuwertig|neu|sehr\s+gut|gut|akzeptabel|like\s+new|used"
    r"|top\s+zustand|komplett|ovp|cib|sealed|ungetestet|defekt|pal|ntsc"
    r"|deutsch|german|für|fuer|for|ohne|inkl(?:usive)?|mit|version"
    r"|spiel\b|game\b)\b",
    re.IGNORECASE,
)


def _build_single_game_search_query(title: str) -> str:
    """Build an eBay search query for a single-game listing.

    Strips platform keywords and condition/noise words from *title*, then
    appends the canonical platform name in the required format::

        "GAME NAME (PLATFORM NAME)"

    If no platform can be detected the cleaned title is returned as-is.

    Examples::

        "Halo 3 Xbox 360 gebraucht"       → "Halo 3  (Microsoft Xbox 360)"
        "Batman Arkham Knight PS4 OVP neu" → "Batman Arkham Knight  (Sony PlayStation 4)"
        "Zelda Breath of the Wild Switch"  → "Zelda Breath of the  (Nintendo Switch)"
    """
    platform = _extract_platform_name(title)

    cleaned = title.strip()

    # Strip every platform pattern so the canonical name is never doubled.
    for pattern, _ in _PLATFORM_MAP:
        cleaned = pattern.sub(" ", cleaned)

    # Strip condition/noise words.
    cleaned = _SINGLE_GAME_NOISE_RE.sub(" ", cleaned)

    # Collapse extra whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n:()-")

    if len(cleaned) < 3:
        # Cleaning was too aggressive – fall back to full original title.
        cleaned = title.strip()

    if platform:
        return f"{cleaned} ({platform})"
    return cleaned


# Words that are never game titles on their own; filtered from extracted candidates.
_NON_TITLE_WORDS_RE = re.compile(
    r"^\s*(\d+|spiele?|games?|stück|pieces?|neu|used|gebraucht|like\s+new"
    r"|nintendo|playstation|ps[1-5]|xbox|sega|atari|pc|psp|ds|3ds|wii"
    r"|switch|gamecube|gameboy|game\s+boy|mega\s+drive"
    r"|sehr\s+gut|gut|akzeptabel|neuwertig|top|set|bundle"
    r"|sammlung|konvolut|paket|lot|collection|inklusive?|inkl|mit|und|and"
    r"|plus|\+|für|fuer|for|the|der|die|das|ein|eine)\s*$",
    re.IGNORECASE,
)

# Separators used to split individual titles within a bundle listing title.
_TITLE_SEPARATOR_RE = re.compile(r"\s*[,;+/&]\s*|\s+[-–—]\s+")

# Maximum number of individual game titles to search per bundle.
_MAX_GAMES_PER_BUNDLE = 8


def _extract_potential_game_titles(title: str) -> List[str]:
    """Attempt to extract individual game titles from a bundle listing title.

    Returns a list of candidate game-title strings (possibly empty).  Titles
    are extracted by splitting on common separators (commas, semicolons, '+'
    etc.) and filtering out generic words that are clearly not game titles.

    Examples::

        "PS4 Bundle: God of War, Spider-Man, Horizon" → ["God of War", "Spider-Man", "Horizon"]
        "10 PS4 Spiele Sammlung Lot"                  → []   # no identifiable titles
        "Zelda + Mario Odyssey + Kirby"               → ["Zelda", "Mario Odyssey", "Kirby"]
    """
    if not title:
        return []

    # Strip quantity patterns ("10 Spiele", "5 Games") from the start.
    cleaned = re.sub(r"^\d+\s+(spiele?|games?)\s*", "", title.strip(), flags=re.IGNORECASE)

    # Remove console/platform prefixes and bundle collection keywords so they
    # don't appear in extracted game titles.
    cleaned = re.sub(
        r"\b(nintendo|playstation|ps[1-5]|xbox|sega|atari|gamecube|gameboy"
        r"|game\s+boy|mega\s+drive|snes|nes|n64|wii|switch|3ds|ds|psp|vita|pc"
        r"|bundle|lot|paket|sammlung|konvolut|spielesammlung|spielepaket"
        r"|spieleset|spiele[- ]set|spiele[- ]paket|collection)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    parts = _TITLE_SEPARATOR_RE.split(cleaned)

    candidates: List[str] = []
    for part in parts:
        part = part.strip(" \t\n:()-")
        # Skip empty, purely numeric, or generic non-title words.
        if not part:
            continue
        if re.match(r"^\d+$", part):
            continue
        # Skip "N Spiele", "N Games" patterns (number + generic word).
        if re.match(r"^\d+\s+(spiele?|games?)\s*$", part, re.IGNORECASE):
            continue
        if _NON_TITLE_WORDS_RE.match(part):
            continue
        # Must have at least 3 characters and at least one letter.
        if len(part) < 3 or not re.search(r"[a-zA-ZäöüÄÖÜß]", part):
            continue
        candidates.append(part)

    return candidates[:_MAX_GAMES_PER_BUNDLE]

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
        # Optional eBay API client used to fetch real per-game prices for bundles.
        self._ebay_client: Optional[Any] = None
        # In-memory cache for eBay price lookup results.
        # Maps query string → (price_or_None, source_label, expire_at_monotonic)
        self._ebay_price_cache: Dict[str, Tuple[Optional[float], str, float]] = {}

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

    def set_ebay_client(self, client: Any) -> None:
        """Register an :class:`EbayApiClient` instance for per-game price lookups.

        When set, bundle listings will have individual game prices fetched from
        the eBay API before the Gemini prompt is constructed so that the AI can
        use real market data instead of guesswork.
        """
        self._ebay_client = client
        logger.info("GeminiAssessor: eBay client registered for per-game price enrichment.")

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

        Deals are grouped into batches of up to ``_BATCH_SIZE`` items (matching
        the Gemini AFC max-remote-calls limit) and sent together in a single
        ``generateContent`` call, dramatically reducing API quota consumption
        compared to one call per deal.

        A hard wall-clock budget of ``_ASSESS_TOTAL_BUDGET_S`` seconds is
        enforced across **all** sub-batches so that the Gunicorn worker is
        never blocked indefinitely even when many batches are queued.  Deals
        from batches that could not be assessed within the budget are returned
        as ``{"ai_error_type": "timeout", "ai_assessed": False}`` so the caller
        always receives a list of the same length as *deals*.

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

        total_batches = (len(deals) + _BATCH_SIZE - 1) // _BATCH_SIZE
        logger.info(
            "GeminiAssessor: assess_deals_batch start – %d deal(s) across "
            "%d batch(es) of ≤%d (AFC cap), budget=%.0f s, per-call timeout=%.0f s.",
            len(deals),
            total_batches,
            _BATCH_SIZE,
            _ASSESS_TOTAL_BUDGET_S,
            _GEMINI_REQUEST_TIMEOUT,
        )

        t_start = time.monotonic()

        # Pre-populate the eBay price cache for all deals in this request.
        # Runs all eBay API lookups in parallel within _EBAY_PREFETCH_BUDGET_S
        # so that the serial per-deal calls inside _build_batch_contents are
        # served from cache and add negligible latency.  Without this step the
        # serial eBay calls (up to 8 games × 2 API tries × 10 s each per deal)
        # could easily exceed the Gunicorn worker timeout before Gemini is
        # even invoked.
        self._prefetch_ebay_prices_parallel(deals)

        results: List[Optional[Dict]] = []
        for batch_idx, batch_start in enumerate(range(0, len(deals), _BATCH_SIZE)):
            elapsed = time.monotonic() - t_start
            budget_remaining = _ASSESS_TOTAL_BUDGET_S - elapsed

            # ── Total-budget guard ────────────────────────────────────────────
            # If the remaining budget is not strictly greater than one full call
            # timeout, any further Gemini call risks running over the Gunicorn
            # worker limit.  Return graceful timeout errors for all remaining
            # deals instead of hanging the worker process.
            if budget_remaining <= _GEMINI_REQUEST_TIMEOUT:
                skipped = len(deals) - len(results)
                logger.warning(
                    "GeminiAssessor: assess_deals_batch budget exhausted after "
                    "%.1f s (budget=%.0f s). Skipping %d remaining deal(s) "
                    "in %d remaining batch(es) – returning timeout errors.",
                    elapsed,
                    _ASSESS_TOTAL_BUDGET_S,
                    skipped,
                    total_batches - batch_idx,
                )
                results.extend(
                    [{"ai_error_type": "timeout", "ai_assessed": False}] * skipped
                )
                break

            batch = deals[batch_start : batch_start + _BATCH_SIZE]
            logger.info(
                "GeminiAssessor: batch %d/%d – %d deal(s), elapsed=%.1f s, "
                "budget_remaining=%.1f s.",
                batch_idx + 1,
                total_batches,
                len(batch),
                elapsed,
                budget_remaining,
            )

            batch_results = self._assess_batch_with_retry(batch)
            # Apply deterministic overrides for each deal in the batch.
            # Sports/Kinect override runs first, then scam override (scam takes
            # priority and can further modify the already-overridden assessment).
            for i, (deal, assessment) in enumerate(zip(batch, batch_results)):
                if assessment is not None:
                    assessment = _apply_sports_kinect_override(deal, assessment)
                    batch_results[i] = _apply_scam_override(deal, assessment)
            results.extend(batch_results)

        total_elapsed = time.monotonic() - t_start
        assessed_ok = sum(1 for a in results if a and a.get("ai_assessed"))
        timed_out = sum(
            1 for a in results if a and a.get("ai_error_type") == "timeout"
        )
        logger.info(
            "GeminiAssessor: assess_deals_batch done – %d assessed, %d timeout, "
            "%d other in %.1f s total.",
            assessed_ok,
            timed_out,
            len(results) - assessed_ok - timed_out,
            total_elapsed,
        )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_image_issues_line(deal: Dict) -> str:
        """Return a formatted 'Image Issues: …\n' line for a deal, or empty string."""
        issues: List[str] = deal.get("image_issues", [])
        return f"Image Issues: {', '.join(issues)}\n" if issues else ""

    # ------------------------------------------------------------------
    # eBay price cache helpers
    # ------------------------------------------------------------------

    def _cached_ebay_price(self, query: str) -> Optional[Tuple[Optional[float], str]]:
        """Return ``(price, source)`` from the in-memory cache if the entry is
        still valid, or ``None`` when there is no cache hit."""
        entry = self._ebay_price_cache.get(query)
        if entry is None:
            return None
        price, source, expire_at = entry
        if time.monotonic() < expire_at:
            return price, source
        # Entry has expired – evict it so stale data is never used.
        del self._ebay_price_cache[query]
        return None

    def _store_ebay_price_in_cache(self, query: str, price: Optional[float], source: str) -> None:
        """Store an eBay price result in the in-memory cache with a TTL."""
        self._ebay_price_cache[query] = (price, source, time.monotonic() + _EBAY_CACHE_TTL)

    def _collect_ebay_queries_for_deal(self, deal: Dict) -> List[str]:
        """Return the list of eBay search queries needed to price *deal*.

        Bundle listings produce one query per extracted game title; single-game
        listings produce a single ``"GAME (PLATFORM)"`` query.  Returns an empty
        list when the eBay client is absent or no meaningful query can be built.
        """
        if self._ebay_client is None:
            return []
        title = deal.get("title", "")
        if not title:
            return []
        if _BUNDLE_TITLE_KEYWORDS_RE.search(title):
            game_titles = _extract_potential_game_titles(title)
            if not game_titles:
                return []
            platform = _extract_platform_name(title)
            return [f"{g} ({platform})" if platform else g for g in game_titles]
        else:
            q = _build_single_game_search_query(title)
            return [q] if q else []

    def _prefetch_ebay_prices_parallel(self, deals: List[Dict]) -> None:
        """Pre-populate the eBay price cache for all queries needed by *deals*.

        Runs all eBay price lookups in parallel (up to ``_EBAY_MAX_WORKERS``
        concurrent threads) and stores results in :attr:`_ebay_price_cache`.
        A hard wall-clock budget of ``_EBAY_PREFETCH_BUDGET_S`` seconds is
        enforced: threads that have not completed by the deadline are abandoned
        (their queries simply won't appear in the cache, and the prompt builders
        will omit price data for those games).

        This method must be called before :meth:`_build_batch_contents` /
        :meth:`assess_deals_batch` so that the per-deal price-fetch calls inside
        those methods are served from cache rather than blocking serially on the
        eBay API.
        """
        if self._ebay_client is None:
            return

        # Collect unique queries across all deals.
        all_queries: List[str] = []
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
            logger.debug(
                "GeminiAssessor: eBay prefetch: all %d queries satisfied from cache.",
                len(all_queries),
            )
            return

        logger.info(
            "GeminiAssessor: eBay prefetch: %d unique queries (%d cached, %d to fetch, ≤%ds budget).",
            len(all_queries),
            len(all_queries) - len(uncached),
            len(uncached),
            _EBAY_PREFETCH_BUDGET_S,
        )

        def _fetch_one(query: str) -> Tuple[str, Optional[float], str]:
            try:
                price, source, _ = self._ebay_client.get_median_sold_price(query, max_results=10)
                return query, price, source
            except Exception as exc:
                logger.warning(
                    "GeminiAssessor: eBay prefetch failed for %r: %s", query, exc
                )
                return query, None, "no_result"

        t0 = time.monotonic()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=_EBAY_MAX_WORKERS)
        try:
            future_to_query = {executor.submit(_fetch_one, q): q for q in uncached}
            done, not_done = concurrent.futures.wait(
                future_to_query, timeout=_EBAY_PREFETCH_BUDGET_S
            )
            for fut in done:
                try:
                    query, price, source = fut.result()
                    self._store_ebay_price_in_cache(query, price, source)
                except Exception as exc:
                    q = future_to_query[fut]
                    logger.warning(
                        "GeminiAssessor: eBay prefetch result error for %r: %s", q, exc
                    )
            if not_done:
                logger.warning(
                    "GeminiAssessor: eBay prefetch budget (%.0fs) exhausted; "
                    "%d/%d queries did not complete in time.",
                    _EBAY_PREFETCH_BUDGET_S,
                    len(not_done),
                    len(uncached),
                )
        finally:
            # Shut down without waiting so abandoned threads do not block the
            # Gunicorn worker past the overall request budget.
            executor.shutdown(wait=False)

        elapsed = time.monotonic() - t0
        found = sum(
            1 for q in all_queries
            if (self._cached_ebay_price(q) or (None,))[0] is not None
        )
        logger.info(
            "GeminiAssessor: eBay prefetch done in %.1fs: %d/%d prices found.",
            elapsed, found, len(all_queries),
        )

    def _fetch_ebay_prices_for_bundle(self, deal: Dict) -> List[Dict]:
        """Fetch real eBay prices for individual games in a bundle listing.

        Returns a list of dicts, each with ``game``, ``price_eur``, and
        ``price_source`` keys.  Returns an empty list when:

        * No eBay client is registered.
        * The listing is not a bundle.
        * No individual game titles can be extracted from the title.
        * All eBay searches return no results.

        The ``price_source`` value is ``"ebay_sold"`` when the Marketplace
        Insights API returned sold-listing data, ``"ebay_active"`` when the
        Browse API active-listings fallback was used, and ``"no_result"`` when
        the search returned no data.
        """
        if self._ebay_client is None:
            return []

        title = deal.get("title", "")
        # Only enrich bundle listings; skip single-game listings.
        if not _BUNDLE_TITLE_KEYWORDS_RE.search(title):
            return []

        game_titles = _extract_potential_game_titles(title)
        if not game_titles:
            logger.debug(
                "GeminiAssessor: Bundle detected but no individual titles extracted from %r",
                title,
            )
            return []

        # Derive the platform name from the deal title so that the eBay search
        # query always uses the format "GAME NAME (PLATFORM NAME)" for accurate
        # minimum-price lookups.
        platform = _extract_platform_name(title)

        results: List[Dict] = []
        for game in game_titles:
            search_query = f"{game} ({platform})" if platform else game
            logger.debug(
                "GeminiAssessor: eBay price query for game %r: %r", game, search_query
            )
            # Serve from in-memory cache when available to avoid redundant API
            # calls (populated by _prefetch_ebay_prices_parallel for batches).
            cached = self._cached_ebay_price(search_query)
            if cached is not None:
                price, source = cached
                errs: List[str] = []
            else:
                try:
                    price, source, errs = self._ebay_client.get_median_sold_price(
                        search_query, max_results=10
                    )
                    self._store_ebay_price_in_cache(search_query, price, source)
                except Exception as exc:
                    logger.warning(
                        "GeminiAssessor: eBay price lookup failed for %r: %s", game, exc
                    )
                    price, source, errs = None, "no_result", []

            for e in errs:
                logger.debug("GeminiAssessor: eBay price lookup note for %r: %s", game, e)

            if price is not None:
                price_source = "ebay_sold" if source == "sold_listings" else "ebay_active"
                results.append({"game": game, "price_eur": round(price, 2), "price_source": price_source})
            else:
                results.append({"game": game, "price_eur": None, "price_source": "no_result"})

        found = sum(1 for r in results if r["price_eur"] is not None)
        logger.info(
            "GeminiAssessor: eBay price enrichment for %r: %d/%d games found prices",
            title, found, len(results),
        )
        return results

    def _fetch_ebay_price_for_single_listing(self, deal: Dict) -> Optional[float]:
        """Fetch the current lowest eBay price for a single-game listing.

        Uses the format ``"GAME NAME (PLATFORM NAME)"`` as required for
        accurate minimum-price lookups.  Returns the price in EUR, or
        ``None`` when the eBay client is unavailable or the search yields
        no results.

        Bundle listings are skipped here — they are handled by
        :meth:`_fetch_ebay_prices_for_bundle`.
        """
        if self._ebay_client is None:
            return None

        title = deal.get("title", "")
        if not title:
            return None

        # Bundle listings are handled by _fetch_ebay_prices_for_bundle.
        if _BUNDLE_TITLE_KEYWORDS_RE.search(title):
            return None

        query = _build_single_game_search_query(title)
        # Serve from in-memory cache when available to avoid redundant API calls
        # (populated by _prefetch_ebay_prices_parallel for batches).
        cached = self._cached_ebay_price(query)
        if cached is not None:
            price, source = cached
            if price is not None:
                logger.debug(
                    "GeminiAssessor: Single-game market price (cache hit) for %r "
                    "(query=%r): €%.2f (%s)",
                    title, query, price, source,
                )
            return price
        try:
            price, source, errs = self._ebay_client.get_median_sold_price(
                query, max_results=10
            )
            self._store_ebay_price_in_cache(query, price, source)
            for e in errs:
                logger.debug(
                    "GeminiAssessor: Single-game price lookup note for %r: %s",
                    title,
                    e,
                )
            if price is not None:
                logger.info(
                    "GeminiAssessor: Single-game market price for %r "
                    "(query=%r): €%.2f (%s)",
                    title,
                    query,
                    price,
                    source,
                )
            return price
        except Exception as exc:
            logger.warning(
                "GeminiAssessor: Single-game price lookup failed for %r: %s",
                title,
                exc,
            )
            return None


    @staticmethod
    def _format_ebay_prices_section(ebay_prices: List[Dict]) -> str:
        """Return a formatted text block describing fetched eBay prices, or empty string."""
        if not ebay_prices:
            return ""
        lines = ["Fetched eBay Prices (use these in itemized_resale_estimates):"]
        for entry in ebay_prices:
            game = entry.get("game", "?")
            price = entry.get("price_eur")
            source = entry.get("price_source", "unknown")
            if price is not None:
                source_label = (
                    "sold listings" if source == "ebay_sold"
                    else "active listings" if source == "ebay_active"
                    else source
                )
                lines.append(f"  - {game}: €{price:.2f} (source: {source_label})")
            else:
                lines.append(f"  - {game}: no eBay data found — please estimate")
        return "\n".join(lines)

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

        # Inject real eBay prices when available for bundle listings.
        # For single-game listings, inject the current market price so the AI
        # can compute an accurate profit estimate and issue GOOD/MUST HAVE ratings.
        ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
        if ebay_prices:
            prices_section = self._format_ebay_prices_section(ebay_prices)
            prompt_lines.append(f"\n{prices_section}")
        else:
            single_price = self._fetch_ebay_price_for_single_listing(deal)
            if single_price is not None:
                prompt_lines.append(
                    f"\nFetched eBay Market Price: €{single_price:.2f} "
                    f"(use as price_eur in itemized_resale_estimates)"
                )

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
            "potential_scam, scam_warning, visual_findings, red_flags, "
            "fair_market_estimate, itemized_resale_estimates, "
            "estimated_total_cost, estimated_gross_profit, verdict_summary."
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

            # Inject real eBay prices when available for bundle listings.
            # For single-game listings, inject the current market price so the
            # AI can compute an accurate profit estimate and issue GOOD/MUST HAVE.
            ebay_prices = self._fetch_ebay_prices_for_bundle(deal)
            if ebay_prices:
                prices_section = self._format_ebay_prices_section(ebay_prices)
                item_lines.append(f"\n{prices_section}")
            else:
                single_price = self._fetch_ebay_price_for_single_listing(deal)
                if single_price is not None:
                    item_lines.append(
                        f"\nFetched eBay Market Price: €{single_price:.2f} "
                        f"(use as price_eur in itemized_resale_estimates)"
                    )

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
        """Send *deals* as a single batch request, retrying on transient errors.

        Each ``generate_content`` call is wrapped in a :class:`ThreadPoolExecutor`
        with a hard timeout of :data:`_GEMINI_REQUEST_TIMEOUT` seconds so that
        the Gunicorn worker is never blocked indefinitely by the Gemini SDK's
        own internal retry/back-off logic.  If the timeout fires, all deals in
        the batch are returned as ``{"ai_error_type": "timeout", "ai_assessed": False}``.
        """
        global _rate_limited_until

        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                contents = self._build_batch_contents(deals)

                # ── Hard timeout wrapper ──────────────────────────────────────
                # Run generate_content in a background thread so we can impose a
                # wall-clock timeout that covers the SDK's own tenacity retries.
                t0 = time.monotonic()
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    future = executor.submit(
                        self._client.models.generate_content,
                        model=self._model_name,
                        contents=contents,
                        config=self._types.GenerateContentConfig(
                            system_instruction=_BATCH_SYSTEM_PROMPT,
                        ),
                    )
                    try:
                        response = future.result(timeout=_GEMINI_REQUEST_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        elapsed = time.monotonic() - t0
                        logger.error(
                            "GeminiAssessor: Batch of %d timed out after %.1f s "
                            "(attempt %d/%d, timeout=%d s). Returning timeout errors.",
                            len(deals), elapsed, attempt + 1, _MAX_RETRIES,
                            _GEMINI_REQUEST_TIMEOUT,
                        )
                        future.cancel()  # no-op for running futures; documents intent
                        return [{"ai_error_type": "timeout", "ai_assessed": False}] * len(deals)
                finally:
                    executor.shutdown(wait=False)
                # ── End timeout wrapper ───────────────────────────────────────

                elapsed = time.monotonic() - t0
                logger.info(
                    "GeminiAssessor: Batch of %d assessed in %.1f s (attempt %d/%d)",
                    len(deals), elapsed, attempt + 1, _MAX_RETRIES,
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
                "ai_itemized_resale_estimates": [],
                "ai_estimated_total_cost": 0.0,
                "ai_estimated_gross_profit": 0.0,
                "ai_verdict_summary": "AI response could not be parsed.",
                "ai_assessed": False,
                "ai_error_type": "parse_error",
            }

        potential_scam = bool(data.get("potential_scam", False))
        try:
            total_cost = float(data.get("estimated_total_cost", 0) or 0)
        except (TypeError, ValueError):
            total_cost = 0.0
        try:
            gross_profit = float(data.get("estimated_gross_profit", 0) or 0)
        except (TypeError, ValueError):
            gross_profit = 0.0
        itemized = data.get("itemized_resale_estimates", [])
        if not isinstance(itemized, list):
            itemized = []
        return {
            "ai_deal_rating": str(data.get("deal_rating", "Unknown")),
            "ai_confidence_score": int(data.get("confidence_score", 0)),
            "ai_potential_scam": potential_scam,
            "ai_scam_warning": str(data.get("scam_warning", "")),
            "ai_visual_findings": data.get("visual_findings", []),
            "ai_red_flags": data.get("red_flags", []),
            "ai_fair_market_estimate": str(data.get("fair_market_estimate", "")),
            "ai_itemized_resale_estimates": itemized,
            "ai_estimated_total_cost": total_cost,
            "ai_estimated_gross_profit": gross_profit,
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
            "ai_itemized_resale_estimates": [],
            "ai_estimated_total_cost": 0.0,
            "ai_estimated_gross_profit": 0.0,
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
                        "ai_itemized_resale_estimates": itemized,
                        "ai_estimated_total_cost": total_cost,
                        "ai_estimated_gross_profit": gross_profit,
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
