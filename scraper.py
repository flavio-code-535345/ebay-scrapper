"""eBay.de HTML search-results scraper — the fallback engine when no Browse API
credentials are configured.

Parses eBay's current result-card markup (``ul.srp-results > li.s-card``).
eBay fronts these pages with bot protection that, depending on the network,
may refuse any non-browser client outright (HTTP 403); when that happens this
scraper says so plainly and points at the official Browse API
(``EBAY_CLIENT_ID``/``EBAY_CLIENT_SECRET``), which is the reliable path.
"""

import logging
import os
import random
import re
import threading
import time

import requests
from bs4 import BeautifulSoup

from models import canonical_listing_id, normalize_condition, parse_listing_date

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.ebay.de/sch/i.html"
_REQUEST_TIMEOUT = 15
# eBay's "Videospiele & Konsolen" category — the same one the Browse API client searches.
_VIDEO_GAMES_CATEGORY = "1249"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_BLOCKED_HINT = (
    "eBay's bot protection refused this automated request — it blocks non-browser clients on some networks. "
    "Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET to use the official eBay Browse API instead."
)

# Card text patterns (all German, as served by ebay.de).
_CONDITION_ROW_RE = re.compile(
    r"^(neu|gebraucht|generalüberholt|hervorragend|sehr gut|gut|akzeptabel|nur ersatzteile|defekt)\b", re.IGNORECASE
)
_BIDS_RE = re.compile(r"^(\d+)\s+gebote?$", re.IGNORECASE)
# "+EUR 9,68 Lieferung", "+ ca. EUR 12,27 Versand", "Gratis 2-3 Tage Lieferung" —
# but not "Lieferung an Abholstation möglich" or "Kostenloser Rückversand" (returns).
_SHIPPING_RE = re.compile(r"^(?:\+\s*(?:ca\.\s*)?eur\s*([\d.,]+)|gratis|kostenlos\w*)\b.*(?:lieferung|versand)", re.I)
_FOREIGN_LOCATION_RE = re.compile(r"^aus\s+(.+)$", re.IGNORECASE)
_AGE_ROW_RE = re.compile(r"^vor\s+\d+\s*\w+\.?\s+eingestellt$", re.IGNORECASE)
_SELLER_RATING_RE = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s*%\s*positiv", re.IGNORECASE)
_POPULARITY_RE = re.compile(r"^(\d+)\+?\s+(verkauft|beobachter)", re.IGNORECASE)
_IMAGE_SIZE_RE = re.compile(r"/s-l\d+\.")


def _parse_eur_amount(text: str) -> float | None:
    """Parse the first amount in an eBay price string ("EUR 1.234,56", "$20.00")."""
    m = re.search(r"\d[\d.,]*", text or "")
    if not m:
        return None
    num = m.group(0).rstrip(".,")
    if "," in num and "." in num:
        num = num.replace(".", "").replace(",", ".") if num.rfind(",") > num.rfind(".") else num.replace(",", "")
    elif "," in num:
        num = num.replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


def _text(el) -> str:
    return el.get_text(" ", strip=True) if el else ""


class EbayScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or ""
        proxy_https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or ""
        if proxy or proxy_https:
            self.session.proxies = {"http": proxy or proxy_https, "https": proxy_https or proxy}
            logger.info("EbayScraper: HTTP proxy configured (%s)", self.session.proxies)
        self._last_request = 0.0
        # Concurrent callers are serialized into randomized, human-ish spacing:
        # eBay's anti-bot heuristic reacts to bursts, not steady volume.
        self._rate_limit_lock = threading.Lock()

    def _rate_limit(self) -> None:
        with self._rate_limit_lock:
            elapsed = time.monotonic() - self._last_request
            delay = random.uniform(1, 3)
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._last_request = time.monotonic()

    def search(self, query: str, max_results: int = 50) -> tuple[list[dict], list[str]]:
        """Search ebay.de, newest listings first, items located in Germany.

        *query* may use eBay's search syntax — ``(a,b,c)`` OR groups and
        ``-word`` exclusions — which the web search honors.
        """
        params = {
            "_nkw": query,
            "_sop": "10",  # newly listed first ("12" is best match)
            "LH_PrefLoc": "1",  # item located in Germany
            "_sacat": _VIDEO_GAMES_CATEGORY,
            "_ipg": "120" if max_results > 60 else "60",
            "rt": "nc",
        }
        self._rate_limit()
        logger.info("eBay scraper: searching %r", query)
        try:
            response = self.session.get(_SEARCH_URL, params=params, timeout=_REQUEST_TIMEOUT)
        except requests.exceptions.Timeout:
            return [], [f"eBay search timed out after {_REQUEST_TIMEOUT}s"]
        except (requests.exceptions.ConnectionError, ConnectionError) as exc:
            return [], [f"eBay connection error: {exc}"]

        if response.status_code in (403, 429):
            logger.warning("eBay scraper: HTTP %d (bot protection)", response.status_code)
            return [], [f"eBay HTTP {response.status_code}: {_BLOCKED_HINT}"]
        if not response.ok:
            return [], [f"eBay HTTP {response.status_code}: {response.reason}"]

        return self.parse_results_page(response.content, max_results)

    def parse_results_page(self, html: bytes | str, max_results: int = 50) -> tuple[list[dict], list[str]]:
        """Parse a search-results page into deals. Separate from :meth:`search`
        so it can be tested against captured real pages."""
        soup = BeautifulSoup(html, "html.parser")
        results_list = soup.select_one("ul.srp-results")
        if results_list is None:
            title = _text(soup.title) or "(no <title>)"
            return [], [f"eBay returned an unexpected page ({title!r}) — no results list. {_BLOCKED_HINT}"]

        # Direct children only: eBay's "Shop on eBay" placeholder cards live
        # outside the results list, and ad/pagination blocks in it aren't s-cards.
        cards = results_list.select(":scope > li.s-card")
        deals: list[dict] = []
        failed = 0
        for card in cards:
            if len(deals) >= max_results:
                break
            try:
                deal = self._parse_card(card)
            except Exception as exc:
                failed += 1
                logger.warning("eBay scraper: failed to parse a result card: %s", exc, exc_info=True)
                continue
            if deal:
                deals.append(deal)

        errors: list[str] = []
        if failed:
            errors.append(f"{failed} eBay result card(s) could not be parsed and were skipped.")
        if cards and not deals and failed:
            errors.append("eBay returned results but none could be parsed — the page markup has likely changed.")
        logger.info("eBay scraper: %d cards → %d deals", len(cards), len(deals))
        return deals, errors

    def _parse_card(self, card) -> dict | None:
        title_el = card.select_one(".s-card__title")
        if title_el is not None:
            for noise in title_el.select(".clipped"):
                noise.decompose()
        title = _text(title_el)
        if not title or title.lower() == "shop on ebay":
            return None

        link = card.select_one(".su-card-container__header a.s-card__link[href]") or card.select_one(
            "a.s-card__link[href]"
        )
        href = link.get("href", "") if link else ""
        listing_num = card.get("data-listingid", "")
        if not (listing_num.isdigit() and len(listing_num) >= 9):
            canonical = canonical_listing_id(href)
            listing_num = canonical.split(":", 1)[1] if canonical else ""
        if not listing_num:
            return None

        primary_rows = card.select(".su-card-container__attributes__primary .s-card__attribute-row")
        # A price *range* ("EUR 17,44 bis EUR 186,09") is a multi-variation
        # listing — the buyer picks one item — not a single-lot deal.
        if any(" bis " in _text(r) for r in primary_rows if r.select_one(".s-card__price")):
            return None
        price = _parse_eur_amount(_text(card.select_one(".s-card__price"))) or 0.0

        condition = "Unknown"
        for sub in card.select(".s-card__subtitle"):
            sub_text = " ".join(_text(sub).split())
            if _CONDITION_ROW_RE.match(sub_text):
                condition = sub_text
                break

        listing_type = "fixed"
        shipping, shipping_cost = "", None
        listing_date = None
        item_location = ""
        seller_count = ""
        is_trending = False
        for text in (_text(r) for r in primary_rows):
            if _BIDS_RE.match(text):
                listing_type = "auction"
            elif not shipping and "rück" not in text.lower() and (m := _SHIPPING_RE.match(text)):
                if m.group(1):
                    shipping_cost = _parse_eur_amount(m.group(1))
                    shipping = f"€{shipping_cost:.2f}" if shipping_cost is not None else text
                else:
                    shipping, shipping_cost = "Free", 0.0
            elif _AGE_ROW_RE.match(text):
                parsed = parse_listing_date(text, "scraper")
                listing_date = parsed.isoformat() if parsed else None
            elif m := _FOREIGN_LOCATION_RE.match(text):
                item_location = m.group(1)
            elif m := _POPULARITY_RE.match(text):
                is_trending = True
                if m.group(2).lower() == "verkauft":
                    seller_count = text

        seller_rating = 0.0
        for row in card.select(".su-card-container__attributes__secondary .s-card__attribute-row"):
            if m := _SELLER_RATING_RE.search(_text(row)):
                seller_rating = float(m.group(1).replace(",", "."))
                break

        image_urls: list[str] = []
        for img in card.select("img.s-card__image"):
            url = img.get("data-defer-load") or img.get("src") or ""
            if "i.ebayimg.com" not in url:
                continue
            url = _IMAGE_SIZE_RE.sub("/s-l500.", url)
            if url not in image_urls:
                image_urls.append(url)

        return {
            "title": title,
            "price": price,
            "condition": condition,
            "condition_normalized": normalize_condition(condition),
            "seller_rating": seller_rating,
            "url": f"https://www.ebay.de/itm/{listing_num}",
            "listing_id": f"ebay:{listing_num}",
            "source": "ebay",
            "shipping": shipping,
            "shipping_cost": shipping_cost,
            "is_trending": is_trending,
            "item_location": item_location,
            "description": "",
            "seller_count": seller_count,
            "listing_date": listing_date,
            "listing_type": listing_type,
            "image_urls": image_urls,
            "image_issues": [] if image_urls else ["no_images"],
        }
