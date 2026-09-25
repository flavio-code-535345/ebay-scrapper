"""Kleinanzeigen.de search scraper.

Primary parser: the structured data every results page embeds for its own
analytics — an Astro island whose props carry ``resultAds[]`` (id, title,
price, date, location, shipping availability, images, ...). That's far more
stable than the page's utility-class HTML, which is the fallback.

Kleinanzeigen IP-bans aggressively ("IP-Bereich vorübergehend gesperrt",
HTTP 403, after roughly six requests a minute), so requests are spaced well
apart and a ban pauses this source instead of retrying into it.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse

import requests
from bs4 import BeautifulSoup

from models import normalize_condition, parse_listing_date

logger = logging.getLogger(__name__)

_BASE = "https://www.kleinanzeigen.de"
# Category 227 = "Videospiele"; the leading slug is cosmetic (the site ignores it).
_SEARCH_URL = _BASE + "/s-pc-videospiele/{slug}/k0c227"
_REQUEST_TIMEOUT = 20
_REQUEST_SPACING_S = 5.0
_BAN_COOLDOWN_S = 600.0
_MAX_IMAGES = 3

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}

_VB_RE = re.compile(r"\b(vb|verhandlungsbasis)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"\d[\d.]*(?:,\d+)?")
_WANTED_TITLE_RE = re.compile(r"^\s*(suche|gesucht)\b", re.IGNORECASE)
_DATE_TEXT_RE = re.compile(r"^(heute|gestern)\b|^\d{1,2}\.\d{1,2}\.\d{4}$", re.IGNORECASE)
_POSTCODE_RE = re.compile(r"^\d{5}\b")
_RESULT_COUNT_RE = re.compile(r"von\s+([\d.]+)")
_IMAGE_RULE_RE = re.compile(r"\?rule=\$_\d+\.AUTO")


def _devalue(node):
    """Unwrap Astro's serialized props, where every value is a ``[type, value]``
    pair (type 1 = array of further pairs, anything else = plain value)."""
    if isinstance(node, list) and len(node) == 2 and isinstance(node[0], int):
        kind, value = node
        return [_devalue(v) for v in value] if kind == 1 else _devalue(value)
    if isinstance(node, dict):
        return {k: _devalue(v) for k, v in node.items()}
    return node


def _parse_price(text: str) -> tuple[float, bool]:
    """Parse a German price string ("1.200 €", "45 € VB", "VB") → (amount, is_vb)."""
    text = text or ""
    is_vb = bool(_VB_RE.search(text))
    m = _AMOUNT_RE.search(text)
    if not m:
        return 0.0, is_vb
    try:
        value = float(m.group(0).replace(".", "").replace(",", "."))
    except ValueError:
        return 0.0, is_vb
    return (value, is_vb) if 0 < value <= 500_000 else (0.0, is_vb)


def _infer_condition(title: str, description: str) -> str:
    """Kleinanzeigen's result list shows no condition field — infer one from text."""
    combined = f"{title} {description}".lower()
    if any(w in combined for w in ("defekt", "kaputt", "bastler", "ersatzteile")):
        return "Defekt"
    if any(w in combined for w in ("neu", "ovp", "originalverpackt", "unbenutzt")):
        return "Neu"
    if any(w in combined for w in ("sehr gut", "top zustand", "einwandfrei")):
        return "Sehr gut"
    if any(w in combined for w in ("gut", "gebraucht")):
        return "Gebraucht"
    return ""


def _full_size_image(url: str) -> str:
    return _IMAGE_RULE_RE.sub("?rule=$_59.AUTO", url)


def _build_deal(
    *,
    ad_id: str,
    title: str,
    url: str,
    price_text: str,
    description: str,
    date_text: str,
    location: str,
    shipping_available: bool,
    image_urls: list[str],
) -> dict:
    price, is_vb = _parse_price(price_text)
    condition = _infer_condition(title, description)
    parsed_date = parse_listing_date(date_text, "kleinanzeigen")
    return {
        "title": title[:300],
        "price": price,
        "condition": condition,
        "condition_normalized": normalize_condition(condition),
        # Kleinanzeigen shows no seller rating on search results.
        "seller_rating": 0.0,
        "url": url,
        "listing_id": f"kleinanzeigen:{ad_id}",
        "source": "kleinanzeigen",
        "shipping": "Versand möglich" if shipping_available else "Nur Abholung",
        # Shipping is arranged per ad; its cost isn't on the results page.
        "shipping_cost": None,
        "shipping_note": "VB" if is_vb else "",
        "is_trending": False,
        "item_location": location,
        "description": description[:2000],
        "seller_count": "",
        "listing_date": parsed_date.isoformat() if parsed_date else None,
        "image_urls": image_urls[:_MAX_IMAGES],
        "image_issues": [] if image_urls else ["no_images"],
    }


class KleinanzeigenScraper:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)
        self._last_request = 0.0
        self._blocked_until = 0.0
        # Held for the whole check-sleep-update sequence so concurrent callers
        # are serialized into properly spaced requests instead of a burst.
        self._rate_limit_lock = threading.Lock()

    def _rate_limit(self) -> None:
        with self._rate_limit_lock:
            elapsed = time.monotonic() - self._last_request
            if elapsed < _REQUEST_SPACING_S:
                time.sleep(_REQUEST_SPACING_S - elapsed)
            self._last_request = time.monotonic()

    def search(self, query: str, max_results: int = 50) -> tuple[list[dict], list[str]]:
        if not query or not query.strip():
            return [], ["query is required"]
        remaining_ban = self._blocked_until - time.monotonic()
        if remaining_ban > 0:
            return [], [f"Kleinanzeigen is blocking this server's IP — paused for {remaining_ban / 60:.0f} more min."]

        slug = urllib.parse.quote(re.sub(r"\s+", "-", query.strip().lower()), safe="-")
        url = _SEARCH_URL.format(slug=slug)
        self._rate_limit()
        try:
            resp = self._session.get(url, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            return [], [f"Kleinanzeigen request error: {exc}"]

        if resp.status_code in (403, 429):
            self._blocked_until = time.monotonic() + _BAN_COOLDOWN_S
            logger.warning("Kleinanzeigen: HTTP %d — pausing this source for %.0fs", resp.status_code, _BAN_COOLDOWN_S)
            return [], [
                f"Kleinanzeigen HTTP {resp.status_code}: it rate-limits by IP — "
                f"pausing Kleinanzeigen searches for {_BAN_COOLDOWN_S / 60:.0f} min."
            ]
        if not resp.ok:
            return [], [f"Kleinanzeigen HTTP {resp.status_code}"]

        # The server sends no charset, which would make requests decode as
        # ISO-8859-1 ("Große" → "GroÃŸe"); the page is UTF-8.
        return self.parse_results_page(resp.content.decode("utf-8", errors="replace"), max_results)

    def parse_results_page(self, html: str, max_results: int = 50) -> tuple[list[dict], list[str]]:
        """Parse a search-results page into deals. Separate from :meth:`search`
        so it can be tested against captured real pages."""
        soup = BeautifulSoup(html, "html.parser")
        deals = self._parse_structured(soup)
        if deals is None:
            logger.info("Kleinanzeigen: no embedded result data — falling back to HTML cards")
            deals = self._parse_html(soup)
        deals = deals[:max_results]

        errors: list[str] = []
        if not deals:
            summary = soup.select_one("#srp-breadcrumb-summary")
            m = _RESULT_COUNT_RE.search(summary.get_text(" ", strip=True)) if summary else None
            if m and int(m.group(1).replace(".", "")) > 0:
                errors.append(
                    "Kleinanzeigen returned results but none could be parsed — its markup has likely changed."
                )
        logger.info("Kleinanzeigen: parsed %d deals", len(deals))
        return deals, errors

    # ── Primary: embedded structured data ───────────────────────────────────

    def _parse_structured(self, soup) -> list[dict] | None:
        for island in soup.select("astro-island[props]"):
            raw = island.get("props", "")
            if "resultAds" not in raw:
                continue
            try:
                ads = _devalue(json.loads(raw)).get("resultAds")
            except (ValueError, AttributeError):
                continue
            if isinstance(ads, list):
                return [d for d in (self._deal_from_preview(ad) for ad in ads) if d]
        return None

    def _deal_from_preview(self, ad: dict) -> dict | None:
        preview = ad.get("organicAdPreview") if isinstance(ad, dict) else None
        if not preview:
            return None  # sponsored-ad slot
        title = (preview.get("title") or "").strip()
        tags = [str(t) for t in (preview.get("attributes") or []) + (preview.get("appliedFeatures") or [])]
        if not title or "Gesuch" in tags or _WANTED_TITLE_RE.match(title):
            return None
        images = []
        for img in preview.get("imageList") or []:
            src = (img or {}).get("xLargeUrl") or (img or {}).get("adTableThumbnailPrioUrl")
            if src:
                images.append(_full_size_image(src))
        location = " ".join(p for p in (preview.get("locationName"), preview.get("parentLocationName")) if p)
        return _build_deal(
            ad_id=str(preview.get("id", "")),
            title=title,
            url=_BASE + (preview.get("seoLink") or ""),
            price_text=preview.get("price") or "",
            description=(preview.get("description") or "").strip(),
            date_text=preview.get("sortingDate") or "",
            location=location,
            shipping_available=bool(preview.get("shippingAvailableValue")),
            image_urls=images,
        )

    # ── Fallback: HTML result cards ─────────────────────────────────────────

    def _parse_html(self, soup) -> list[dict]:
        deals = []
        for article in soup.select("article[data-adid]"):
            try:
                deal = self._deal_from_article(article)
            except Exception as exc:
                logger.warning("Kleinanzeigen: failed to parse an ad card: %s", exc)
                continue
            if deal:
                deals.append(deal)
        return deals

    def _deal_from_article(self, article) -> dict | None:
        link = article.select_one("h3 a[href]")
        title = link.get_text(" ", strip=True) if link else ""
        spans = [s.get_text(" ", strip=True) for s in article.select("span")]
        if not title or "Gesuch" in spans or _WANTED_TITLE_RE.match(title):
            return None
        href = link.get("href") or article.get("data-href") or ""
        prices = [p for p in article.select("p") if "line-through" not in (p.get("class") or [])]
        price_text = next((p.get_text(" ", strip=True) for p in prices if "€" in p.text or _VB_RE.search(p.text)), "")
        description_el = link.find_parent("h3").find_next_sibling("p")
        tags = [t.get_text(" ", strip=True) for t in article.select("span[data-dhl-promotion]")]
        img = article.select_one("img[src]")
        return _build_deal(
            ad_id=article.get("data-adid", ""),
            title=title,
            url=href if href.startswith("http") else _BASE + href,
            price_text=price_text,
            description=description_el.get_text(" ", strip=True) if description_el else "",
            date_text=next((s for s in spans if _DATE_TEXT_RE.match(s)), ""),
            location=next((s for s in spans if _POSTCODE_RE.match(s)), ""),
            shipping_available="Versand möglich" in tags,
            image_urls=[_full_size_image(img["src"])] if img else [],
        )
