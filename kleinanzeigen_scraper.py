"""Kleinanzeigen.de scraper — HTML parser for classifieds listings."""

from __future__ import annotations

import logging
import re
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_KLEINANZEIGEN_BASE = "https://www.kleinanzeigen.de"
_SEARCH_URL = f"{_KLEINANZEIGEN_BASE}/s-{{}}/k0"
_REQUEST_TIMEOUT = 20
_REQUEST_DELAY = 1.5

# Kleinanzeigen uses hashed class names. Match by known patterns.
_ARTICLE_SELECTORS = [
    "article.aditem",
    "li.ad-listitem",
    '[class*="aditem"]',
    '[class*="AdItem"]',
    "ul[id*='srp'] > li",
    "ul[class*='srp'] > li",
]

_LINK_SELECTORS = [
    'a[class*="ellipsis"]',
    "article a[href]",
    "li a[href]",
    'a[href*="/s-anzeige/"]',
    "a[href*='s-anzeige']",
]

_PRICE_SELECTORS = [
    '[class*="aditem-main--middle--price"]',
    '[class*="Price"]',
    "p[class*='metr']",
    "span[class*='price']",
    "p[class*='price']",
]

_LOCATION_SELECTORS = [
    '[class*="aditem-main--top--left"]',
    "[class*='top']",
    '[class*="Top"]',
    "span[class*='loc']",
]

_DESCRIPTION_SELECTORS = [
    '[class*="aditem-main--middle--description"]',
    '[class*="Description"]',
    "p[class*='desc']",
    "span[class*='desc']",
]

_RATING_SELECTORS = [
    'span[class*="rating"], span[class*="Rating"]',
    'span[class*="top"]',
    '[class*="Bewertung"]',
]

_VB_PATTERN = re.compile(r"\b(vb|verhandlungsbasis|verhandelbar|preis\s*vorschlag)\b", re.IGNORECASE)
_PRICE_RE = re.compile(r"[\d.,\s]+")


class KleinanzeigenScraper:
    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
            }
        )
        self._last_request = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < _REQUEST_DELAY:
            time.sleep(_REQUEST_DELAY - elapsed)
        self._last_request = time.monotonic()

    def search(self, query: str, max_results: int = 50) -> tuple[list[dict], list[str]]:
        if not query or not query.strip():
            return [], ["query is required"]
        errors: list[str] = []
        all_deals: list[dict] = []

        search_url = _SEARCH_URL.format(requests.utils.quote(query.strip()))
        self._rate_limit()
        try:
            resp = self._session.get(search_url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 429 or resp.status_code == 403:
                logger.warning("Kleinanzeigen: blocked (HTTP %d) — skipping.", resp.status_code)
                return [], [f"Kleinanzeigen returned {resp.status_code} — blocked or captcha"]
            if not resp.ok:
                logger.warning("Kleinanzeigen: HTTP %d", resp.status_code)
                return [], [f"Kleinanzeigen HTTP {resp.status_code}"]
        except requests.RequestException as exc:
            logger.warning("Kleinanzeigen: request failed: %s", exc)
            return [], [f"Kleinanzeigen request error: {exc}"]

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = []
        for sel in _ARTICLE_SELECTORS:
            articles = soup.select(sel)
            if articles:
                break

        if not articles:
            return all_deals, errors

        for article in articles[:max_results]:
            deal = self._parse_article(article)
            if deal:
                all_deals.append(deal)

        return all_deals, errors

    # ── Parsing ────────────────────────────────────────────────────────

    def _parse_article(self, article) -> dict | None:
        try:
            link_el = self._select_first(article, _LINK_SELECTORS)
            if not link_el:
                return None
            href = link_el.get("href", "")
            if not href:
                return None
            if href.startswith("/"):
                href = _KLEINANZEIGEN_BASE + href

            title = self._extract_title(article, link_el)
            if not title:
                return None

            price, is_vb = self._extract_price(article)
            location = self._extract_text(article, _LOCATION_SELECTORS)
            description = self._extract_text(article, _DESCRIPTION_SELECTORS)
            rating = self._extract_rating(article)
            condition = self._extract_condition(description, title)
            date_el = article.select_one('[class*="aditem-main--top--right"]') or article.select_one("[class*='date']")
            listing_date = date_el.get_text(strip=True) if date_el else ""

            # Extract images from the article
            images = []
            for img in article.select("img[src], img[data-src]"):
                src = img.get("src") or img.get("data-src") or ""
                if src and "kleinanzeigen" in src:
                    images.append(src)
            for img in article.select("img[srcset]"):
                srcset = img.get("srcset", "")
                for part in srcset.split(","):
                    part_url = part.strip().split(" ")[0]
                    if part_url and "kleinanzeigen" in part_url:
                        images.append(part_url)

            return {
                "title": title[:300],
                "price": price,
                "condition": condition,
                "seller_rating": rating,
                "url": href,
                "shipping": "VB" if is_vb else "",
                "is_trending": False,
                "item_location": location,
                "description": description[:2000],
                "seller_count": "",
                "listing_date": listing_date,
                "image_urls": images[:3],
                "image_issues": [] if images else ["no_images"],
            }
        except Exception:
            return None

    # ── Extractors ─────────────────────────────────────────────────────

    @staticmethod
    def _select_first(element, selectors: list[str]):
        for sel in selectors:
            found = element.select_one(sel)
            if found:
                return found
        return None

    @staticmethod
    def _extract_text(element, selectors: list[str]) -> str:
        el = KleinanzeigenScraper._select_first(element, selectors)
        return el.get_text(strip=True) if el else ""

    @staticmethod
    def _extract_title(article, link_el) -> str:
        for sel in ['[class*="ellipsis"]', "h2", "h3", "a"]:
            title_el = link_el.select_one(sel) if link_el else None
            if title_el:
                return title_el.get_text(strip=True)
        return link_el.get_text(strip=True) if link_el else ""

    def _extract_price(self, article) -> tuple[float, bool]:
        price_text = self._extract_text(article, _PRICE_SELECTORS)
        if not price_text:
            return 0.0, False
        is_vb = bool(_VB_PATTERN.search(price_text))
        raw = price_text.replace("\xa0", " ").replace("€", "").strip()
        raw = _VB_PATTERN.sub("", raw).strip()

        # Find the numeric part: handle "1.234,56" (DE) or "1,234.56" (EN) or "1234,56" or "35"
        m = re.search(r"[\d.,\s]+", raw)
        if not m:
            return 0.0, is_vb
        num_str = m.group(0).strip().replace(" ", "")

        # Determine format by looking at the last 3 characters
        last_comma = num_str.rfind(",")
        last_dot = num_str.rfind(".")
        if last_comma > last_dot and last_comma == len(num_str) - 3:
            # German: "1.234,56" → comma is decimal
            num_str = num_str.replace(".", "").replace(",", ".")
        elif last_dot > last_comma and last_dot == len(num_str) - 3:
            # English: "1,234.56" → dot is decimal
            num_str = num_str.replace(",", "")
        elif last_comma > last_dot:
            # "1234,56" → comma is decimal
            num_str = num_str.replace(",", ".")
        elif last_dot > last_comma:
            # "1234.56" → dot is decimal
            pass
        else:
            # No separators or ambiguous: remove all non-digits
            num_str = re.sub(r"[^0-9]", "", num_str)

        try:
            val = float(num_str)
            if val <= 0 or val > 500000:
                return 0.0, is_vb
            return val, is_vb
        except ValueError:
            return 0.0, is_vb

    def _extract_rating(self, article) -> float:
        for sel in _RATING_SELECTORS:
            el = article.select_one(sel)
            if el:
                text = el.get_text(strip=True).lower()
                if "top" in text:
                    return 100.0
                if "ok" in text or "okay" in text:
                    return 85.0
                if "zuverlässig" in text:
                    return 90.0
        # Check for "TOP" badge images
        top_img = article.select_one('img[alt*="TOP"], img[alt*="Bewertung"]')
        if top_img:
            return 100.0
        return 0.0

    def _extract_condition(self, description: str, title: str) -> str:
        combined = f"{title} {description}".lower()
        if any(w in combined for w in ("neu", "ovp", "originalverpackt", "unbenutzt")):
            return "Neu"
        if any(w in combined for w in ("sehr gut", "top zustand", "einwandfrei")):
            return "Sehr gut"
        if any(w in combined for w in ("gut", "gebraucht")):
            return "Gebraucht"
        if any(w in combined for w in ("defekt", "kaputt", "bastler", "ersatzteile")):
            return "Defekt"
        return ""
