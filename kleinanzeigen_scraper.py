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
        articles = soup.select("article.aditem")
        if not articles:
            articles = soup.select("li.ad-listitem")
        if not articles:
            articles = soup.select('[class*="aditem"]')

        for article in articles[:max_results]:
            deal = self._parse_article(article)
            if deal:
                all_deals.append(deal)

        return all_deals, errors

    def _parse_article(self, article) -> dict | None:
        try:
            link_el = article.select_one('a[class*="ellipsis"]') or article.select_one("a[href]")
            if not link_el:
                return None
            href = link_el.get("href", "")
            if not href:
                return None
            if href.startswith("/"):
                href = _KLEINANZEIGEN_BASE + href

            title_el = link_el.select_one('[class*="ellipsis"]') or link_el
            title = (title_el.get_text(strip=True) if title_el else "") or link_el.get_text(strip=True) or ""
            if not title:
                return None

            price_el = article.select_one('[class*="aditem-main--middle--price"]') or article.select_one(
                "[class*='price']"
            )
            price_text = price_el.get_text(strip=True) if price_el else "0"
            price = self._parse_price(price_text)

            loc_el = article.select_one('[class*="aditem-main--top--left"]') or article.select_one("[class*='top']")
            location = loc_el.get_text(strip=True) if loc_el else ""

            desc_el = article.select_one('[class*="aditem-main--middle--description"]') or article.select_one(
                "[class*='description']"
            )
            description = desc_el.get_text(strip=True) if desc_el else ""

            return {
                "title": title[:300],
                "price": price,
                "condition": "",
                "seller_rating": 0.0,
                "url": href,
                "shipping": "",
                "is_trending": False,
                "item_location": location,
                "description": description[:500],
                "seller_count": "",
                "listing_date": "",
                "image_urls": [],
                "image_issues": ["no_images"],
            }
        except Exception:
            return None

    @staticmethod
    def _parse_price(text: str) -> float:
        text = text.replace("\xa0", " ").replace("€", "").replace(",", ".").strip()
        numbers = re.findall(r"[\d.]+", text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        return 0.0
