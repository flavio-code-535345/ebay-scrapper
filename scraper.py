#!/usr/bin/env python3
"""
eBay Web Scraper Engine
Handles fetching and parsing eBay listings (defaults to ebay.de)
"""

import logging
import re
import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Tuple
import random

logger = logging.getLogger(__name__)

# German condition keywords used on ebay.de for best-effort text matching
_DE_CONDITION_KEYWORDS = [
    'Neu',
    'Gebraucht',
    'Generalüberholt',
    'Akzeptabler Zustand',
    'Sehr guter Zustand',
    'Guter Zustand',
    'Für Ersatzteile',
    'Zustand:',
]

# German shipping keywords
_DE_SHIPPING_KEYWORDS = ['versand', 'lieferung', 'kostenlos', 'gratis', 'free']

# German / eBay trending / popularity keywords
_DE_TRENDING_KEYWORDS = {'beliebt', 'sehr beliebt', 'angesagt', 'hot', 'trending'}

# Maximum text length thresholds used to avoid accidentally matching large wrapper text
_MAX_CONDITION_TEXT_LEN = 80
_MAX_SELLER_TEXT_LEN = 120
_MAX_SHIPPING_TEXT_LEN = 100

# Text phrases (lower-case) that indicate a listing has variant/dropdown selectors.
# Listings with these phrases are NOT fixed-price bundles and are excluded.
_VARIANT_TEXT_PATTERNS = [
    # German
    'mehrere ausführungen',
    'ausführung wählen',
    'farbe wählen',
    'größe wählen',
    'modell wählen',
    'variante wählen',
    'auswahl treffen',
    'variation verfügbar',
    'varianten verfügbar',
    # English (fallback for mixed-language results)
    'available in multiple',
    'choose your',
    'select a color',
    'select a size',
    'select your',
    'color:',
    'size:',
    'style:',
]

# Maximum text length when scanning for variant text patterns.
_MAX_VARIANT_TEXT_LEN = 80

# eBay injects short badge/label spans directly inside the title element.
# These must be stripped out to recover the actual listing title.
# All entries must be lower-cased for case-insensitive comparison.
_TITLE_NOISE_PHRASES = frozenset({
    'neues angebot',        # "New Listing" badge – German
    'new listing',          # "New Listing" badge – English
    'gesponsert',           # "Sponsored" badge – German
    'sponsored',            # "Sponsored" badge – English
    'top-rated plus',
    'top-bewerteter anbieter',
})

# Regex that matches eBay CDN image URL size codes indicating very low resolution
# (below 230 px wide) – e.g. ``s-l140``, ``s-l225``.  These are often placeholder
# thumbnails rather than real product shots.
_LOW_RES_URL_RE = re.compile(r's-l(1[0-9]{2}|2[0-2][0-9])\b')


class EbayScraper:
    def __init__(self):
        self.base_url = "https://www.ebay.de/sch/i.html"
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/122.0.0.0 Safari/537.36'
            ),
            'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        self.session = requests.Session()

    def search(self, query: str, max_results: int = 50) -> Tuple[List[Dict], List[str]]:
        """Search eBay for items matching query.

        Returns a tuple of (deals, errors) where errors is a list of
        human-readable strings describing any problems encountered.
        """
        errors: List[str] = []

        try:
            params = {
                '_nkw': query,
                '_sop': '12',  # Sort by newly listed
                'LH_ItemCondition': '3000|3000|1000',  # All conditions
                'rt': 'nc'
            }

            logger.info("Searching eBay for %r (max_results=%d)", query, max_results)

            try:
                response = self.session.get(
                    self.base_url, params=params, headers=self.headers, timeout=10
                )
            except requests.exceptions.Timeout:
                msg = "HTTP request timed out after 10 seconds"
                logger.error(msg)
                errors.append(msg)
                return [], errors
            except requests.exceptions.ConnectionError as exc:
                msg = f"Connection error: {exc}"
                logger.error(msg)
                errors.append(msg)
                return [], errors

            logger.info("HTTP %d %s", response.status_code, response.reason)

            if not response.ok:
                msg = f"HTTP {response.status_code}: {response.reason}"
                logger.error("eBay returned an error – %s", msg)
                errors.append(msg)
                if response.status_code == 403:
                    errors.append(
                        "Access denied – eBay may be blocking automated requests. "
                        "Try again later or use a different network."
                    )
                elif response.status_code == 429:
                    errors.append(
                        "Rate limited – too many requests sent in a short time. "
                        "Wait a few minutes before searching again."
                    )
                return [], errors

            soup = BeautifulSoup(response.content, 'html.parser')
            deals: List[Dict] = []

            # ── Selector strategy ─────────────────────────────────────────────
            # eBay uses <li class="s-item …"> inside <ul class="srp-results …">.
            # eBay periodically changes or adds class names, so we try a cascade
            # of increasingly broad selectors and log which one fired.

            # 1. Primary: class-based s-item selector (covers li AND div variants)
            items = soup.select('li.s-item, div.s-item')
            selector_used = 'li.s-item / div.s-item'

            if not items:
                # 2. Fallback A: any direct <li> children of the srp-results
                #    container that contain a product link.  This handles the
                #    case where eBay removed or renamed the s-item class while
                #    keeping the overall srp-results wrapper intact.
                srp_container = soup.find(class_='srp-results')
                if srp_container:
                    candidate_lis = srp_container.find_all('li', recursive=False)
                    items = [
                        li for li in candidate_lis
                        if li.find('a', class_='s-item__link')
                        or li.find('a', href=lambda h: h and '/itm/' in h)
                    ]
                    if items:
                        selector_used = 'srp-results > li (fallback A)'
                        logger.warning(
                            "Primary 's-item' selector returned 0 results; "
                            "fell back to 'srp-results > li' — eBay likely changed "
                            "the item class name.  This is a markup/selector change, "
                            "NOT a connectivity or ban problem."
                        )

            if not items:
                # 3. Fallback B: any element (any tag) carrying the s-item__wrapper
                #    class, which has been stable across several eBay redesigns.
                items = soup.select('.s-item__wrapper')
                if items:
                    selector_used = '.s-item__wrapper (fallback B)'
                    logger.warning(
                        "Fell back to '.s-item__wrapper' selector — eBay may have "
                        "changed their markup.  This is a selector/markup issue, NOT "
                        "a connectivity or ban problem."
                    )

            logger.info(
                "BeautifulSoup found %d raw item elements (selector: %s)",
                len(items), selector_used,
            )

            if not items:
                # Gather diagnostic context so developers can tell whether the
                # page loaded at all vs. the selectors simply no longer match.
                srp_container = soup.find(class_='srp-results')
                page_title = (
                    soup.title.string.strip()
                    if soup.title and soup.title.string
                    else "(no <title>)"
                )
                html_preview = response.text[:300].replace('\n', ' ')

                if srp_container:
                    diag = (
                        "An 'srp-results' container was found on the page, which means "
                        "eBay returned a valid search results page. None of the known "
                        "item selectors ('li.s-item', 'div.s-item', 'srp-results > li', "
                        "'.s-item__wrapper') matched any elements — eBay has likely "
                        "changed their item markup.  This is a selector/markup issue, "
                        "NOT a connectivity or ban problem."
                    )
                else:
                    diag = (
                        f"No 'srp-results' container and no item elements were found. "
                        f"Page title: \"{page_title}\". "
                        f"eBay may have significantly restructured their search results page "
                        f"or returned an unexpected page (CAPTCHA, login wall, etc.)."
                    )

                msg = (
                    "BeautifulSoup found 0 item elements after trying all known "
                    "selectors ('li.s-item', 'srp-results > li', '.s-item__wrapper'). "
                    "eBay has likely changed their HTML structure — this is a markup/"
                    "selector issue, not a connectivity or ban problem. "
                    "The scraper's selectors need to be updated to match the new page layout."
                )
                logger.warning(msg)
                logger.debug("Zero-item diagnostic: %s HTML preview: %r", diag, html_preview)
                errors.append(msg)
                errors.append(diag)

            parse_errors = 0
            for item in items[:max_results]:
                try:
                    deal = self._parse_item(item)
                    if deal:
                        deals.append(deal)
                except Exception as exc:
                    parse_errors += 1
                    logger.warning("Error parsing item element: %s", exc, exc_info=True)
                    continue

            if parse_errors:
                errors.append(f"{parse_errors} item(s) could not be parsed and were skipped.")

            logger.info("Returning %d deals (%d errors)", len(deals), len(errors))
            time.sleep(random.uniform(1, 3))  # Rate limiting
            return deals, errors

        except Exception as exc:
            msg = f"Unexpected error during search: {exc}"
            logger.error(msg, exc_info=True)
            errors.append(msg)
            return [], errors

    def _parse_item(self, item_element) -> Dict:
        """Parse individual item element into deal dictionary.

        Uses a cascade of selectors for each field so that parsing continues
        to work when eBay tweaks their class names between the well-known
        `s-item__*` names and any new variants they introduce.
        """
        try:
            # Extract title – eBay uses <h3> in newer layouts, <h2> in older ones;
            # using a CSS class selector avoids the tag dependency entirely.
            # eBay also injects badge spans (e.g. "Neues Angebot", "Gesponsert")
            # as child elements inside the title wrapper – strip those out so only
            # the real listing name is kept.
            title_elem = (
                item_element.select_one('.s-item__title')
                or item_element.select_one('[data-testid="item-card-title"]')  # newer eBay layout
                or item_element.find('h3')
                or item_element.find('h2')
            )
            if title_elem:
                # Join all non-empty text nodes, skipping known eBay badge labels.
                title_parts = []
                for s in title_elem.strings:
                    stripped = s.strip()
                    if stripped and stripped.lower() not in _TITLE_NOISE_PHRASES:
                        title_parts.append(stripped)
                title = ' '.join(title_parts).strip()
                if not title:
                    title = "Unknown"
            else:
                title = "Unknown"

            # Skip eBay's placeholder cards (English and German variants).
            _skip = {"shop on ebay", "zu ebay", "results matching fewer words"}
            if title.lower() in _skip or title.lower().startswith("ergebnisse für"):
                return None

            # Skip dropdown/variant listings – these are NOT fixed-price bundles.
            if self._is_dropdown_variant(item_element):
                logger.debug("Skipping dropdown/variant listing: %r", title)
                return None

            # Extract price – try stable class first, then any element containing
            # a currency symbol (€ for ebay.de, $ for .com).
            price_elem = item_element.find(class_='s-item__price')
            if not price_elem:
                price_elem = item_element.find(
                    'span', string=lambda s: s and ('€' in s or 'EUR' in s or '$' in s)
                )
            price_text = price_elem.text.strip() if price_elem else "€0,00"
            price = self._parse_price(price_text)

            # Extract condition (Zustand) with multiple fallback strategies.
            condition = self._extract_condition(item_element)

            # Extract seller rating with multiple fallback strategies.
            seller_rating = self._extract_seller_rating(item_element)

            # Extract item URL – try the dedicated link class first, then any
            # anchor that points to an individual eBay listing page (/itm/).
            link_elem = (
                item_element.find('a', class_='s-item__link')
                or item_element.find('a', href=lambda h: h and '/itm/' in h)
            )
            item_url = link_elem.get('href', '') if link_elem else ""

            # Extract shipping cost (Versand) with multiple fallback strategies.
            shipping = self._extract_shipping(item_element)

            # Check if item is trending / popular (Beliebt).
            is_trending = self._extract_trending(item_element)

            # Extract listing image URLs for AI/visual analysis.
            image_urls = self._extract_image_urls(item_element)

            # Detect any image quality issues (no extra HTTP requests needed).
            image_issues = self._detect_image_issues(image_urls)

            return {
                'title': title,
                'price': price,
                'condition': condition,
                'seller_rating': seller_rating,
                'url': item_url,
                'shipping': shipping,
                'is_trending': is_trending,
                'image_urls': image_urls,
                'image_issues': image_issues,
                'timestamp': time.time()
            }

        except Exception as exc:
            logger.warning("Error in _parse_item: %s", exc, exc_info=True)
            return None

    # ── Field-level extraction helpers ────────────────────────────────────────

    def _extract_condition(self, item_element) -> str:
        """Extract item condition (Zustand) using a cascade of strategies."""
        # 1. Known class names (stable across several eBay layouts)
        elem = (
            item_element.find(class_='SECONDARY_INFO')
            or item_element.select_one('[class*="SECONDARY_INFO"]')
            or item_element.find(class_='s-item__subtitle')
            or item_element.select_one('[class*="subtitle"]')
            or item_element.select_one('[class*="condition"]')
            or item_element.select_one('[class*="Condition"]')
        )
        if elem:
            text = elem.text.strip()
            if text:
                logger.debug("condition via class: %r", text)
                return text

        # 2. Text-based search for German condition keywords
        for node in item_element.find_all(['span', 'div', 'li']):
            text = node.text.strip()
            # Guard against pulling in large wrapper text
            if len(text) > _MAX_CONDITION_TEXT_LEN:
                continue
            for kw in _DE_CONDITION_KEYWORDS:
                if text.lower().startswith(kw.lower()):
                    logger.debug("condition via keyword %r: %r", kw, text)
                    return text

        logger.warning("Could not extract condition for item; defaulting to 'Unknown'")
        return "Unknown"

    def _extract_seller_rating(self, item_element) -> float:
        """Extract seller rating percentage using a cascade of strategies."""
        # 1. Known class names
        seller_elem = (
            item_element.find(class_='s-item__seller-info-text')
            or item_element.find(class_='s-item__seller-info')
            or item_element.select_one('[class*="seller-info"]')
            or item_element.select_one('[class*="sellerInfo"]')
        )
        if seller_elem:
            rating = self._parse_seller_rating(seller_elem.text)
            if rating > 0:
                logger.debug("seller rating via class: %.1f%%", rating)
                return rating

        # 2. Regex search for a percentage value in any short span/div.
        for node in item_element.find_all(['span', 'div']):
            text = node.text.strip()
            if len(text) > _MAX_SELLER_TEXT_LEN or '%' not in text:
                continue
            match = re.search(r'(\d{1,3}(?:[.,]\d+)?)\s*%', text)
            if match:
                try:
                    rating = float(match.group(1).replace(',', '.'))
                    if 0 < rating <= 100:
                        logger.debug("seller rating via regex: %.1f%%", rating)
                        return rating
                except ValueError:
                    pass

        logger.warning("Could not extract seller rating for item; defaulting to 0.0%%")
        return 0.0

    def _extract_shipping(self, item_element) -> str:
        """Extract shipping cost (Versand) using a cascade of strategies."""
        # 1. Known class names
        shipping_elem = (
            item_element.find(class_='s-item__shipping')
            or item_element.find(class_='s-item__logisticsCost')
            or item_element.select_one('[class*="shipping"]')
            or item_element.select_one('[class*="logisticsCost"]')
            or item_element.select_one('[class*="Shipping"]')
        )
        if shipping_elem:
            text = shipping_elem.text.strip()
            if text:
                logger.debug("shipping via class: %r", text)
                return text

        # 2. Text-based search for German shipping keywords
        for node in item_element.find_all(['span', 'div']):
            text = node.text.strip()
            if len(text) > _MAX_SHIPPING_TEXT_LEN:
                continue
            text_lower = text.lower()
            for kw in _DE_SHIPPING_KEYWORDS:
                if kw in text_lower:
                    logger.debug("shipping via keyword %r: %r", kw, text)
                    return text

        logger.warning("Could not extract shipping info for item; defaulting to 'Nicht angegeben'")
        return "Nicht angegeben"

    def _extract_trending(self, item_element) -> bool:
        """Detect trending / popular status using a cascade of strategies."""
        # 1. Known class names
        if (
            item_element.find(class_='SHOP_NEW_TAG')
            or item_element.find(class_='s-item__trending-price')
            or item_element.select_one('[class*="trending"]')
            or item_element.select_one('[class*="TRENDING"]')
            or item_element.select_one('[class*="hot"]')
            or item_element.select_one('[class*="popular"]')
        ):
            return True

        # 2. Text-based search for German popularity keywords
        for node in item_element.find_all(['span', 'div', 'mark', 'strong']):
            text = node.text.strip().lower()
            if text in _DE_TRENDING_KEYWORDS:
                logger.debug("trending via keyword: %r", text)
                return True

        return False

    def _extract_image_urls(self, item_element) -> List[str]:
        """Extract listing image URLs from the item element.

        eBay lazy-loads images using ``data-src`` / ``s-src`` attributes;
        this method checks both standard and lazy-load variants.
        """
        urls: List[str] = []
        seen: set = set()

        for img in item_element.find_all('img'):
            for attr in ('src', 'data-src', 's-src'):
                url = img.get(attr, '').strip()
                if (
                    url
                    and url.startswith('http')
                    and url not in seen
                    # Skip eBay placeholder / spacer images (GIF files)
                    and not url.lower().endswith('.gif')
                    and 's-l' in url  # eBay image CDN pattern (e.g. s-l500, s-l1600)
                ):
                    seen.add(url)
                    urls.append(url)
                    break  # one URL per <img> tag is enough

        if urls:
            logger.debug("Extracted %d image URL(s) for listing", len(urls))
        return urls

    def _is_dropdown_variant(self, item_element) -> bool:
        """Return True if the listing has dropdown/variant selectors.

        Variant/option listings (e.g. "choose colour", "select size") are NOT
        fixed-price bundles and should be excluded from deal results.  We use a
        set of CSS-class signals and text patterns that eBay injects into search-
        result cards for multi-variation listings.
        """
        # 1. CSS class signals used by eBay to mark variation/variant listings.
        if (
            item_element.find(class_='s-item__variations')
            or item_element.find(class_='s-item__variants-button')
            or item_element.select_one('[class*="variations"]')
            or item_element.select_one('[class*="variant"]')
        ):
            logger.debug("Dropdown/variant detected via CSS class")
            return True

        # 2. Rendered <select> dropdown elements inside the card.
        if item_element.find('select'):
            logger.debug("Dropdown/variant detected via <select> element")
            return True

        # 3. Text-based signals (German + English) for short inline elements.
        for node in item_element.find_all(['span', 'div', 'a']):
            text = node.get_text(strip=True).lower()
            if not text or len(text) > _MAX_VARIANT_TEXT_LEN:
                continue
            for phrase in _VARIANT_TEXT_PATTERNS:
                if phrase in text:
                    logger.debug(
                        "Dropdown/variant detected via text phrase %r: %r", phrase, text
                    )
                    return True

        return False

    def _detect_image_issues(self, image_urls: List[str]) -> List[str]:
        """Detect potential image quality issues from the URL list alone.

        Returns a (possibly empty) list of human-readable issue identifiers.
        No extra HTTP requests are made; issues are inferred from URL metadata.

        Possible identifiers returned:
        - ``"no_images"``   – the listing has no product images at all.
        """
        if not image_urls:
            return ["no_images"]

        return []

    # ── Value parsers ──────────────────────────────────────────────────────────

    def _parse_price(self, price_str: str) -> float:
        """Extract numeric price from a string, handling EUR/€ and German
        number formatting (period as thousands separator, comma as decimal)."""
        try:
            # Strip currency labels and surrounding whitespace
            clean = price_str.replace('EUR', '').replace('€', '').replace('$', '').strip()
            # Take only the first price token (handles ranges like "10,00 bis 20,00")
            clean = clean.split()[0]
            # Detect German number format: has both '.' and ',' with ',' last
            if ',' in clean and '.' in clean:
                last_comma = clean.rindex(',')
                last_dot = clean.rindex('.')
                if last_comma > last_dot:
                    # German: "1.234,56" → remove '.', replace ',' with '.'
                    clean = clean.replace('.', '').replace(',', '.')
                else:
                    # English: "1,234.56" → remove ','
                    clean = clean.replace(',', '')
            elif ',' in clean:
                # Only comma: German decimal "12,99" → "12.99"
                clean = clean.replace(',', '.')
            # Remove any remaining non-numeric characters except '.' and '-'
            clean = re.sub(r'[^\d.\-]', '', clean)
            return float(clean)
        except Exception:
            return 0.0

    def _parse_seller_rating(self, seller_str: str) -> float:
        """Extract seller rating percentage from a seller info string."""
        try:
            if '%' in seller_str:
                match = re.search(r'(\d{1,3}(?:[.,]\d+)?)\s*%', seller_str)
                if match:
                    return float(match.group(1).replace(',', '.'))
            return 0.0
        except Exception:
            return 0.0

    def get_item_details(self, item_url: str) -> Dict:
        """Fetch detailed information about specific item"""
        try:
            response = self.session.get(item_url, headers=self.headers, timeout=10)
            logger.info("get_item_details HTTP %d %s", response.status_code, response.reason)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            details = {
                'views': self._extract_views(soup),
                'watchers': self._extract_watchers(soup),
                'sold_count': self._extract_sold_count(soup),
                'time_listed': self._extract_time_listed(soup)
            }

            time.sleep(random.uniform(1, 2))
            return details

        except Exception as exc:
            logger.error("Error getting item details: %s", exc, exc_info=True)
            return {}
    
    def _extract_views(self, soup) -> int:
        """Extract view count from item page"""
        try:
            views_elem = soup.find('span', string=lambda s: s and 'views' in s.lower())
            if views_elem:
                count = views_elem.text.split()[0].replace(',', '')
                return int(count)
        except:
            pass
        return 0
    
    def _extract_watchers(self, soup) -> int:
        """Extract watcher count from item page"""
        try:
            watchers_elem = soup.find('span', string=lambda s: s and 'watchers' in s.lower())
            if watchers_elem:
                count = watchers_elem.text.split()[0].replace(',', '')
                return int(count)
        except:
            pass
        return 0
    
    def _extract_sold_count(self, soup) -> int:
        """Extract sold count from item page"""
        try:
            sold_elem = soup.find('span', string=lambda s: s and 'sold' in s.lower())
            if sold_elem:
                count = sold_elem.text.split()[0].replace(',', '')
                return int(count)
        except:
            pass
        return 0
    
    def _extract_time_listed(self, soup) -> str:
        """Extract when item was listed"""
        try:
            time_elem = soup.find('span', string=lambda s: s and 'listed' in s.lower())
            if time_elem:
                return time_elem.text.strip()
        except:
            pass
        return "Unknown"