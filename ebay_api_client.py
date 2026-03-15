#!/usr/bin/env python3
"""
eBay Official Browse API Client
Uses the eBay Browse API (OAuth client-credentials flow) to search listings
and returns data in the same schema as the legacy EbayScraper.
"""

import base64
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Mapping eBay API conditionId → human-readable English label understood by
# the existing DealAssessor (which scores on 'new', 'refurbished', 'used', etc.)
_CONDITION_ID_MAP: Dict[str, str] = {
    "1000": "New",
    "1500": "New – Other",
    "1750": "New with defects",
    "2000": "Manufacturer Refurbished",
    "2500": "Seller Refurbished",
    "3000": "Used",
    "4000": "Very Good",
    "5000": "Good",
    "6000": "Acceptable",
    "7000": "For parts or not working",
}

# Mapping eBay marketplace ID → locale settings.
# Keys:
#   language — value for the HTTP Accept-Language request header.  Tells eBay
#              which language to use for item titles, category names, and other
#              localised metadata in API responses.
#   country  — ISO-3166 alpha-2 country code.  Used for the deliveryCountry
#              filter to restrict results to items that ship domestically within
#              the target market, and for the X-EBAY-C-ENDUSERCTX header.
#   locale   — value for the X-EBAY-C-LOCALE request header.  This is the
#              primary signal eBay's Browse API uses to return content in the
#              correct regional language.  Format: <language>_<COUNTRY>.
_MARKETPLACE_LOCALE_MAP: Dict[str, Dict[str, str]] = {
    "EBAY_AT": {"language": "de-AT,de;q=0.9", "country": "AT", "locale": "de_AT"},
    "EBAY_AU": {"language": "en-AU,en;q=0.9", "country": "AU", "locale": "en_AU"},
    "EBAY_BE": {"language": "nl-BE,nl;q=0.9,fr-BE;q=0.8", "country": "BE", "locale": "nl_BE"},
    "EBAY_CA": {"language": "en-CA,en;q=0.9", "country": "CA", "locale": "en_CA"},
    "EBAY_CH": {"language": "de-CH,de;q=0.9", "country": "CH", "locale": "de_CH"},
    "EBAY_DE": {"language": "de-DE,de;q=0.9", "country": "DE", "locale": "de_DE"},
    "EBAY_ES": {"language": "es-ES,es;q=0.9", "country": "ES", "locale": "es_ES"},
    "EBAY_FR": {"language": "fr-FR,fr;q=0.9", "country": "FR", "locale": "fr_FR"},
    "EBAY_GB": {"language": "en-GB,en;q=0.9", "country": "GB", "locale": "en_GB"},
    "EBAY_HK": {"language": "zh-HK,zh;q=0.9,en;q=0.8", "country": "HK", "locale": "zh_HK"},
    "EBAY_IE": {"language": "en-IE,en;q=0.9", "country": "IE", "locale": "en_IE"},
    "EBAY_IN": {"language": "en-IN,en;q=0.9", "country": "IN", "locale": "en_IN"},
    "EBAY_IT": {"language": "it-IT,it;q=0.9", "country": "IT", "locale": "it_IT"},
    "EBAY_MY": {"language": "en-MY,en;q=0.9", "country": "MY", "locale": "en_MY"},
    "EBAY_NL": {"language": "nl-NL,nl;q=0.9", "country": "NL", "locale": "nl_NL"},
    "EBAY_PH": {"language": "en-PH,en;q=0.9", "country": "PH", "locale": "en_PH"},
    "EBAY_PL": {"language": "pl-PL,pl;q=0.9", "country": "PL", "locale": "pl_PL"},
    "EBAY_SG": {"language": "en-SG,en;q=0.9", "country": "SG", "locale": "en_SG"},
    "EBAY_US": {"language": "en-US,en;q=0.9", "country": "US", "locale": "en_US"},
}


class EbayApiClient:
    """eBay Browse API search client.

    Authenticates via OAuth2 application-level client credentials and calls
    the ``/buy/browse/v1/item_summary/search`` endpoint.  Results are
    normalised to the same dict schema produced by :class:`EbayScraper` so
    that the rest of the application (assessors, database, frontend) requires
    no changes.

    Environment variables
    ----------------------
    EBAY_CLIENT_ID       — eBay developer application Client ID (required)
    EBAY_CLIENT_SECRET   — eBay developer application Client Secret (required)
    EBAY_MARKETPLACE_ID  — eBay marketplace (default: ``EBAY_DE``).
                           Controls the regional catalogue, ``Accept-Language``
                           header, ``X-EBAY-C-LOCALE`` header, and
                           ``deliveryCountry`` filter so that results come from
                           the correct national eBay site and are returned in
                           the local language.
    EBAY_ENVIRONMENT     — ``production`` (default) or ``sandbox``

    .. note::
        Item titles and descriptions are stored in eBay in the language chosen
        by the seller at listing time.  Even with all locale headers set
        correctly, items that were listed in English by their sellers will
        still be returned with English titles.  The locale headers maximize the
        proportion of native-language results but cannot override seller-entered
        content.
    """

    _OAUTH_PATH = "/identity/v1/oauth2/token"
    _SEARCH_PATH = "/buy/browse/v1/item_summary/search"
    _MARKETPLACE_INSIGHTS_PATH = "/buy/marketplace_insights/v1_beta/item_summary/search"
    _SCOPE = "https://api.ebay.com/oauth/api_scope"

    # Token refresh 60 s before expiry to avoid using a stale token.
    _TOKEN_REFRESH_BUFFER = 60

    def __init__(self) -> None:
        self.client_id: str = os.environ.get("EBAY_CLIENT_ID", "").strip()
        self.client_secret: str = os.environ.get("EBAY_CLIENT_SECRET", "").strip()
        self.marketplace_id: str = os.environ.get("EBAY_MARKETPLACE_ID", "EBAY_DE").strip().upper()

        env = os.environ.get("EBAY_ENVIRONMENT", "production").strip().lower()
        if env == "sandbox":
            self._base_url = "https://api.sandbox.ebay.com"
        else:
            self._base_url = "https://api.ebay.com"

        # Resolve locale settings from the marketplace ID.
        _locale = _MARKETPLACE_LOCALE_MAP.get(self.marketplace_id)
        if _locale is None:
            logger.warning(
                "Unknown EBAY_MARKETPLACE_ID %r — falling back to EBAY_DE locale. "
                "Supported marketplaces: %s",
                self.marketplace_id,
                ", ".join(sorted(_MARKETPLACE_LOCALE_MAP)),
            )
            _locale = _MARKETPLACE_LOCALE_MAP["EBAY_DE"]
        self.accept_language: str = _locale["language"]
        self.delivery_country: str = _locale["country"]
        self.locale: str = _locale["locale"]

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self.session = requests.Session()

        logger.info(
            "EbayApiClient: marketplace=%s locale=%s language=%s country=%s env=%s",
            self.marketplace_id, self.locale, self.accept_language, self.delivery_country, env,
        )

    # ── Public interface ───────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when both required credentials are present."""
        return bool(self.client_id and self.client_secret)

    def search(self, query: str, max_results: int = 50) -> Tuple[List[Dict], List[str]]:
        """Search eBay via the Browse API.

        Returns a ``(deals, errors)`` tuple that matches the contract of
        :meth:`EbayScraper.search` so the two engines are interchangeable.
        """
        errors: List[str] = []

        if not self.is_configured:
            errors.append(
                "eBay API credentials are not configured. "
                "Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET in your environment."
            )
            return [], errors

        # Obtain a valid access token.
        try:
            token = self._get_access_token()
        except requests.exceptions.HTTPError as exc:
            msg = f"eBay API authentication failed: {exc}"
            logger.error(msg)
            errors.append(msg)
            return [], errors
        except requests.exceptions.Timeout:
            msg = "eBay OAuth token request timed out"
            logger.error(msg)
            errors.append(msg)
            return [], errors
        except requests.exceptions.ConnectionError as exc:
            msg = f"eBay OAuth connection error: {exc}"
            logger.error(msg)
            errors.append(msg)
            return [], errors

        # Call the search endpoint.
        url = self._base_url + self._SEARCH_PATH
        # Build the filter string:
        #   itemLocationCountry — restricts results to items *physically located*
        #                         in the target country (e.g. DE).  This is the
        #                         primary filter that ensures "Germany-only" deals.
        #   deliveryCountry     — additionally restricts to items that ship to the
        #                         target country, preventing cross-border listings
        #                         that technically deliver to DE but originate abroad.
        api_filter = (
            f"itemLocationCountry:{self.delivery_country},"
            f"deliveryCountry:{self.delivery_country}"
        )
        params = {
            "q": query,
            "limit": min(max(1, max_results), 200),
            "sort": "newlyListed",
            "filter": api_filter,
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
            # Tell eBay which language to use for item titles and metadata.
            "Accept-Language": self.accept_language,
            # X-EBAY-C-LOCALE is the primary signal the Browse API uses to return
            # content in the correct regional language (format: language_COUNTRY).
            "X-EBAY-C-LOCALE": self.locale,
            # Provide contextual location so eBay routes to the correct regional
            # catalogue and returns localised pricing/shipping.
            "X-EBAY-C-ENDUSERCTX": f"contextualLocation=country%3D{self.delivery_country}",
            "Content-Type": "application/json",
        }

        logger.info(
            "eBay Browse API search: q=%r limit=%d marketplace=%s locale=%s",
            query, params["limit"], self.marketplace_id, self.locale,
        )

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=15)
        except requests.exceptions.Timeout:
            msg = "eBay Browse API request timed out after 15 seconds"
            logger.error(msg)
            errors.append(msg)
            return [], errors
        except requests.exceptions.ConnectionError as exc:
            msg = f"eBay Browse API connection error: {exc}"
            logger.error(msg)
            errors.append(msg)
            return [], errors

        logger.info("eBay Browse API HTTP %d %s", response.status_code, response.reason)

        if not response.ok:
            try:
                err_body = response.json()
                api_msg = "; ".join(
                    e.get("message", "") for e in err_body.get("errors", [])
                ) or response.reason
            except Exception:
                api_msg = response.reason
            msg = f"eBay Browse API error {response.status_code}: {api_msg}"
            logger.error(msg)
            errors.append(msg)
            if response.status_code == 401:
                # Invalidate cached token so it will be refreshed on the next call.
                self._token = None
                self._token_expires_at = 0.0
                errors.append(
                    "Authentication token was rejected. Check EBAY_CLIENT_ID and "
                    "EBAY_CLIENT_SECRET and ensure your application has the "
                    "required Browse API scopes."
                )
            elif response.status_code == 429:
                errors.append(
                    "eBay API rate limit exceeded. Wait before making another request."
                )
            return [], errors

        try:
            body = response.json()
        except Exception as exc:
            msg = f"Failed to parse eBay API response as JSON: {exc}"
            logger.error(msg)
            errors.append(msg)
            return [], errors

        raw_items = body.get("itemSummaries", [])
        total = body.get("total", len(raw_items))
        logger.info("eBay Browse API returned %d/%d items", len(raw_items), total)

        if not raw_items:
            # Provide a helpful diagnostic when zero items come back.
            warnings = body.get("warnings", [])
            if warnings:
                for w in warnings:
                    errors.append(f"eBay API warning: {w.get('message', w)}")
            else:
                errors.append(
                    f"eBay Browse API returned 0 results for query {query!r}. "
                    "Try a different search term."
                )
            return [], errors

        deals: List[Dict] = []
        parse_errors = 0
        for item in raw_items:
            try:
                deal = self._normalize_item(item)
                if deal:
                    deals.append(deal)
            except Exception as exc:
                parse_errors += 1
                logger.warning("Failed to parse API item %r: %s", item.get("itemId"), exc)

        if parse_errors:
            errors.append(f"{parse_errors} item(s) from the eBay API could not be parsed and were skipped.")

        logger.info("Returning %d normalised deals (%d errors)", len(deals), len(errors))
        return deals, errors

    def get_median_sold_price(
        self, query: str, max_results: int = 10
    ) -> "Tuple[Optional[float], str, List[str]]":
        """Return the median price for *query* from recently sold or active listings.

        Tries the eBay Marketplace Insights API (sold/completed listings) first.
        Falls back to the Browse API (active listings) when the Insights API is
        unavailable or returns no results.

        Returns a 3-tuple ``(median_price, source_label, errors)`` where:

        * ``median_price`` – median sale/ask price in the listing currency, or
          ``None`` when no results are found.
        * ``source_label`` – one of ``"sold_listings"`` or ``"active_listings"``
          indicating which data source was used; ``"none"`` when both fail.
        * ``errors`` – list of warning/error strings (may be non-empty even when
          a price is returned, e.g. because the primary source failed but the
          fallback succeeded).
        """
        errors: List[str] = []

        if not self.is_configured:
            errors.append("eBay API credentials not configured — cannot fetch item prices.")
            return None, "none", errors

        try:
            token = self._get_access_token()
        except Exception as exc:
            errors.append(f"eBay API auth failed while fetching price for {query!r}: {exc}")
            return None, "none", errors

        common_headers = {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
            "Accept-Language": self.accept_language,
            "X-EBAY-C-LOCALE": self.locale,
            "X-EBAY-C-ENDUSERCTX": f"contextualLocation=country%3D{self.delivery_country}",
            "Content-Type": "application/json",
        }
        api_filter = (
            f"itemLocationCountry:{self.delivery_country},"
            f"deliveryCountry:{self.delivery_country}"
        )
        params = {
            "q": query,
            "limit": min(max(1, max_results), 50),
            "filter": api_filter,
        }

        # ── Attempt 1: Marketplace Insights API (sold/completed listings) ──
        try:
            insights_url = self._base_url + self._MARKETPLACE_INSIGHTS_PATH
            resp = self.session.get(
                insights_url, headers=common_headers, params=params, timeout=10
            )
            if resp.ok:
                body = resp.json()
                prices = self._extract_prices_from_items(body.get("itemSales", []))
                if prices:
                    median = sorted(prices)[len(prices) // 2]
                    logger.info(
                        "eBay Insights API: %d sold results for %r, median=€%.2f",
                        len(prices), query, median,
                    )
                    return median, "sold_listings", errors
                # No results from Insights API — fall through to Browse.
                errors.append(
                    f"eBay Insights API returned 0 sold results for {query!r}; "
                    "falling back to active listings."
                )
            elif resp.status_code in (403, 404):
                # Marketplace Insights scope not granted or endpoint unavailable.
                errors.append(
                    f"eBay Insights API unavailable (HTTP {resp.status_code}); "
                    "falling back to active listings."
                )
            else:
                errors.append(
                    f"eBay Insights API error {resp.status_code} for {query!r}; "
                    "falling back to active listings."
                )
        except Exception as exc:
            errors.append(
                f"eBay Insights API request failed for {query!r}: {exc}; "
                "falling back to active listings."
            )

        # ── Attempt 2: Browse API (active listings, used as price proxy) ──
        try:
            browse_url = self._base_url + self._SEARCH_PATH
            resp = self.session.get(
                browse_url, headers=common_headers, params=params, timeout=10
            )
            if resp.ok:
                body = resp.json()
                raw_items = body.get("itemSummaries", [])
                prices = self._extract_prices_from_items(raw_items)
                if prices:
                    median = sorted(prices)[len(prices) // 2]
                    logger.info(
                        "eBay Browse API: %d active results for %r, median=€%.2f",
                        len(prices), query, median,
                    )
                    return median, "active_listings", errors
                errors.append(f"eBay Browse API returned 0 results for {query!r}.")
            else:
                errors.append(
                    f"eBay Browse API error {resp.status_code} for {query!r}."
                )
        except Exception as exc:
            errors.append(f"eBay Browse API request failed for {query!r}: {exc}")

        return None, "none", errors

    # ── Private helpers ────────────────────────────────────────────────────

    def _get_access_token(self) -> str:
        """Return a valid OAuth application access token, refreshing if needed."""
        now = time.monotonic()
        if self._token and now < self._token_expires_at:
            return self._token

        logger.info("Requesting new eBay OAuth application token")

        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("ascii")

        response = self.session.post(
            self._base_url + self._OAUTH_PATH,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": self._SCOPE,
            },
            timeout=10,
        )
        response.raise_for_status()

        token_data = response.json()
        self._token = token_data["access_token"]
        expires_in = int(token_data.get("expires_in", 7200))
        self._token_expires_at = now + expires_in - self._TOKEN_REFRESH_BUFFER
        logger.info("eBay OAuth token obtained (expires in %ds)", expires_in)
        return self._token

    @staticmethod
    def _extract_prices_from_items(items: list) -> List[float]:
        """Extract numeric prices from a list of Browse/Insights API item dicts."""
        prices: List[float] = []
        for item in items:
            # Browse API uses 'price'; Marketplace Insights uses 'lastSoldPrice'
            price_obj = item.get("lastSoldPrice") or item.get("price") or {}
            try:
                val = float(price_obj.get("value", 0))
                if val > 0:
                    prices.append(val)
            except (TypeError, ValueError):
                pass
        return prices

    def _normalize_item(self, item: dict) -> Optional[Dict]:
        """Map a Browse API ``itemSummary`` object to our internal deal dict.

        Returns ``None`` when the item lacks both a title and a URL (i.e. is
        not usable data).
        """
        title = (item.get("title") or "").strip()
        url = item.get("itemWebUrl", "").strip()

        if not title and not url:
            return None

        # ── Price ──────────────────────────────────────────────────────────
        price_obj = item.get("price") or {}
        try:
            price = float(price_obj.get("value", 0))
        except (TypeError, ValueError):
            price = 0.0

        # ── Condition ──────────────────────────────────────────────────────
        condition_id = str(item.get("conditionId", ""))
        condition_text = item.get("condition", "")
        condition = _CONDITION_ID_MAP.get(condition_id, condition_text or "Unknown")

        # ── Seller ─────────────────────────────────────────────────────────
        seller = item.get("seller") or {}
        try:
            seller_rating = float(seller.get("feedbackPercentage", 0))
        except (TypeError, ValueError):
            seller_rating = 0.0

        # ── Shipping ───────────────────────────────────────────────────────
        shipping = self._parse_shipping(item.get("shippingOptions") or [])

        # ── Item location ──────────────────────────────────────────────────
        # itemLocation holds the physical location where the item is stored.
        # We expose country + city so the UI can show "Germany 🇩🇪" on deal cards.
        item_location_obj = item.get("itemLocation") or {}
        item_location_country = (item_location_obj.get("country") or "").strip().upper()
        item_location_city = (item_location_obj.get("city") or "").strip()
        if item_location_country and item_location_city:
            item_location = f"{item_location_city}, {item_location_country}"
        elif item_location_country:
            item_location = item_location_country
        elif item_location_city:
            item_location = item_location_city
        else:
            item_location = ""

        # ── Trending ───────────────────────────────────────────────────────
        # The Browse API exposes a "topRatedBuyingExperience" flag and
        # a "priorityListing" flag; treat either as "trending".
        is_trending = bool(
            item.get("topRatedBuyingExperience")
            or item.get("priorityListing")
            or item.get("watchCount", 0) > 10
        )

        # ── Description ────────────────────────────────────────────────────
        # Browse API may return a short description; use it when available.
        description = (item.get("shortDescription") or "").strip()

        # ── Seller count (available/sold quantity) ──────────────────────────
        # "X verfügbar Y verkauft" style info that can indicate bait-and-switch
        # scams where one item from a bundle is sold per transaction.
        try:
            _qty_left_raw = item.get("availableQuantity")
            if _qty_left_raw is None:
                _qty_left_raw = item.get("quantityLeft")
            qty_left = int(_qty_left_raw) if _qty_left_raw is not None else 0
        except (TypeError, ValueError):
            qty_left = 0
        try:
            _qty_sold_raw = item.get("soldQuantity")
            if _qty_sold_raw is None:
                _qty_sold_raw = item.get("itemSoldCount")
            qty_sold = int(_qty_sold_raw) if _qty_sold_raw is not None else 0
        except (TypeError, ValueError):
            qty_sold = 0
        if qty_left > 0 and qty_sold > 0:
            seller_count = f"{qty_left} verfügbar, {qty_sold} verkauft"
        elif qty_left > 0:
            seller_count = f"{qty_left} verfügbar"
        elif qty_sold > 0:
            seller_count = f"{qty_sold} verkauft"
        else:
            seller_count = ""

        # ── Images ─────────────────────────────────────────────────────────
        image_urls: List[str] = []
        primary_image = item.get("image") or item.get("thumbnailImages", [{}])[0]
        if isinstance(primary_image, dict) and primary_image.get("imageUrl"):
            image_urls.append(primary_image["imageUrl"])
        for img in item.get("additionalImages") or []:
            if isinstance(img, dict) and img.get("imageUrl"):
                image_urls.append(img["imageUrl"])

        # ── Listing date ───────────────────────────────────────────────────
        # itemCreationDate is an ISO-8601 string like "2024-03-01T10:00:00.000Z".
        listing_date: Optional[str] = (item.get("itemCreationDate") or "").strip() or None

        return {
            "title": title or "Unknown",
            "price": price,
            "condition": condition,
            "seller_rating": seller_rating,
            "url": url,
            "shipping": shipping,
            "is_trending": is_trending,
            # Physical location of the item (country code + optional city).
            # Set by itemLocationCountry filter; e.g. "Berlin, DE" or "DE".
            "item_location": item_location,
            "image_urls": image_urls,
            # Short description text from the listing (may be empty).
            "description": description,
            # Quantity/sold info formatted as a human-readable string, e.g.
            # "4 verfügbar, 1 verkauft" — a key bait-and-switch scam indicator.
            "seller_count": seller_count,
            # ISO-8601 date when the listing was created on eBay (may be None
            # for older API responses or the legacy scraper).
            "listing_date": listing_date,
        }

    @staticmethod
    def _parse_shipping(shipping_options: list) -> str:
        """Convert Browse API shippingOptions list to a human-readable string."""
        if not shipping_options:
            return "N/A"

        option = shipping_options[0]
        cost_type = (option.get("shippingCostType") or "").upper()

        if cost_type in ("FREE", "FREE_SHIPPING"):
            return "Free"

        cost_obj = option.get("shippingCost") or {}
        try:
            amount = float(cost_obj.get("value", 0))
        except (TypeError, ValueError):
            amount = 0.0

        if amount == 0.0:
            return "Free"

        currency = (cost_obj.get("currency") or "EUR").upper()
        symbol = "€" if currency == "EUR" else currency
        return f"{symbol}{amount:.2f}"
