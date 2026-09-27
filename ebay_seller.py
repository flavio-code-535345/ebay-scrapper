"""Selling on the user's own eBay account — the "Sell" page's backend.

Connects the account (OAuth authorization-code grant), copies the listing
setup from one of the seller's existing listings, looks products up by EAN,
uploads photos and publishes fixed-price listings.

Listings are created with the Trading API (``AddFixedPriceItem``), not the
Inventory API: listings made through the Inventory API can't be edited in
Seller Hub or the eBay app afterwards, and the seller manages offers and
prices there. Photos go through the Media API, product data through the
Catalog API; all of it uses the seller's own OAuth user token, stored in the
``settings`` table (never returned by any endpoint).

Everything the new listing inherits — shipping service and cost, return
policy, location, business-policy IDs, description, best-offer setting — is
copied from a listing the seller already has (``import_template``), so new
listings match the old ones exactly without guessing eBay's service codes.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field

import requests

import database
from models import canonical_listing_id

logger = logging.getLogger(__name__)

SCOPES = (
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/sell.inventory",  # Media API (photos) and Catalog API
)
_TRADING_COMPATIBILITY_LEVEL = "1477"
_SITE_ID_GERMANY = "77"
_MARKETPLACE_ID = "EBAY_DE"
_NS = "urn:ebay:apis:eBLBaseComponents"
_NSMAP = {"e": _NS}
_REQUEST_TIMEOUT = 30
_TOKEN_REFRESH_MARGIN_S = 120
_CONDITIONS_CACHE_S = 86400

_KEY_REFRESH = "ebay_seller_refresh_token"
_KEY_REFRESH_EXPIRES = "ebay_seller_refresh_expires_at"
_KEY_ACCESS = "ebay_seller_access_token"
_KEY_ACCESS_EXPIRES = "ebay_seller_access_expires_at"
_KEY_USERNAME = "ebay_seller_username"
_KEY_TEMPLATE = "ebay_listing_template"
_TOKEN_KEYS = (_KEY_REFRESH, _KEY_REFRESH_EXPIRES, _KEY_ACCESS, _KEY_ACCESS_EXPIRES, _KEY_USERNAME)

_ENVIRONMENTS = {
    "production": {
        "api": "https://api.ebay.com",
        "apim": "https://apim.ebay.com",
        "auth": "https://auth.ebay.com/oauth2/authorize",
        "trading": "https://api.ebay.com/ws/api.dll",
        "item_url": "https://www.ebay.de/itm/{}",
    },
    "sandbox": {
        "api": "https://api.sandbox.ebay.com",
        "apim": "https://apim.sandbox.ebay.com",
        "auth": "https://auth.sandbox.ebay.com/oauth2/authorize",
        "trading": "https://api.sandbox.ebay.com/ws/api.dll",
        "item_url": "https://sandbox.ebay.com/itm/{}",
    },
}


class SellerError(Exception):
    """A problem to show the seller as-is (eBay's own error text where there is one)."""


def suggest_price(offers: list[dict]) -> float | None:
    """The seller's pricing rule: the cheapest comparable Buy-It-Now offer's
    *total* (item + that seller's shipping) becomes the item price; the buyer
    then pays the seller's own shipping (from the template) on top."""
    totals = [o["total"] for o in offers if o.get("total") is not None]
    return round(min(totals), 2) if totals else None


def item_id_from(text: str) -> str | None:
    """ "318839546855" or any eBay item URL → the item ID."""
    text = (text or "").strip()
    if re.fullmatch(r"\d{9,15}", text):
        return text
    key = canonical_listing_id(text)
    return key.split(":", 1)[1] if key and key.startswith("ebay:") else None


# ── Listing template (copied from one of the seller's listings) ──────────────


@dataclass
class ListingTemplate:
    source_item_id: str
    title: str = ""
    category_id: str = "139973"
    condition_id: str = ""
    country: str = "DE"
    currency: str = "EUR"
    location: str = ""
    postal_code: str = ""
    dispatch_time_max: str = ""
    description_html: str = ""
    best_offer: bool = False
    shipping_type: str = "Flat"
    # [{"priority", "service", "cost", "additional_cost", "free"}] — domestic services, in eBay's order
    shipping_services: list[dict] = field(default_factory=list)
    # {"returns_accepted_option", "returns_within_option", "refund_option", "shipping_cost_paid_by_option"}
    return_policy: dict = field(default_factory=dict)
    # Business-policy IDs {"shipping", "return", "payment"} when the account uses them (they replace inline details)
    seller_profiles: dict = field(default_factory=dict)

    @property
    def shipping_cost(self) -> float | None:
        """What the buyer pays for shipping with the first (default) service."""
        if not self.shipping_services:
            return None
        first = self.shipping_services[0]
        return 0.0 if first.get("free") else first.get("cost")

    def summary(self) -> dict:
        return {
            "source_item_id": self.source_item_id,
            "title": self.title,
            "category_id": self.category_id,
            "condition_id": self.condition_id,
            "location": ", ".join(p for p in (self.postal_code, self.location) if p),
            "shipping_service": self.shipping_services[0]["service"] if self.shipping_services else "",
            "shipping_cost": self.shipping_cost,
            "returns": self.return_policy.get("returns_accepted_option", ""),
            "business_policies": bool(self.seller_profiles),
            "best_offer": self.best_offer,
            "description_preview": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", self.description_html)).strip()[:160],
        }


def _text(element: ET.Element | None, path: str, default: str = "") -> str:
    found = element.find(path, _NSMAP) if element is not None else None
    return (found.text or "").strip() if found is not None and found.text else default


def _amount(element: ET.Element | None, path: str) -> float | None:
    raw = _text(element, path)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def parse_template(item: ET.Element) -> ListingTemplate:
    """A GetItem ``<Item>`` → the settings a new listing should copy."""
    services = []
    for opt in item.findall("e:ShippingDetails/e:ShippingServiceOptions", _NSMAP):
        service = _text(opt, "e:ShippingService")
        if not service:
            continue
        services.append(
            {
                "priority": _text(opt, "e:ShippingServicePriority", str(len(services) + 1)),
                "service": service,
                "cost": _amount(opt, "e:ShippingServiceCost"),
                "additional_cost": _amount(opt, "e:ShippingServiceAdditionalCost"),
                "free": _text(opt, "e:FreeShipping").lower() == "true",
            }
        )
    returns = item.find("e:ReturnPolicy", _NSMAP)
    return_policy = {
        key: value
        for key, value in {
            "returns_accepted_option": _text(returns, "e:ReturnsAcceptedOption"),
            "returns_within_option": _text(returns, "e:ReturnsWithinOption"),
            "refund_option": _text(returns, "e:RefundOption"),
            "shipping_cost_paid_by_option": _text(returns, "e:ShippingCostPaidByOption"),
        }.items()
        if value
    }
    profiles = item.find("e:SellerProfiles", _NSMAP)
    seller_profiles = {
        key: value
        for key, value in {
            "shipping": _text(profiles, "e:SellerShippingProfile/e:ShippingProfileID"),
            "return": _text(profiles, "e:SellerReturnProfile/e:ReturnProfileID"),
            "payment": _text(profiles, "e:SellerPaymentProfile/e:PaymentProfileID"),
        }.items()
        if value
    }
    return ListingTemplate(
        source_item_id=_text(item, "e:ItemID"),
        title=_text(item, "e:Title"),
        category_id=_text(item, "e:PrimaryCategory/e:CategoryID", "139973"),
        condition_id=_text(item, "e:ConditionID"),
        country=_text(item, "e:Country", "DE"),
        currency=_text(item, "e:Currency", "EUR"),
        location=_text(item, "e:Location"),
        postal_code=_text(item, "e:PostalCode"),
        dispatch_time_max=_text(item, "e:DispatchTimeMax"),
        description_html=_text(item, "e:Description"),
        best_offer=_text(item, "e:BestOfferDetails/e:BestOfferEnabled").lower() == "true",
        shipping_type=_text(item, "e:ShippingDetails/e:ShippingType", "Flat"),
        shipping_services=services,
        return_policy=return_policy,
        seller_profiles=seller_profiles,
    )


# ── A new listing ────────────────────────────────────────────────────────────


@dataclass
class Draft:
    title: str
    price: float
    condition_id: str
    photo_urls: list[str]
    ean: str = ""
    epid: str = ""
    best_offer: bool | None = None  # None: as in the template
    # Only sent for products eBay's catalog doesn't know — a catalog match brings its own.
    item_specifics: dict[str, str] = field(default_factory=dict)

    def problems(self) -> list[str]:
        found = []
        if not self.title.strip():
            found.append("Title is missing.")
        if len(self.title) > 80:
            found.append("The title is longer than 80 characters.")
        if not self.price or self.price <= 0:
            found.append("Price is missing.")
        if not self.condition_id:
            found.append("Condition is missing.")
        if not self.photo_urls:
            found.append("At least one photo is needed.")
        if len(self.photo_urls) > 24:
            found.append("At most 24 photos.")
        return found


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrs: str) -> ET.Element:
    element = ET.SubElement(parent, tag, attrs)
    if text is not None:
        element.text = text
    return element


def _money(value: float) -> str:
    return f"{value:.2f}"


def build_item(template: ListingTemplate, draft: Draft) -> ET.Element:
    """The ``<Item>`` for Verify/AddFixedPriceItem: the draft's own data plus
    everything else copied from the template."""
    item = ET.Element("Item")
    _sub(item, "Title", draft.title.strip())
    _sub(item, "Description", template.description_html)
    category = _sub(item, "PrimaryCategory")
    _sub(category, "CategoryID", template.category_id)
    _sub(item, "StartPrice", _money(draft.price), currencyID=template.currency)
    _sub(item, "ConditionID", draft.condition_id)
    _sub(item, "Country", template.country)
    _sub(item, "Currency", template.currency)
    if template.location:
        _sub(item, "Location", template.location)
    if template.postal_code:
        _sub(item, "PostalCode", template.postal_code)
    if template.dispatch_time_max:
        _sub(item, "DispatchTimeMax", template.dispatch_time_max)
    _sub(item, "ListingDuration", "GTC")
    _sub(item, "ListingType", "FixedPriceItem")
    _sub(item, "Quantity", "1")
    _sub(item, "Site", "Germany")

    if draft.ean or draft.epid:
        product = _sub(item, "ProductListingDetails")
        if draft.ean:
            _sub(product, "EAN", draft.ean)
        else:
            _sub(product, "ProductReferenceID", draft.epid)
        _sub(product, "IncludeeBayProductDetails", "true")
        _sub(product, "UseFirstProduct", "true")
        # The seller's own photos only — no catalog stock photo in front of them.
        _sub(product, "IncludeStockPhotoURL", "false")
        _sub(product, "UseStockPhotoURLAsGallery", "false")
    if draft.item_specifics:
        specifics = _sub(item, "ItemSpecifics")
        for name, value in draft.item_specifics.items():
            pair = _sub(specifics, "NameValueList")
            _sub(pair, "Name", name)
            _sub(pair, "Value", value)

    pictures = _sub(item, "PictureDetails")
    for url in draft.photo_urls:
        _sub(pictures, "PictureURL", url)

    if template.seller_profiles:
        profiles = _sub(item, "SellerProfiles")
        for key, container, id_tag in (
            ("payment", "SellerPaymentProfile", "PaymentProfileID"),
            ("return", "SellerReturnProfile", "ReturnProfileID"),
            ("shipping", "SellerShippingProfile", "ShippingProfileID"),
        ):
            if template.seller_profiles.get(key):
                _sub(_sub(profiles, container), id_tag, template.seller_profiles[key])
    if not template.seller_profiles.get("shipping") and template.shipping_services:
        shipping = _sub(item, "ShippingDetails")
        _sub(shipping, "ShippingType", template.shipping_type or "Flat")
        for service in template.shipping_services:
            option = _sub(shipping, "ShippingServiceOptions")
            _sub(option, "ShippingServicePriority", str(service["priority"]))
            _sub(option, "ShippingService", service["service"])
            if service.get("free"):
                _sub(option, "FreeShipping", "true")
            currency = template.currency
            _sub(option, "ShippingServiceCost", _money(service.get("cost") or 0.0), currencyID=currency)
            if service.get("additional_cost") is not None:
                _sub(option, "ShippingServiceAdditionalCost", _money(service["additional_cost"]), currencyID=currency)
    if not template.seller_profiles.get("return") and template.return_policy:
        returns = _sub(item, "ReturnPolicy")
        for key, tag in (
            ("returns_accepted_option", "ReturnsAcceptedOption"),
            ("returns_within_option", "ReturnsWithinOption"),
            ("refund_option", "RefundOption"),
            ("shipping_cost_paid_by_option", "ShippingCostPaidByOption"),
        ):
            if template.return_policy.get(key):
                _sub(returns, tag, template.return_policy[key])

    best_offer = template.best_offer if draft.best_offer is None else draft.best_offer
    if best_offer:
        _sub(_sub(item, "BestOfferDetails"), "BestOfferEnabled", "true")
    return item


@dataclass
class ListingResult:
    item_id: str
    url: str
    fees: dict[str, float]
    warnings: list[str]


# ── The seller's account ─────────────────────────────────────────────────────


class EbaySeller:
    def __init__(self, session: requests.Session | None = None) -> None:
        self.client_id = os.environ.get("EBAY_CLIENT_ID", "").strip()
        self.client_secret = os.environ.get("EBAY_CLIENT_SECRET", "").strip()
        self.runame = os.environ.get("EBAY_RUNAME", "").strip()
        env = os.environ.get("EBAY_ENVIRONMENT", "production").strip().lower()
        self.urls = _ENVIRONMENTS["sandbox" if env == "sandbox" else "production"]
        self.session = session or requests.Session()
        self._token_lock = threading.Lock()
        self._conditions: dict[str, tuple[float, list[dict]]] = {}

    # ── Connection ──────────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.runame)

    @property
    def is_connected(self) -> bool:
        expires = float(database.get_setting(_KEY_REFRESH_EXPIRES, "0") or 0)
        return bool(database.get_setting(_KEY_REFRESH)) and expires > time.time()

    @property
    def username(self) -> str:
        return database.get_setting(_KEY_USERNAME, "") or ""

    def authorize_url(self, state: str) -> str:
        """eBay's consent page; the seller signs in there, never in this app."""
        query = urllib.parse.urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": self.runame,
                "response_type": "code",
                "scope": " ".join(SCOPES),
                "state": state,
            },
            quote_via=urllib.parse.quote,
        )
        return f"{self.urls['auth']}?{query}"

    def connect(self, code: str) -> str:
        """Exchange the consent page's code for tokens; returns the eBay username."""
        tokens = self._token_request({"grant_type": "authorization_code", "code": code, "redirect_uri": self.runame})
        if "refresh_token" not in tokens:
            raise SellerError("eBay returned no refresh token — please connect again.")
        now = time.time()
        database.set_setting(_KEY_REFRESH, tokens["refresh_token"])
        database.set_setting(_KEY_REFRESH_EXPIRES, str(now + int(tokens.get("refresh_token_expires_in", 47_304_000))))
        self._store_access_token(tokens, now)
        username = self.get_user()
        database.set_setting(_KEY_USERNAME, username)
        logger.info("eBay seller account connected: %s", username)
        return username

    def disconnect(self) -> None:
        database.delete_settings(*_TOKEN_KEYS)

    def access_token(self) -> str:
        """A valid user access token, refreshed from the stored refresh token when needed."""
        token = database.get_setting(_KEY_ACCESS, "")
        if token and float(database.get_setting(_KEY_ACCESS_EXPIRES, "0") or 0) > time.time():
            return token
        with self._token_lock:
            token = database.get_setting(_KEY_ACCESS, "")
            if token and float(database.get_setting(_KEY_ACCESS_EXPIRES, "0") or 0) > time.time():
                return token
            refresh = database.get_setting(_KEY_REFRESH, "")
            if not refresh:
                raise SellerError("No eBay account connected.")
            try:
                tokens = self._token_request(
                    {"grant_type": "refresh_token", "refresh_token": refresh, "scope": " ".join(SCOPES)}
                )
            except SellerError as exc:
                if "invalid_grant" in str(exc):
                    self.disconnect()
                    raise SellerError("The eBay connection has expired — please connect again.") from exc
                raise
            return self._store_access_token(tokens, time.time())

    def _store_access_token(self, tokens: dict, now: float) -> str:
        expires_at = now + int(tokens.get("expires_in", 7200)) - _TOKEN_REFRESH_MARGIN_S
        database.set_setting(_KEY_ACCESS, tokens["access_token"])
        database.set_setting(_KEY_ACCESS_EXPIRES, str(expires_at))
        return tokens["access_token"]

    def _token_request(self, data: dict) -> dict:
        credentials = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode("ascii")
        try:
            resp = self.session.post(
                self.urls["api"] + "/identity/v1/oauth2/token",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data=data,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise SellerError(f"Could not reach eBay: {exc}") from exc
        body = _json(resp)
        if not resp.ok or "access_token" not in body:
            reason = body.get("error_description") or body.get("error") or f"HTTP {resp.status_code}"
            raise SellerError(f"eBay sign-in failed: {reason} ({body.get('error', '')})")
        return body

    # ── Trading API ─────────────────────────────────────────────────────────

    def _trading(self, call: str, *children: ET.Element) -> tuple[ET.Element, list[str]]:
        """Run a Trading API call → (response root, warning texts); eBay's errors raise SellerError."""
        request = ET.Element(f"{call}Request", xmlns=_NS)
        _sub(request, "ErrorLanguage", "en_US")
        _sub(request, "WarningLevel", "High")
        for child in children:
            request.append(child)
        body = b'<?xml version="1.0" encoding="utf-8"?>' + ET.tostring(request, encoding="utf-8")
        try:
            resp = self.session.post(
                self.urls["trading"],
                data=body,
                headers={
                    "X-EBAY-API-IAF-TOKEN": self.access_token(),
                    "X-EBAY-API-CALL-NAME": call,
                    "X-EBAY-API-SITEID": _SITE_ID_GERMANY,
                    "X-EBAY-API-COMPATIBILITY-LEVEL": _TRADING_COMPATIBILITY_LEVEL,
                    "Content-Type": "text/xml; charset=utf-8",
                },
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise SellerError(f"Could not reach eBay: {exc}") from exc
        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            raise SellerError(f"eBay {call}: unexpected response (HTTP {resp.status_code}).") from exc
        errors, warnings = [], []
        for err in root.findall("e:Errors", _NSMAP):
            message = _text(err, "e:LongMessage") or _text(err, "e:ShortMessage")
            code = _text(err, "e:ErrorCode")
            text = f"{message} (code {code})" if code else message
            (warnings if _text(err, "e:SeverityCode") == "Warning" else errors).append(text)
        if _text(root, "e:Ack") not in ("Success", "Warning") or errors:
            raise SellerError(" · ".join(errors) or f"eBay {call} failed.")
        return root, warnings

    def get_user(self) -> str:
        root, _ = self._trading("GetUser")
        return _text(root, "e:User/e:UserID")

    def import_template(self, item_ref: str) -> ListingTemplate:
        """Copy the listing setup from one of the seller's own listings."""
        item_id = item_id_from(item_ref)
        if not item_id:
            raise SellerError("That is not an eBay item number or item URL.")
        root, _ = self._trading(
            "GetItem",
            _element("ItemID", item_id),
            _element("DetailLevel", "ReturnAll"),
            _element("IncludeItemSpecifics", "true"),
        )
        item = root.find("e:Item", _NSMAP)
        if item is None:
            raise SellerError("eBay did not return the item.")
        seller = _text(item, "e:Seller/e:UserID")
        if self.username and seller and seller.lower() != self.username.lower():
            raise SellerError(f"That item belongs to {seller}, not {self.username}.")
        template = parse_template(item)
        if not template.shipping_services and not template.seller_profiles:
            raise SellerError("That item has no shipping details to copy.")
        database.set_setting(_KEY_TEMPLATE, json.dumps(asdict(template)))
        return template

    def template(self) -> ListingTemplate | None:
        raw = database.get_setting(_KEY_TEMPLATE, "")
        if not raw:
            return None
        try:
            return ListingTemplate(**json.loads(raw))
        except (TypeError, ValueError):
            logger.warning("Stored listing template is unreadable — import it again")
            return None

    def publish(self, draft: Draft, *, verify_only: bool = False) -> ListingResult:
        """Validate the listing with eBay (``VerifyAddFixedPriceItem``, which
        creates nothing), then — unless *verify_only* — list it."""
        template = self.template()
        if template is None:
            raise SellerError("Import the template from one of your listings first.")
        problems = draft.problems()
        if problems:
            raise SellerError(" ".join(problems))
        root, warnings = self._trading("VerifyAddFixedPriceItem", build_item(template, draft))
        if verify_only:
            return ListingResult("", "", _fees(root), warnings)
        root, add_warnings = self._trading("AddFixedPriceItem", build_item(template, draft))
        item_id = _text(root, "e:ItemID")
        if not item_id:
            raise SellerError("eBay returned no item number.")
        return ListingResult(item_id, self.urls["item_url"].format(item_id), _fees(root), add_warnings)

    # ── REST APIs (Catalog, Media, Metadata) ────────────────────────────────

    def _rest(self, method: str, url: str, **kwargs) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.access_token()}",
            "X-EBAY-C-MARKETPLACE-ID": _MARKETPLACE_ID,
            "Accept-Language": "de-DE",
            **kwargs.pop("headers", {}),
        }
        try:
            return self.session.request(method, url, headers=headers, timeout=_REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            raise SellerError(f"Could not reach eBay: {exc}") from exc

    def lookup_products(self, query: str, limit: int = 5) -> list[dict]:
        """eBay catalog products for an EAN (or, failing a barcode, a title)."""
        query = re.sub(r"\s+", " ", query or "").strip()
        if not query:
            return []
        digits = query.replace(" ", "")
        params = {"gtin": digits} if re.fullmatch(r"\d{8,14}", digits) else {"q": query}
        params["limit"] = str(limit)
        resp = self._rest("GET", self.urls["api"] + "/commerce/catalog/v1_beta/product_summary/search", params=params)
        if resp.status_code == 204:
            return []
        body = _json(resp)
        if not resp.ok:
            raise SellerError(f"eBay catalog: {_rest_error(body, resp)}")
        return [_product(p) for p in body.get("productSummaries") or []]

    def upload_photo(self, data: bytes, filename: str, content_type: str) -> str:
        """Upload one photo to eBay's picture service → its eBay picture URL."""
        base = self.urls["apim"] + "/commerce/media/v1_beta/image"
        resp = self._rest("POST", base + "/create_image_from_file", files={"image": (filename, data, content_type)})
        body = _json(resp)
        if not resp.ok:
            raise SellerError(f"Photo upload: {_rest_error(body, resp)}")
        if body.get("imageUrl"):
            return body["imageUrl"]
        image_id = (resp.headers.get("Location") or "").rstrip("/").rsplit("/", 1)[-1]
        if not image_id:
            raise SellerError("Photo upload: eBay returned no image ID.")
        resp = self._rest("GET", f"{base}/{image_id}")
        body = _json(resp)
        if not resp.ok or not body.get("imageUrl"):
            raise SellerError(f"Photo upload: {_rest_error(body, resp)}")
        return body["imageUrl"]

    def conditions(self, category_id: str) -> list[dict]:
        """The conditions eBay allows in *category_id* ({"id", "name"}), cached for a day."""
        cached = self._conditions.get(category_id)
        if cached and time.monotonic() - cached[0] < _CONDITIONS_CACHE_S:
            return cached[1]
        resp = self._rest(
            "GET",
            f"{self.urls['api']}/sell/metadata/v1/marketplace/{_MARKETPLACE_ID}/get_item_condition_policies",
            params={"filter": f"categoryIds:{{{category_id}}}"},
        )
        body = _json(resp)
        if not resp.ok:
            raise SellerError(f"Conditions: {_rest_error(body, resp)}")
        policies = body.get("itemConditionPolicies") or [{}]
        found = [
            {"id": str(c.get("conditionId")), "name": c.get("conditionDescription") or str(c.get("conditionId"))}
            for c in policies[0].get("itemConditions") or []
            if c.get("conditionId")
        ]
        self._conditions[category_id] = (time.monotonic(), found)
        return found


def _element(tag: str, text: str) -> ET.Element:
    element = ET.Element(tag)
    element.text = text
    return element


def _fees(root: ET.Element) -> dict[str, float]:
    """Non-zero fees from a Verify/Add response ({"InsertionFee": 0.5, ...}).

    "ListingFee" is eBay's total of the others, so it's left out — listing it
    too would count every fee twice.
    """
    fees = {}
    for fee in root.findall("e:Fees/e:Fee", _NSMAP):
        name, amount = _text(fee, "e:Name"), _amount(fee, "e:Fee")
        if amount and name != "ListingFee":
            fees[name] = amount
    return fees


def _json(resp: requests.Response) -> dict:
    try:
        body = resp.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _rest_error(body: dict, resp: requests.Response) -> str:
    messages = [e.get("longMessage") or e.get("message") for e in body.get("errors") or [] if isinstance(e, dict)]
    return " · ".join(m for m in messages if m) or f"HTTP {resp.status_code}"


def _aspect(product: dict, *names: str) -> str:
    for aspect in product.get("aspects") or []:
        if aspect.get("localizedName") in names and aspect.get("localizedValues"):
            return aspect["localizedValues"][0]
    return ""


def _product(summary: dict) -> dict:
    gtins = summary.get("ean") or summary.get("gtin") or summary.get("upc") or []
    return {
        "epid": summary.get("epid", ""),
        "title": summary.get("title", ""),
        "ean": gtins[0] if gtins else "",
        "image": (summary.get("image") or {}).get("imageUrl", ""),
        "platform": _aspect(summary, "Plattform", "Platform"),
        "game": _aspect(summary, "Spielname", "Game Name"),
    }
