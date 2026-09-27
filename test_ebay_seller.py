"""Tests for ebay_seller.py — listing on the seller's own eBay account.

HTTP is faked; response shapes follow eBay's API documentation (Trading GetItem /
Verify/AddFixedPriceItem, OAuth token, Media, Catalog and Metadata APIs).
"""

import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import database
import ebay_seller
from ebay_seller import Draft, EbaySeller, ListingTemplate, SellerError, build_item, item_id_from, parse_template

_NS = {"e": "urn:ebay:apis:eBLBaseComponents"}
_GETITEM = (Path(__file__).parent / "fixtures" / "ebay_getitem_response.xml").read_bytes()


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()


def _item_from_fixture() -> ET.Element:
    return ET.fromstring(_GETITEM).find("e:Item", _NS)


def _response(status=200, body=None, content=b"", headers=None):
    resp = MagicMock(status_code=status, ok=200 <= status < 300, headers=headers or {})
    resp.json.side_effect = (lambda: body) if body is not None else ValueError("no json")
    resp.content = content
    return resp


def _trading_response(call: str, inner: str = "", ack: str = "Success") -> MagicMock:
    xml = (
        f'<?xml version="1.0"?><{call}Response xmlns="urn:ebay:apis:eBLBaseComponents">'
        f"<Ack>{ack}</Ack>{inner}</{call}Response>"
    )
    return _response(content=xml.encode())


class FakeSession:
    """Answers by endpoint; records every request."""

    def __init__(self):
        self.calls = []
        self.trading = {}  # call name → response
        self.rest = {}  # URL substring → response
        self.tokens = []  # queued OAuth token responses

    def post(self, url, data=None, headers=None, **kwargs):
        self.calls.append(("POST", url, data, headers or {}, kwargs))
        if url.endswith("/identity/v1/oauth2/token"):
            return self.tokens.pop(0)
        if url.endswith("/ws/api.dll"):
            return self.trading[headers["X-EBAY-API-CALL-NAME"]]
        raise AssertionError(f"unexpected POST {url}")

    def request(self, method, url, headers=None, **kwargs):
        self.calls.append((method, url, None, headers or {}, kwargs))
        for part, resp in self.rest.items():
            if part in url:
                return resp
        raise AssertionError(f"unexpected {method} {url}")

    def trading_calls(self):
        return [c[3]["X-EBAY-API-CALL-NAME"] for c in self.calls if c[1].endswith("/ws/api.dll")]


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def seller(session, monkeypatch):
    monkeypatch.setenv("EBAY_CLIENT_ID", "app-id")
    monkeypatch.setenv("EBAY_CLIENT_SECRET", "cert-id")
    monkeypatch.setenv("EBAY_RUNAME", "Flavio-Deal-Sell-abcd")
    monkeypatch.delenv("EBAY_ENVIRONMENT", raising=False)
    return EbaySeller(session=session)


def _connected(seller, username="wucha23"):
    """Put a valid stored connection in place, as connect() would."""
    now = time.time()
    database.set_setting("ebay_seller_refresh_token", "refresh-1")
    database.set_setting("ebay_seller_refresh_expires_at", str(now + 3600 * 24 * 500))
    database.set_setting("ebay_seller_access_token", "access-1")
    database.set_setting("ebay_seller_access_expires_at", str(now + 3600))
    database.set_setting("ebay_seller_username", username)
    return seller


def _template() -> ListingTemplate:
    return parse_template(_item_from_fixture())


def _draft(**overrides) -> Draft:
    values = {
        "title": "Tom Clancy's EndWar (Microsoft Xbox 360)",
        "price": 2.5,
        "condition_id": "5000",
        "photo_urls": ["https://i.ebayimg.com/images/g/a/s-l1600.jpg", "https://i.ebayimg.com/images/g/b/s-l1600.jpg"],
        "ean": "3307210333339",
    }
    values.update(overrides)
    return Draft(**values)


# ── Template ─────────────────────────────────────────────────────────────────


class TestTemplate:
    def test_copies_the_listing_setup(self):
        t = _template()
        assert (t.source_item_id, t.category_id, t.condition_id) == ("318839546855", "139973", "5000")
        assert (t.location, t.postal_code, t.country, t.currency) == ("FURTH,BY", "12345", "DE", "EUR")
        (service,) = t.shipping_services
        assert service == {
            "priority": "1",
            "service": "DE_DHLAlterssichtprüfung18",  # the seller's real service code
            "cost": 1.8,
            "additional_cost": None,
            "free": False,
        }
        assert t.shipping_cost == 1.8
        assert t.return_policy == {"returns_accepted_option": "ReturnsNotAccepted"}
        assert t.seller_profiles == {}
        assert t.best_offer is True
        assert t.dispatch_time_max == "3"
        assert t.description_html.startswith('<div><font face="Arial" size="4">Die CD(s) läuft/laufen einwandfrei')

    def test_business_policies_are_kept(self):
        item = _item_from_fixture()
        profiles = ET.SubElement(item, "{urn:ebay:apis:eBLBaseComponents}SellerProfiles")
        for container, tag, value in (
            ("SellerShippingProfile", "ShippingProfileID", "111"),
            ("SellerReturnProfile", "ReturnProfileID", "222"),
            ("SellerPaymentProfile", "PaymentProfileID", "333"),
        ):
            parent = ET.SubElement(profiles, "{urn:ebay:apis:eBLBaseComponents}" + container)
            ET.SubElement(parent, "{urn:ebay:apis:eBLBaseComponents}" + tag).text = value
        assert parse_template(item).seller_profiles == {"shipping": "111", "return": "222", "payment": "333"}

    def test_summary(self):
        summary = _template().summary()
        assert summary["shipping_cost"] == 1.8
        assert summary["location"] == "12345, FURTH,BY"
        assert summary["description_preview"].startswith("Die CD(s) läuft/laufen einwandfrei")


# ── The new listing ──────────────────────────────────────────────────────────


def _xml(item: ET.Element) -> ET.Element:
    """Round-trip through the default namespace, as eBay would read it."""
    wrapper = ET.Element("AddFixedPriceItemRequest", xmlns=_NS["e"])
    wrapper.append(item)
    return ET.fromstring(ET.tostring(wrapper)).find("e:Item", _NS)


class TestBuildItem:
    def test_draft_fields_and_copied_setup(self):
        item = _xml(build_item(_template(), _draft()))
        text = lambda path: item.findtext(path, namespaces=_NS)  # noqa: E731
        assert text("e:Title") == "Tom Clancy's EndWar (Microsoft Xbox 360)"
        assert text("e:StartPrice") == "2.50"
        assert item.find("e:StartPrice", _NS).get("currencyID") == "EUR"
        assert text("e:ConditionID") == "5000"
        assert text("e:PrimaryCategory/e:CategoryID") == "139973"
        assert (text("e:ListingDuration"), text("e:ListingType"), text("e:Quantity")) == ("GTC", "FixedPriceItem", "1")
        assert (text("e:Location"), text("e:PostalCode"), text("e:DispatchTimeMax")) == ("FURTH,BY", "12345", "3")
        assert text("e:Description").startswith('<div><font face="Arial" size="4">Die CD(s)')
        assert [p.text for p in item.findall("e:PictureDetails/e:PictureURL", _NS)] == _draft().photo_urls
        assert text("e:ShippingDetails/e:ShippingServiceOptions/e:ShippingService") == "DE_DHLAlterssichtprüfung18"
        assert text("e:ShippingDetails/e:ShippingServiceOptions/e:ShippingServiceCost") == "1.80"
        assert text("e:ReturnPolicy/e:ReturnsAcceptedOption") == "ReturnsNotAccepted"
        assert text("e:BestOfferDetails/e:BestOfferEnabled") == "true"

    def test_catalog_product_by_ean_with_own_photos_only(self):
        product = _xml(build_item(_template(), _draft())).find("e:ProductListingDetails", _NS)
        assert product.findtext("e:EAN", namespaces=_NS) == "3307210333339"
        assert product.findtext("e:IncludeeBayProductDetails", namespaces=_NS) == "true"
        assert product.findtext("e:IncludeStockPhotoURL", namespaces=_NS) == "false"

    def test_epid_when_there_is_no_ean(self):
        product = _xml(build_item(_template(), _draft(ean="", epid="12050700050"))).find("e:ProductListingDetails", _NS)
        assert product.findtext("e:ProductReferenceID", namespaces=_NS) == "12050700050"

    def test_best_offer_can_be_switched_off(self):
        item = _xml(build_item(_template(), _draft(best_offer=False)))
        assert item.find("e:BestOfferDetails", _NS) is None

    def test_business_policies_replace_inline_shipping_and_returns(self):
        template = _template()
        template.seller_profiles = {"shipping": "111", "return": "222", "payment": "333"}
        item = _xml(build_item(template, _draft()))
        assert item.find("e:ShippingDetails", _NS) is None
        assert item.find("e:ReturnPolicy", _NS) is None
        assert item.findtext("e:SellerProfiles/e:SellerShippingProfile/e:ShippingProfileID", namespaces=_NS) == "111"
        assert item.findtext("e:SellerProfiles/e:SellerPaymentProfile/e:PaymentProfileID", namespaces=_NS) == "333"

    def test_item_specifics_for_products_outside_the_catalog(self):
        draft = _draft(ean="", item_specifics={"Spielname": "Halo 3", "Plattform": "Microsoft Xbox 360"})
        item = _xml(build_item(_template(), draft))
        pairs = {
            p.findtext("e:Name", namespaces=_NS): p.findtext("e:Value", namespaces=_NS)
            for p in item.findall("e:ItemSpecifics/e:NameValueList", _NS)
        }
        assert pairs == {"Spielname": "Halo 3", "Plattform": "Microsoft Xbox 360"}
        assert item.find("e:ProductListingDetails", _NS) is None

    def test_draft_problems(self):
        assert _draft().problems() == []
        assert _draft(title=" ", price=0, condition_id="", photo_urls=[]).problems() == [
            "Title is missing.",
            "Price is missing.",
            "Condition is missing.",
            "At least one photo is needed.",
        ]
        assert "longer than 80" in _draft(title="x" * 81).problems()[0]


def test_suggested_price_is_the_cheapest_total():
    offers = [{"total": 7.49}, {"total": 5.5}, {"total": None}]
    assert ebay_seller.suggest_price(offers) == 5.5
    assert ebay_seller.suggest_price([]) is None


def test_item_id_from():
    assert item_id_from("318839546855") == "318839546855"
    assert item_id_from("https://www.ebay.de/itm/318839546855?hash=x") == "318839546855"
    assert item_id_from("https://www.ebay.de/itm/tom-clancys-endwar/318839546855") == "318839546855"
    assert item_id_from("nonsense") is None


# ── Account connection ───────────────────────────────────────────────────────


class TestConnection:
    def test_authorize_url(self, seller):
        url = urllib.parse.urlsplit(seller.authorize_url("state-123"))
        query = urllib.parse.parse_qs(url.query)
        assert url.netloc == "auth.ebay.com"
        assert query["redirect_uri"] == ["Flavio-Deal-Sell-abcd"]
        assert query["state"] == ["state-123"]
        assert query["response_type"] == ["code"]
        assert "https://api.ebay.com/oauth/api_scope/sell.inventory" in query["scope"][0].split()

    def test_connect_stores_tokens_and_username(self, seller, session):
        session.tokens.append(
            _response(
                body={
                    "access_token": "access-1",
                    "expires_in": 7200,
                    "refresh_token": "refresh-1",
                    "refresh_token_expires_in": 47304000,
                }
            )
        )
        session.trading["GetUser"] = _trading_response("GetUser", "<User><UserID>wucha23</UserID></User>")
        assert seller.connect("v^1.1#code") == "wucha23"
        assert seller.is_connected
        assert seller.username == "wucha23"
        token_call = session.calls[0]
        assert token_call[2] == {
            "grant_type": "authorization_code",
            "code": "v^1.1#code",
            "redirect_uri": "Flavio-Deal-Sell-abcd",
        }
        assert session.calls[1][3]["X-EBAY-API-IAF-TOKEN"] == "access-1"

    def test_expired_access_token_is_refreshed(self, seller, session):
        _connected(seller)
        database.set_setting("ebay_seller_access_expires_at", "0")
        session.tokens.append(_response(body={"access_token": "access-2", "expires_in": 7200}))
        assert seller.access_token() == "access-2"
        assert session.calls[0][2]["grant_type"] == "refresh_token"
        assert seller.access_token() == "access-2"  # stored, not refreshed again
        assert len(session.calls) == 1

    def test_revoked_refresh_token_disconnects(self, seller, session):
        _connected(seller)
        database.set_setting("ebay_seller_access_expires_at", "0")
        session.tokens.append(
            _response(400, body={"error": "invalid_grant", "error_description": "the provided token is invalid"})
        )
        with pytest.raises(SellerError, match="expired"):
            seller.access_token()
        assert not seller.is_connected

    def test_not_connected(self, seller):
        assert not seller.is_connected
        with pytest.raises(SellerError, match="No eBay account"):
            seller.access_token()


# ── Trading API calls ────────────────────────────────────────────────────────


class TestTrading:
    def test_import_template(self, seller, session):
        _connected(seller)
        session.trading["GetItem"] = _response(content=_GETITEM)
        template = seller.import_template("https://www.ebay.de/itm/318839546855")
        assert template.shipping_cost == 1.8
        assert seller.template() == template  # stored
        request = ET.fromstring(session.calls[-1][2].split(b"?>", 1)[1])
        assert request.findtext("e:ItemID", namespaces=_NS) == "318839546855"
        headers = session.calls[-1][3]
        assert (headers["X-EBAY-API-SITEID"], headers["X-EBAY-API-COMPATIBILITY-LEVEL"]) == ("77", "1477")

    def test_template_must_be_the_sellers_own_listing(self, seller, session):
        _connected(seller, username="someone_else")
        session.trading["GetItem"] = _response(content=_GETITEM)
        with pytest.raises(SellerError, match="belongs to wucha23"):
            seller.import_template("318839546855")

    def test_ebay_errors_are_shown(self, seller, session):
        _connected(seller)
        session.trading["GetItem"] = _trading_response(
            "GetItem",
            "<Errors><ShortMessage>Item not found.</ShortMessage><LongMessage>The item ID is invalid.</LongMessage>"
            "<ErrorCode>17</ErrorCode><SeverityCode>Error</SeverityCode></Errors>",
            ack="Failure",
        )
        with pytest.raises(SellerError, match=r"The item ID is invalid\. \(code 17\)"):
            seller.import_template("318839546855")

    def _ready(self, seller, session):
        _connected(seller)
        session.trading["GetItem"] = _response(content=_GETITEM)
        seller.import_template("318839546855")
        fees = (
            "<Fees><Fee><Name>FeaturedFee</Name><Fee currencyID='EUR'>0.0</Fee></Fee>"
            "<Fee><Name>InsertionFee</Name><Fee currencyID='EUR'>0.5</Fee></Fee>"
            "<Fee><Name>ListingFee</Name><Fee currencyID='EUR'>0.5</Fee></Fee></Fees>"
        )
        warning = (
            "<Errors><LongMessage>Stock photo not used.</LongMessage><ErrorCode>21919</ErrorCode>"
            "<SeverityCode>Warning</SeverityCode></Errors>"
        )
        session.trading["VerifyAddFixedPriceItem"] = _trading_response("VerifyAddFixedPriceItem", fees + warning)
        session.trading["AddFixedPriceItem"] = _trading_response(
            "AddFixedPriceItem", "<ItemID>318900000001</ItemID>" + fees
        )

    def test_publish_verifies_then_lists(self, seller, session):
        self._ready(seller, session)
        result = seller.publish(_draft())
        assert session.trading_calls()[-2:] == ["VerifyAddFixedPriceItem", "AddFixedPriceItem"]
        assert result.item_id == "318900000001"
        assert result.url == "https://www.ebay.de/itm/318900000001"
        assert result.fees == {"InsertionFee": 0.5}  # ListingFee is their total

    def test_verify_only_lists_nothing(self, seller, session):
        self._ready(seller, session)
        result = seller.publish(_draft(), verify_only=True)
        assert session.trading_calls()[-1] == "VerifyAddFixedPriceItem"
        assert result.item_id == ""
        assert result.warnings == ["Stock photo not used. (code 21919)"]

    def test_incomplete_draft_makes_no_call(self, seller, session):
        self._ready(seller, session)
        before = len(session.calls)
        with pytest.raises(SellerError, match="At least one photo"):
            seller.publish(_draft(photo_urls=[]))
        assert len(session.calls) == before

    def test_no_template_yet(self, seller):
        _connected(seller)
        with pytest.raises(SellerError, match="template"):
            seller.publish(_draft())


# ── REST APIs ────────────────────────────────────────────────────────────────


class TestRest:
    def test_photo_upload(self, seller, session):
        _connected(seller)
        session.rest["create_image_from_file"] = _response(
            201, body={"imageUrl": "https://i.ebayimg.com/images/g/x/s-l1600.jpg"}
        )
        assert seller.upload_photo(b"jpeg", "photo.jpg", "image/jpeg") == "https://i.ebayimg.com/images/g/x/s-l1600.jpg"
        method, url, _, headers, kwargs = session.calls[-1]
        assert url == "https://apim.ebay.com/commerce/media/v1_beta/image/create_image_from_file"
        assert headers["Authorization"] == "Bearer access-1"
        assert kwargs["files"]["image"] == ("photo.jpg", b"jpeg", "image/jpeg")

    def test_photo_upload_via_location_header(self, seller, session):
        _connected(seller)
        session.rest["create_image_from_file"] = _response(
            201, body={}, headers={"Location": "https://apim.ebay.com/commerce/media/v1_beta/image/IMG123"}
        )
        session.rest["/image/IMG123"] = _response(body={"imageUrl": "https://i.ebayimg.com/images/g/y/s-l1600.jpg"})
        assert seller.upload_photo(b"jpeg", "photo.jpg", "image/jpeg").endswith("/y/s-l1600.jpg")

    def test_product_lookup_by_ean(self, seller, session):
        _connected(seller)
        session.rest["product_summary/search"] = _response(
            body={
                "productSummaries": [
                    {
                        "epid": "12050700050",
                        "title": "Tom Clancy's EndWar (Microsoft Xbox 360)",
                        "ean": ["3307210333339"],
                        "image": {"imageUrl": "https://i.ebayimg.com/images/g/p/s-l225.jpg"},
                        "aspects": [
                            {"localizedName": "Plattform", "localizedValues": ["Microsoft Xbox 360"]},
                            {"localizedName": "Spielname", "localizedValues": ["Tom Clancy's EndWar"]},
                        ],
                    }
                ]
            }
        )
        (product,) = seller.lookup_products("3307210333339")
        assert product == {
            "epid": "12050700050",
            "title": "Tom Clancy's EndWar (Microsoft Xbox 360)",
            "ean": "3307210333339",
            "image": "https://i.ebayimg.com/images/g/p/s-l225.jpg",
            "platform": "Microsoft Xbox 360",
            "game": "Tom Clancy's EndWar",
        }
        assert session.calls[-1][4]["params"] == {"gtin": "3307210333339", "limit": "5"}
        assert session.calls[-1][3]["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_DE"

    def test_product_lookup_by_title_and_no_match(self, seller, session):
        _connected(seller)
        session.rest["product_summary/search"] = _response(204)
        assert seller.lookup_products("endwar xbox") == []
        assert session.calls[-1][4]["params"]["q"] == "endwar xbox"

    def test_conditions_are_cached(self, seller, session):
        _connected(seller)
        session.rest["get_item_condition_policies"] = _response(
            body={
                "itemConditionPolicies": [
                    {
                        "categoryId": "139973",
                        "itemConditions": [
                            {"conditionId": "1000", "conditionDescription": "Neu"},
                            {"conditionId": "5000", "conditionDescription": "Gut"},
                        ],
                    }
                ]
            }
        )
        assert seller.conditions("139973") == [{"id": "1000", "name": "Neu"}, {"id": "5000", "name": "Gut"}]
        seller.conditions("139973")
        assert len(session.calls) == 1
        assert session.calls[0][4]["params"] == {"filter": "categoryIds:{139973}"}


def test_template_round_trips_through_the_database(seller):
    template = _template()
    database.set_setting("ebay_listing_template", json.dumps(template.__dict__))
    assert seller.template() == template
