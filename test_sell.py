"""Tests for sell.py — the Sell page's routes (eBay itself is faked)."""

import io
import os
import urllib.parse
from types import SimpleNamespace

import pytest

os.environ["GEMINI_API_KEY"] = ""
os.environ["EBAY_CLIENT_ID"] = ""
os.environ["EBAY_CLIENT_SECRET"] = ""

import app  # noqa: E402
import database  # noqa: E402
import sell  # noqa: E402
from ebay_seller import ListingResult, ListingTemplate, SellerError  # noqa: E402

_PASSWORD = "test-password-123"


class FakeSeller:
    is_configured = True
    is_connected = True
    username = "wucha23"

    def __init__(self):
        self.calls = []
        self._template = ListingTemplate(
            source_item_id="318839546855",
            condition_id="5000",
            postal_code="12345",
            shipping_services=[{"priority": "1", "service": "DE_DHLPaket", "cost": 1.8, "free": False}],
        )
        self.publish_error = None

    def authorize_url(self, state):
        return "https://auth.ebay.com/oauth2/authorize?state=" + state

    def connect(self, code):
        self.calls.append(("connect", code))
        return "wucha23"

    def template(self):
        return self._template

    def conditions(self, category_id):
        return [{"id": "5000", "name": "Gut"}]

    def listings_this_month(self):
        return 57

    def lookup_products(self, query):
        self.calls.append(("lookup", query))
        return [
            {
                "epid": "12050700050",
                "title": "Tom Clancy's EndWar (Microsoft Xbox 360)",
                "ean": "3307210333339",
                "image": "",
                "platform": "Microsoft Xbox 360",
                "game": "Tom Clancy's EndWar",
            }
        ]

    def upload_photo(self, data, filename, content_type):
        self.calls.append(("photo", len(data), content_type))
        return "https://i.ebayimg.com/images/g/x/s-l1600.jpg"

    def publish(self, draft, verify_only=False):
        self.calls.append(("publish", draft, verify_only))
        if self.publish_error:
            raise SellerError(self.publish_error)
        if verify_only:
            return ListingResult("", "", {}, ["a warning"])
        return ListingResult("318900000001", "https://www.ebay.de/itm/318900000001", {"FinalValueFee": 0.35}, [])


class FakePrices:
    is_configured = True

    def __init__(self):
        self.calls = []

    def cheapest_offers(self, gtin, *, exclude_seller="", postal_code="", limit=3):
        self.calls.append((gtin, exclude_seller, postal_code))
        return [{"title": "x", "url": "https://www.ebay.de/itm/1", "price": 2.99, "shipping": 0.0, "total": 2.99}], []


@pytest.fixture
def fakes(monkeypatch):
    seller, prices = FakeSeller(), FakePrices()
    monkeypatch.setattr(sell, "seller", seller)
    monkeypatch.setattr(sell, "prices", prices)
    monkeypatch.setattr(sell, "_FAILED_LOGIN_DELAY_S", 0)
    return SimpleNamespace(seller=seller, prices=prices)


@pytest.fixture
def client(tmp_path, monkeypatch, fakes):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "test.db"))
    database.init_db()
    monkeypatch.setenv("APP_PASSWORD", _PASSWORD)
    app.app.config["TESTING"] = True
    with app.app.test_client() as c:
        yield c


def _login(client):
    return client.post("/sell/login", data={"password": _PASSWORD})


class TestPasswordGate:
    def test_without_app_password_the_page_only_explains_setup(self, client, monkeypatch):
        monkeypatch.delenv("APP_PASSWORD")
        assert b"Set a password first" in client.get("/sell").data
        assert client.get("/api/sell/status").status_code == 401
        assert _login(client).status_code == 302
        assert client.get("/api/sell/status").status_code == 401

    def test_wrong_password(self, client):
        resp = client.post("/sell/login", data={"password": "nope"})
        assert "Wrong+password" in resp.headers["Location"]
        assert client.get("/api/sell/status").status_code == 401

    def test_every_sell_endpoint_needs_the_login(self, client):
        for method, path in [
            ("get", "/api/sell/status"),
            ("get", "/api/sell/product?q=1"),
            ("post", "/api/sell/template"),
            ("post", "/api/sell/photos"),
            ("post", "/api/sell/publish"),
            ("post", "/api/sell/ebay/code"),
            ("post", "/api/sell/ebay/disconnect"),
        ]:
            assert getattr(client, method)(path).status_code == 401, path
        assert client.get("/sell/ebay/connect").status_code == 302  # back to the login page

    def test_signed_in(self, client):
        _login(client)
        body = client.get("/api/sell/status").get_json()
        assert body["connected"] is True
        assert body["username"] == "wucha23"
        assert body["template"]["shipping_cost"] == 1.8
        assert body["conditions"] == [{"id": "5000", "name": "Gut"}]
        assert body["free_listings"] == {"used": 57, "allowance": 320}
        assert b'id="sellApp"' in client.get("/sell").data

    def test_changing_the_password_signs_everyone_out(self, client, monkeypatch):
        _login(client)
        monkeypatch.setenv("APP_PASSWORD", "a-new-password")
        assert client.get("/api/sell/status").status_code == 401

    def test_sign_out(self, client):
        _login(client)
        client.post("/sell/logout")
        assert client.get("/api/sell/status").status_code == 401


class TestEbayConnection:
    def test_connect_and_callback(self, client, fakes):
        _login(client)
        resp = client.get("/sell/ebay/connect")
        state = urllib.parse.parse_qs(urllib.parse.urlsplit(resp.headers["Location"]).query)["state"][0]
        resp = client.get("/sell/ebay/callback", query_string={"code": "v^1.1#abc", "state": state})
        assert "Connected+to+wucha23" in resp.headers["Location"]
        assert fakes.seller.calls == [("connect", "v^1.1#abc")]

    def test_callback_with_a_foreign_state_is_refused(self, client, fakes):
        _login(client)
        client.get("/sell/ebay/connect")
        resp = client.get("/sell/ebay/callback", query_string={"code": "x", "state": "forged"})
        assert "does+not+belong+to+this+session" in resp.headers["Location"]
        assert fakes.seller.calls == []

    def test_pasted_address(self, client, fakes):
        _login(client)
        resp = client.get("/sell/ebay/connect")
        state = urllib.parse.parse_qs(urllib.parse.urlsplit(resp.headers["Location"]).query)["state"][0]
        pasted = "https://example.org/sell/ebay/callback?" + urllib.parse.urlencode({"state": state, "code": "c-1"})
        assert client.post("/api/sell/ebay/code", json={"url": pasted}).get_json() == {"username": "wucha23"}
        assert fakes.seller.calls == [("connect", "c-1")]


class TestListing:
    def test_product_with_price_suggestion(self, client, fakes):
        _login(client)
        body = client.get("/api/sell/product?q=3307210333339").get_json()
        assert body["products"][0]["title"] == "Tom Clancy's EndWar (Microsoft Xbox 360)"
        assert body["suggested_price"] == 2.99
        assert body["shipping_cost"] == 1.8
        # own listings excluded; shipping costs quoted to the template's postal code
        assert fakes.prices.calls == [("3307210333339", "wucha23", "12345")]

    def test_photo_upload(self, client, fakes, monkeypatch):
        _login(client)
        resp = client.post("/api/sell/photos", data={"photo": (io.BytesIO(b"jpegdata"), "photo.jpg", "image/jpeg")})
        assert resp.get_json() == {"url": "https://i.ebayimg.com/images/g/x/s-l1600.jpg"}
        monkeypatch.setattr(sell, "_MAX_PHOTO_BYTES", 4)
        resp = client.post("/api/sell/photos", data={"photo": (io.BytesIO(b"jpegdata"), "photo.jpg", "image/jpeg")})
        assert resp.status_code == 400

    def _publish(self, client, **extra):
        body = {
            "title": "Tom Clancy's EndWar (Microsoft Xbox 360)",
            "ean": "3307210333339",
            "price": "2,99",
            "condition_id": "5000",
            "photos": ["https://i.ebayimg.com/images/g/x/s-l1600.jpg", "http://evil.example/x.jpg"],
            "best_offer": True,
            **extra,
        }
        return client.post("/api/sell/publish", json=body)

    def test_publish(self, client, fakes):
        _login(client)
        body = self._publish(client).get_json()
        assert body["url"] == "https://www.ebay.de/itm/318900000001"
        _, draft, verify_only = fakes.seller.calls[-1]
        assert (draft.price, draft.condition_id, draft.best_offer, verify_only) == (2.99, "5000", True, False)
        assert draft.photo_urls == ["https://i.ebayimg.com/images/g/x/s-l1600.jpg"]  # only eBay-hosted https URLs
        (listed,) = database.get_sell_listings()
        assert (listed["item_id"], listed["price"]) == ("318900000001", 2.99)

    def test_check_with_ebay_lists_nothing(self, client, fakes):
        _login(client)
        body = self._publish(client, verify_only=True).get_json()
        assert body["verified_only"] is True
        assert body["warnings"] == ["a warning"]
        assert database.get_sell_listings() == []

    def test_ebay_refusal_is_shown(self, client, fakes):
        _login(client)
        fakes.seller.publish_error = "The title is longer than 80 characters."
        resp = self._publish(client)
        assert resp.status_code == 422
        assert resp.get_json()["error"] == "The title is longer than 80 characters."
