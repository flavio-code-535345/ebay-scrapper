"""Tests for app.py — Flask API routes."""

import os
import time
from unittest.mock import patch

import pytest

os.environ["GEMINI_API_KEY"] = ""
os.environ["EBAY_CLIENT_ID"] = ""
os.environ["EBAY_CLIENT_SECRET"] = ""

import app
import database


@pytest.fixture
def client(tmp_path):
    """Flask test client with a temporary database."""
    db_file = tmp_path / "test.db"
    old_path = database.DB_PATH
    database.DB_PATH = str(db_file)
    database.init_db()
    app.app.config["TESTING"] = True
    with app.app.test_client() as c:
        yield c
    database.DB_PATH = old_path


@pytest.fixture(autouse=True)
def _reset_globals(monkeypatch):
    """Reset module-level state between tests."""
    app.assessor.enabled = False
    app.assessor.user_enabled = True
    app.assessor._enricher.ebay_client = None
    # No test may reach a real site: sources a test doesn't stub return nothing.
    monkeypatch.setattr(app.scraper, "search_auctions", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(app, "kleinanzeigen", None)
    yield


# ── Index ──────────────────────────────────────────────────────────────────


class TestIndex:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Deal Finder" in resp.data


# ── Health ──────────────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_healthy(self, client):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "ai_enabled" in data
        assert "data_source" in data


# ── Search ──────────────────────────────────────────────────────────────────


class TestSearch:
    def test_search_requires_json(self, client):
        resp = client.post("/api/search", data="not json", content_type="text/plain")
        assert resp.status_code == 400
        assert "JSON" in resp.get_json()["error"]

    def test_search_requires_query(self, client):
        resp = client.post("/api/search", json={"query": ""})
        assert resp.status_code == 400
        assert "query" in resp.get_json()["error"].lower()

    def test_search_empty_results(self, client):
        """When both engines return no results, search returns empty."""
        app.kleinanzeigen = None
        with patch.object(app.scraper, "search", return_value=([], [])):
            resp = client.post("/api/search", json={"query": "nothing"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["deal_count"] == 0
        assert data["deals"] == []

    def test_search_parses_deals(self, client):
        """Search returns deals from the scraper engine."""
        fake_deals = [
            {
                "title": "Test Deal",
                "price": 10.0,
                "condition": "Used",
                "seller_rating": 95.0,
                "url": "http://ebay.de/itm/1",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "Berlin, DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": "2024-06-01T00:00:00Z",
            }
        ]
        with patch.object(app.scraper, "search", return_value=(fake_deals, [])):
            resp = client.post("/api/search", json={"query": "test"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["deal_count"] == 1
        assert data["deals"][0]["title"] == "Test Deal"

    def test_search_filters_non_german_items(self, client):
        """Items with location outside Germany are filtered out."""
        fake_deals = [
            {
                "title": "German Item",
                "price": 10.0,
                "condition": "Used",
                "seller_rating": 95.0,
                "url": "http://ebay.de/itm/de",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "Berlin, DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            },
            {
                "title": "US Item",
                "price": 5.0,
                "condition": "Used",
                "seller_rating": 90.0,
                "url": "http://ebay.de/itm/us",
                "shipping": "$5",
                "is_trending": False,
                "item_location": "New York, US",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            },
        ]
        with patch.object(app.scraper, "search", return_value=(fake_deals, [])):
            resp = client.post("/api/search", json={"query": "test"})
        data = resp.get_json()
        assert data["deal_count"] == 1
        assert data["deals"][0]["title"] == "German Item"

    def test_search_filters_sports_kinect(self, client):
        """Sports/Kinect items are filtered out client-side."""
        fake_deals = [
            {
                "title": "FIFA 22 PS4 Bundle",
                "price": 10.0,
                "condition": "Used",
                "seller_rating": 95.0,
                "url": "http://ebay.de/itm/fifa",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            },
            {
                "title": "Zelda Switch",
                "price": 30.0,
                "condition": "New",
                "seller_rating": 99.0,
                "url": "http://ebay.de/itm/zelda",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            },
        ]
        with patch.object(app.scraper, "search", return_value=(fake_deals, [])):
            resp = client.post("/api/search", json={"query": "test"})
        data = resp.get_json()
        assert data["deal_count"] == 1
        assert data["deals"][0]["title"] == "Zelda Switch"


# ── Search deadline ─────────────────────────────────────────────────────────
# Regression tests for the Cloudflare 524 fix: a slow search phase plus
# Gemini's own (much longer, independent) assessment budget could together
# exceed a reverse proxy's timeout, which then returns its own HTML error
# page instead of anything from this app. /api/search now hands
# assess_deals_batch a deadline derived from the request's own start time.


class TestSearchDeadline:
    def test_search_passes_deadline_to_assess_deals_batch(self, client):
        app.assessor.enabled = True
        app.assessor.user_enabled = True
        app.kleinanzeigen = None
        fake_deals = [
            {
                "title": "Test Deal",
                "price": 10.0,
                "condition": "Used",
                "seller_rating": 95.0,
                "url": "http://ebay.de/itm/1",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "Berlin, DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            }
        ]
        before = time.monotonic()
        with (
            patch.object(app.scraper, "search", return_value=(fake_deals, [])),
            patch.object(app.assessor, "assess_deals_batch", return_value=[None]) as mock_assess,
        ):
            resp = client.post("/api/search", json={"query": "test"})
        after = time.monotonic()
        assert resp.status_code == 200
        assert mock_assess.call_count == 1
        _, kwargs = mock_assess.call_args
        assert "deadline" in kwargs
        # Must be "request start" + the configured budget — not the
        # provider's own, much larger, independent default budget.
        assert before + app._SEARCH_DEADLINE_S <= kwargs["deadline"] <= after + app._SEARCH_DEADLINE_S

    def test_search_deadline_leaves_headroom_under_common_proxy_timeouts(self):
        """The default end-to-end deadline must stay comfortably under
        common reverse-proxy timeouts (Cloudflare's proxied-HTTP default is
        100s) so /api/search returns its own response instead of a proxy
        killing the connection and returning an HTML error page that the
        frontend can't parse as JSON."""
        assert app._SEARCH_DEADLINE_S <= 85, (
            f"_SEARCH_DEADLINE_S={app._SEARCH_DEADLINE_S} leaves too little margin "
            "under a 100s proxy timeout once response serialization/transmission is included"
        )


# ── Sort order ──────────────────────────────────────────────────────────────
# Regression test for a real bug in the old sort: deals with no known
# listing_date (e.g. every HTML-scraper-sourced deal, which exposes no date
# at all) collapsed to datetime.min and always sank below EVERY dated deal,
# even an old one — see models.sort_key_for_deal.


class TestSearchSortOrder:
    def _fake_deal(self, *, url, listing_date):
        return {
            "title": f"Deal {url}",
            "price": 10.0,
            "condition": "Used",
            "seller_rating": 95.0,
            "url": url,
            "shipping": "Free",
            "is_trending": False,
            "item_location": "Berlin, DE",
            "image_urls": [],
            "image_issues": [],
            "listing_date": listing_date,
        }

    def test_dated_deal_sorts_before_undated_deal_in_the_same_tier(self, client):
        """A deal with no known date must not sink below an OLD dated deal —
        it should sort right after dated deals within its rating tier."""
        app.assessor.enabled = True
        app.assessor.user_enabled = True
        app.kleinanzeigen = None
        fake_deals = [
            self._fake_deal(url="http://ebay.de/itm/undated", listing_date=None),
            self._fake_deal(url="http://ebay.de/itm/old-dated", listing_date="2020-01-01T00:00:00+00:00"),
        ]
        with (
            patch.object(app.scraper, "search", return_value=(fake_deals, [])),
            patch.object(
                app.assessor,
                "assess_deals_batch",
                return_value=[{"ai_deal_rating": "Good", "ai_assessed": True}] * 2,
            ),
        ):
            resp = client.post("/api/search", json={"query": "test"})
        urls = [d["url"] for d in resp.get_json()["deals"]]
        assert urls == ["http://ebay.de/itm/old-dated", "http://ebay.de/itm/undated"]

    def test_must_have_still_sorts_first_regardless_of_date(self, client):
        app.assessor.enabled = True
        app.assessor.user_enabled = True
        app.kleinanzeigen = None
        fake_deals = [
            self._fake_deal(url="http://ebay.de/itm/good-newer", listing_date="2024-06-01T00:00:00+00:00"),
            self._fake_deal(url="http://ebay.de/itm/must-have-undated", listing_date=None),
        ]
        ratings = {"http://ebay.de/itm/good-newer": "Good", "http://ebay.de/itm/must-have-undated": "Must Have"}

        def rate_by_url(deals, deadline=None):
            # Deals reach the AI best-first (search.pipeline.select), not in
            # scraper order — rate each by identity, not by position.
            return [{"ai_deal_rating": ratings[d["url"]], "ai_assessed": True} for d in deals]

        with (
            patch.object(app.scraper, "search", return_value=(fake_deals, [])),
            patch.object(app.assessor, "assess_deals_batch", side_effect=rate_by_url),
        ):
            resp = client.post("/api/search", json={"query": "test"})
        urls = [d["url"] for d in resp.get_json()["deals"]]
        assert urls == ["http://ebay.de/itm/must-have-undated", "http://ebay.de/itm/good-newer"]


# ── Settings ────────────────────────────────────────────────────────────────


class TestSettings:
    def test_get_settings(self, client):
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "gemini_model" in data
        assert "ai_enabled" in data
        assert "data_source" in data

    def test_update_data_source(self, client):
        resp = client.post("/api/settings", json={"data_source": "scraper"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["updated"]["data_source"] == "scraper"
        assert database.get_setting("data_source") == "scraper"

    def test_update_invalid_data_source(self, client):
        resp = client.post("/api/settings", json={"data_source": "invalid"})
        assert resp.status_code == 400
        assert "data_source" in resp.get_json()["errors"]

    def test_update_ai_enabled(self, client):
        resp = client.post("/api/settings", json={"ai_enabled": False})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ai_enabled"] is False
        # Check it was persisted
        assert database.get_setting("ai_enabled") == "false"

    def test_update_ai_enabled_non_bool(self, client):
        resp = client.post("/api/settings", json={"ai_enabled": "yes"})
        assert resp.status_code == 400

    def test_update_gemini_model(self, client):
        resp = client.post("/api/settings", json={"gemini_model": "gemini-2.0-flash"})
        assert resp.status_code == 200
        assert database.get_setting("gemini_model") == "gemini-2.0-flash"

    def test_update_gemini_model_empty(self, client):
        resp = client.post("/api/settings", json={"gemini_model": ""})
        assert resp.status_code == 400


# ── Save / Skip / Saved / Skipped ──────────────────────────────────────────


class TestSaveDeal:
    def test_save_requires_url(self, client):
        resp = client.post("/api/deals/save", json={})
        assert resp.status_code == 400

    def test_save_deal(self, client):
        resp = client.post(
            "/api/deals/save",
            json={
                "url": "http://ebay.de/itm/save1",
                "title": "Save Me",
                "price": 15.0,
            },
        )
        assert resp.status_code == 200
        assert resp.get_json()["saved"] is True
        saved = database.get_saved_deals()
        assert len(saved) == 1

    def test_unsave_deal(self, client):
        database.save_deal("http://ebay.de/itm/unsave1", "Unsave", 5.0)
        resp = client.post("/api/deals/unsave", json={"url": "http://ebay.de/itm/unsave1"})
        assert resp.status_code == 200
        assert resp.get_json()["saved"] is False
        assert database.get_saved_deals() == []

    def test_saved_list(self, client):
        database.save_deal("http://ebay.de/itm/sl1", "SL1", 1.0)
        database.save_deal("http://ebay.de/itm/sl2", "SL2", 2.0)
        resp = client.get("/api/deals/saved")
        data = resp.get_json()
        assert len(data) == 2


class TestSkipDeal:
    def test_skip_deal(self, client):
        resp = client.post(
            "/api/deals/skip",
            json={
                "url": "http://ebay.de/itm/skip1",
                "title": "Skip Me",
                "price": 99.0,
            },
        )
        assert resp.status_code == 200
        assert resp.get_json()["skipped"] is True
        assert len(database.get_skipped_deals()) == 1

    def test_skip_filters_from_search(self, client):
        """Skipped deals are excluded from search results."""
        database.skip_deal("http://ebay.de/itm/skipped-url", "Skipped", 10.0)
        fake_deals = [
            {
                "title": "Skipped Deal",
                "price": 10.0,
                "condition": "U",
                "seller_rating": 90.0,
                "url": "http://ebay.de/itm/skipped-url",
                "shipping": "F",
                "is_trending": False,
                "item_location": "DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            },
            {
                "title": "Good Deal",
                "price": 5.0,
                "condition": "N",
                "seller_rating": 99.0,
                "url": "http://ebay.de/itm/good",
                "shipping": "F",
                "is_trending": False,
                "item_location": "DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            },
        ]
        with patch.object(app.scraper, "search", return_value=(fake_deals, [])):
            resp = client.post("/api/search", json={"query": "test"})
        data = resp.get_json()
        assert data["deal_count"] == 1
        assert data["deals"][0]["title"] == "Good Deal"

    def test_unskip_deal(self, client):
        database.skip_deal("http://ebay.de/itm/unskip1", "Unskip", 5.0)
        resp = client.post("/api/deals/unskip", json={"url": "http://ebay.de/itm/unskip1"})
        assert resp.status_code == 200
        assert resp.get_json()["skipped"] is False

    def test_skipped_list(self, client):
        database.skip_deal("http://ebay.de/itm/sk1", "SK1", 1.0)
        resp = client.get("/api/deals/skipped")
        data = resp.get_json()
        assert len(data) == 1


# ── History ─────────────────────────────────────────────────────────────────


class TestHistory:
    def test_history_empty(self, client):
        resp = client.get("/api/history")
        assert resp.get_json() == []

    def test_history_after_search(self, client):
        with patch.object(app.scraper, "search", return_value=([], [])):
            client.post("/api/search", json={"query": "hist-test"})
        resp = client.get("/api/history")
        data = resp.get_json()
        assert len(data) >= 1
        assert data[0]["query"] == "hist-test"

    def test_history_non_numeric_limit_returns_400(self, client):
        """A malformed limit must not 500 the request."""
        resp = client.get("/api/history?limit=not-a-number")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_history_limit_is_clamped(self, client):
        """An out-of-range limit is clamped rather than passed through raw."""
        resp = client.get("/api/history?limit=999999")
        assert resp.status_code == 200


# ── Export ──────────────────────────────────────────────────────────────────


class TestExport:
    def test_export_csv(self, client):
        resp = client.get("/api/export")
        assert resp.status_code == 200
        assert resp.mimetype == "text/csv"
        assert "Content-Disposition" in resp.headers


# ── Stats ───────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats(self, client):
        resp = client.get("/api/stats")
        data = resp.get_json()
        assert "total_searches" in data
        assert "total_deals" in data
