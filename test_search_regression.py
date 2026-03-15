"""
Regression tests for the broken-search bug.

Previously the search route could silently swallow errors and return 500 HTML
(instead of JSON), or return 0 deals even when the underlying search engine
succeeded.  These tests verify:

  1.  Search returns deals when the scraper works normally.
  2.  Search returns deals (rules-based) even when Gemini is broken.
  3.  Search returns a JSON error (not 500 HTML) when an unexpected server
      error occurs, and the frontend can parse it.
  4.  A DB failure in save_search does NOT prevent results from being returned.
  5.  The /api/search endpoint accepts the exact payload the frontend sends
      (Content-Type: application/json, keys: query + max_results).
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from app import app, scraper, gemini


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_DEALS = [
    {
        "title": "Xbox 360 Spielesammlung 20 Spiele Bundle",
        "price": 18.0,
        "condition": "Used",
        "seller_rating": 99.0,
        "url": "https://www.ebay.de/itm/111111",
        "shipping": "Free",
        "is_trending": False,
        "item_location": "Berlin, DE",
        "image_urls": [],
        "description": "20 Spiele im Paket",
        "seller_count": "1 verfügbar",
        "listing_date": "2026-03-10T10:00:00.000Z",
    },
    {
        "title": "PS3 Bundle 10 Spiele Paket",
        "price": 25.0,
        "condition": "Good",
        "seller_rating": 98.0,
        "url": "https://www.ebay.de/itm/222222",
        "shipping": "Free",
        "is_trending": True,
        "item_location": "Hamburg, DE",
        "image_urls": [],
        "description": "",
        "seller_count": "",
        "listing_date": "2026-03-12T08:00:00.000Z",
    },
]


class TestSearchRoute(unittest.TestCase):
    """End-to-end tests for POST /api/search."""

    def setUp(self):
        app.testing = True
        self.client = app.test_client()

    # ------------------------------------------------------------------
    # 1. Normal search returns deals
    # ------------------------------------------------------------------
    def test_search_returns_deals_when_scraper_succeeds(self):
        with patch.object(scraper, "search", return_value=(_SAMPLE_DEALS, [])):
            resp = self.client.post(
                "/api/search",
                data=json.dumps({"query": "Xbox 360 Spiele", "max_results": 10}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsNotNone(data, "Response must be valid JSON")
        self.assertGreater(data["deal_count"], 0, "Expected at least one deal")
        self.assertEqual(len(data["deals"]), data["deal_count"])

    # ------------------------------------------------------------------
    # 2. Search still returns deals when Gemini is broken (invalid model)
    # ------------------------------------------------------------------
    def test_search_returns_deals_when_gemini_fails(self):
        """
        Regression: the invalid model 'gemini-3.1-flash-lite-preview' (logged
        in the bug report) must not prevent results from being returned.
        Gemini failures should fall back to rules-based scoring silently.
        """
        def _broken_gemini(deals):
            raise RuntimeError("404 models/gemini-3.1-flash-lite-preview is not found")

        with patch.object(scraper, "search", return_value=(_SAMPLE_DEALS, [])):
            with patch.object(gemini, "assess_deals_batch", side_effect=_broken_gemini):
                with patch.object(gemini, "enabled", True):
                    resp = self.client.post(
                        "/api/search",
                        data=json.dumps({"query": "Xbox 360", "max_results": 10}),
                        content_type="application/json",
                    )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsNotNone(data)
        self.assertGreater(
            data["deal_count"], 0,
            "Deals must still be returned when Gemini raises an exception",
        )

    # ------------------------------------------------------------------
    # 3. Unexpected server error returns JSON, not HTML 500
    # ------------------------------------------------------------------
    def test_unexpected_server_error_returns_json(self):
        """
        If an unexpected exception escapes the search logic, the route must
        return a JSON body with an 'error' key (not a raw HTML 500 page) so
        the frontend can parse and display it properly.
        """
        with patch.object(scraper, "search", side_effect=RuntimeError("boom")):
            resp = self.client.post(
                "/api/search",
                data=json.dumps({"query": "test", "max_results": 5}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 500)
        self.assertIn("application/json", resp.content_type)
        data = resp.get_json()
        self.assertIsNotNone(data)
        self.assertIn("error", data)

    # ------------------------------------------------------------------
    # 4. DB failure in save_search does NOT prevent results being returned
    # ------------------------------------------------------------------
    def test_db_failure_in_save_search_does_not_break_response(self):
        """
        If the database write fails after search completes, the user should
        still get their results (non-fatal error path).
        """
        import database

        with patch.object(scraper, "search", return_value=(_SAMPLE_DEALS, [])):
            with patch.object(database, "save_search", side_effect=Exception("disk full")):
                resp = self.client.post(
                    "/api/search",
                    data=json.dumps({"query": "Xbox 360", "max_results": 10}),
                    content_type="application/json",
                )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsNotNone(data)
        self.assertGreater(data["deal_count"], 0)

    # ------------------------------------------------------------------
    # 5. Exact frontend payload format is accepted
    # ------------------------------------------------------------------
    def test_frontend_payload_format_is_accepted(self):
        """
        The frontend sends: {"query": "...", "max_results": 50}
        with Content-Type: application/json.  This must return 200.
        """
        with patch.object(scraper, "search", return_value=([], [])):
            resp = self.client.post(
                "/api/search",
                data=json.dumps({"query": "Nintendo DS Lot", "max_results": 50}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("deals", data)
        self.assertIn("deal_count", data)

    # ------------------------------------------------------------------
    # 6. Missing Content-Type header returns 400 JSON
    # ------------------------------------------------------------------
    def test_missing_content_type_returns_400(self):
        resp = self.client.post(
            "/api/search",
            data='{"query": "test", "max_results": 5}',
            # No content_type → Flask won't parse JSON
        )
        self.assertIn(resp.status_code, (400, 415))
        # Must be JSON even for error responses
        self.assertIn("application/json", resp.content_type)

    # ------------------------------------------------------------------
    # 7. Empty query returns 400 JSON
    # ------------------------------------------------------------------
    def test_empty_query_returns_400(self):
        resp = self.client.post(
            "/api/search",
            data=json.dumps({"query": "  ", "max_results": 5}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
