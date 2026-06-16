"""Tests for ebay_api_client.py — eBay Browse API client."""

from unittest.mock import MagicMock, patch

import pytest

from ebay_api_client import _CONDITION_ID_MAP, EbayApiClient


@pytest.fixture
def client():
    """Return an EbayApiClient with dummy credentials."""
    c = EbayApiClient()
    c.client_id = "test-id"
    c.client_secret = "test-secret"
    c._base_url = "https://api.sandbox.ebay.com"
    return c


class TestIsConfigured:
    def test_not_configured_when_missing(self):
        c = EbayApiClient()
        assert not c.is_configured

    def test_configured_when_present(self):
        c = EbayApiClient()
        c.client_id = "id"
        c.client_secret = "secret"
        assert c.is_configured

    def test_not_configured_with_only_id(self):
        c = EbayApiClient()
        c.client_id = "id"
        assert not c.is_configured


class TestGetAccessToken:
    def test_returns_cached_token(self, client):
        """In-memory token is reused when not expired."""
        client._token = "cached-token"
        client._token_expires_at = 9999999999.0
        token = client._get_access_token()
        assert token == "cached-token"

    def test_fetches_new_token_on_expiry(self, client):
        """Expired token triggers a fresh OAuth request."""
        client._token = "stale"
        client._token_expires_at = 0.0
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"access_token": "fresh-token", "expires_in": 7200}
        with patch.object(client.session, "post", return_value=mock_resp):
            token = client._get_access_token()
        assert token == "fresh-token"
        assert client._token == "fresh-token"


class TestSearch:
    def test_not_configured_returns_error(self):
        c = EbayApiClient()
        deals, errors = c.search("query")
        assert deals == []
        assert len(errors) > 0

    def test_auth_failure_returns_error(self, client):
        """OAuth failure returns empty list with error."""
        with patch.object(client, "_get_access_token") as mock_auth:
            mock_auth.side_effect = Exception("auth failed")
            deals, errors = client.search("query")
        assert deals == []
        assert any("auth" in e.lower() for e in errors)

    def test_api_error_returns_error(self, client):
        """Non-OK HTTP response returns error."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.reason = "Server Error"
        mock_resp.json.return_value = {"errors": [{"message": "internal error"}]}
        with patch.object(client, "_get_access_token", return_value="tok"):
            with patch.object(client.session, "get", return_value=mock_resp):
                deals, errors = client.search("query")
        assert deals == []
        assert len(errors) > 0

    def test_401_invalidates_token(self, client):
        """401 response clears the cached token."""
        client._token = "bad-token"
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 401
        mock_resp.reason = "Unauthorized"
        mock_resp.json.return_value = {}
        with patch.object(client, "_get_access_token", return_value="tok"):
            with patch.object(client.session, "get", return_value=mock_resp):
                client.search("query")
        assert client._token is None
        assert client._token_expires_at == 0.0

    def test_parses_item_summaries(self, client):
        """Successful response returns normalised deals."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total": 1,
            "itemSummaries": [
                {
                    "itemId": "123",
                    "title": "Xbox 360 Bundle",
                    "price": {"value": "45.00", "currency": "EUR"},
                    "condition": "Used",
                    "conditionId": "3000",
                    "seller": {"feedbackPercentage": "98.5"},
                    "shippingOptions": [
                        {"shippingCostType": "FREE", "shippingCost": {"value": "0.00"}}
                    ],
                    "itemLocation": {"country": "DE", "city": "Berlin"},
                    "itemWebUrl": "http://ebay.de/itm/123",
                    "image": {"imageUrl": "http://i.ebayimg.com/test.jpg"},
                    "itemCreationDate": "2024-03-01T10:00:00.000Z",
                }
            ],
        }
        with patch.object(client, "_get_access_token", return_value="tok"):
            with patch.object(client.session, "get", return_value=mock_resp):
                deals, errors = client.search("xbox")
        assert len(deals) == 1
        assert deals[0]["title"] == "Xbox 360 Bundle"
        assert deals[0]["price"] == 45.0
        assert deals[0]["condition"] == "Used"
        assert deals[0]["seller_rating"] == 98.5
        assert deals[0]["shipping"] == "Free"
        assert "Berlin" in deals[0]["item_location"]
        assert deals[0]["listing_date"] == "2024-03-01T10:00:00.000Z"

    def test_normalize_item_missing_title_url(self, client):
        """Item without title and URL returns None."""
        item = {"itemId": "999"}
        assert client._normalize_item(item) is None

    def test_normalize_item_condition_id_mapping(self, client):
        """conditionId is mapped to human-readable label."""
        for cid, label in _CONDITION_ID_MAP.items():
            item = {
                "title": "Test",
                "itemWebUrl": "http://ex.com",
                "price": {"value": "10", "currency": "EUR"},
                "conditionId": cid,
                "condition": "original",
            }
            deal = client._normalize_item(item)
            assert deal["condition"] == label, f"conditionId {cid} should map to {label}"


class TestGetMedianSoldPrice:
    def test_not_configured_returns_none(self):
        c = EbayApiClient()
        price, source, errors = c.get_median_sold_price("test")
        assert price is None
        assert source == "none"

    def test_insights_api_fallback_to_browse(self, client):
        """Insights API failure falls back to Browse API."""
        mock_token = MagicMock(return_value="tok")
        # First call (Insights) fails, second (Browse) succeeds
        mock_fail = MagicMock()
        mock_fail.ok = False
        mock_fail.status_code = 403
        mock_fail.reason = "Forbidden"

        mock_ok = MagicMock()
        mock_ok.ok = True
        mock_ok.status_code = 200
        mock_ok.json.return_value = {
            "itemSummaries": [
                {
                    "title": "Test Game",
                    "price": {"value": "15.00", "currency": "EUR"},
                    "conditionId": "3000",
                }
            ]
        }

        with patch.object(client, "_get_access_token", mock_token):
            with patch.object(
                client.session, "get", side_effect=[mock_fail, mock_ok]
            ):
                price, source, errors = client.get_median_sold_price("Test Game Xbox 360")
        assert price == 15.0
        assert source == "active_listings"
        assert len(errors) >= 1  # fallback warning


class TestParseShipping:
    def test_free_shipping(self, client):
        assert client._parse_shipping([{"shippingCostType": "FREE"}]) == "Free"

    def test_no_options(self, client):
        assert client._parse_shipping([]) == "N/A"

    def test_paid_shipping(self, client):
        opts = [{"shippingCost": {"value": "4.50", "currency": "EUR"}}]
        assert "€4.50" in client._parse_shipping(opts)


class TestExtractPricesFromItems:
    def test_excludes_for_parts(self):
        items = [
            {"conditionId": "7000", "price": {"value": "5"}},
            {"conditionId": "3000", "price": {"value": "20"}},
        ]
        prices = EbayApiClient._extract_prices_from_items(items)
        assert 5.0 not in prices  # "For parts" excluded
        assert prices == [20.0]

    def test_handles_both_price_sources(self):
        items = [
            {"lastSoldPrice": {"value": "10"}},
            {"price": {"value": "15"}},
        ]
        prices = EbayApiClient._extract_prices_from_items(items)
        assert sorted(prices) == [10.0, 15.0]
