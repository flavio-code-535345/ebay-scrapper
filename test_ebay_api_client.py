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
                deals, errors = client.search("query")
        assert client._token is None
        assert client._token_expires_at == 0.0
        assert deals == []
        assert any("EBAY_CLIENT_ID" in e for e in errors)

    def test_429_returns_rate_limit_message(self, client):
        """429 response includes a specific rate-limit explanation."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.reason = "Too Many Requests"
        mock_resp.json.return_value = {}
        with patch.object(client, "_get_access_token", return_value="tok"):
            with patch.object(client.session, "get", return_value=mock_resp):
                deals, errors = client.search("query")
        assert deals == []
        assert any("rate limit" in e.lower() for e in errors)

    def test_zero_results_gives_helpful_diagnostic(self, client):
        """An empty itemSummaries list returns a clear explanation instead of
        silently succeeding with nothing."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"total": 0, "itemSummaries": []}
        with patch.object(client, "_get_access_token", return_value="tok"):
            with patch.object(client.session, "get", return_value=mock_resp):
                deals, errors = client.search("some rare query")
        assert deals == []
        assert any("0 results" in e for e in errors)

    def test_one_bad_item_does_not_abort_the_whole_search(self, client):
        """A single item that fails to normalize is skipped, not fatal —
        the other, valid items in the same response still come back."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "total": 2,
            "itemSummaries": [
                {"itemId": "bad"},  # will raise inside _normalize_item
                {
                    "itemId": "123",
                    "title": "Good Item",
                    "price": {"value": "10.00", "currency": "EUR"},
                    "itemWebUrl": "http://ebay.de/itm/123",
                },
            ],
        }
        with (
            patch.object(client, "_get_access_token", return_value="tok"),
            patch.object(client.session, "get", return_value=mock_resp),
            patch.object(client, "_normalize_item", side_effect=[Exception("boom"), {"title": "Good Item"}]),
        ):
            deals, errors = client.search("query")
        assert deals == [{"title": "Good Item"}]
        assert any("could not be parsed" in e for e in errors)

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
                    "shippingOptions": [{"shippingCostType": "FREE", "shippingCost": {"value": "0.00"}}],
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

    def test_does_not_restrict_condition_ids(self, client):
        """Regression: the fixed-price search used to filter conditionIds to
        only 3000|1500 (Used / New-Other), silently excluding "Very Good"
        (4000), "Good" (5000), "Acceptable" (6000), and plain "New" (1000)
        — conditions _CONDITION_ID_MAP explicitly lists as understood by the
        assessor. That starved fixed-price ("Buy It Now") results relative
        to search_auctions(), which has never had a condition filter, so
        auctions ended up dominating merged search results. The filter must
        not restrict by condition at all, matching search_auctions()."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"total": 0, "itemSummaries": []}
        with patch.object(client, "_get_access_token", return_value="tok"):
            with patch.object(client.session, "get", return_value=mock_resp) as mock_get:
                client.search("xbox")
        _, kwargs = mock_get.call_args
        assert "conditionIds" not in kwargs["params"]["filter"]
        # Every condition the assessor understands should be a plausible
        # fixed-price result now.
        assert set(_CONDITION_ID_MAP) == {
            "1000",
            "1500",
            "1750",
            "2000",
            "2500",
            "3000",
            "4000",
            "5000",
            "6000",
            "7000",
        }

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

    def test_normalize_item_condition_normalized_present(self, client):
        """Every conditionId label maps to a known models.Condition value —
        never a silent 'unknown' for a value the API itself defines."""
        from models import Condition

        for cid in _CONDITION_ID_MAP:
            item = {
                "title": "Test",
                "itemWebUrl": "http://ex.com",
                "price": {"value": "10", "currency": "EUR"},
                "conditionId": cid,
            }
            deal = client._normalize_item(item)
            assert deal["condition_normalized"] != Condition.UNKNOWN, f"conditionId {cid} normalized to UNKNOWN"

    def test_normalize_item_image_issues_present(self, client):
        """image_issues is always present, matching the other two deal
        sources' schema — [] when images exist, ["no_images"] otherwise."""
        with_image = client._normalize_item(
            {"title": "Test", "itemWebUrl": "http://ex.com", "image": {"imageUrl": "http://x/1.jpg"}}
        )
        assert with_image["image_issues"] == []
        without_image = client._normalize_item({"title": "Test", "itemWebUrl": "http://ex.com"})
        assert without_image["image_issues"] == ["no_images"]


class TestGetLowestMarketPrice:
    def test_not_configured_returns_none(self):
        c = EbayApiClient()
        price, source, errors = c.get_lowest_market_price("test")
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
            with patch.object(client.session, "get", side_effect=[mock_fail, mock_ok]):
                price, source, errors = client.get_lowest_market_price("Test Game Xbox 360")
        assert price == 15.0
        assert source == "active_listings"
        assert len(errors) >= 1  # fallback warning


class TestParseShipping:
    def test_free_shipping(self, client):
        assert client._parse_shipping([{"shippingCostType": "FREE"}]) == ("Free", 0.0)

    def test_no_options(self, client):
        assert client._parse_shipping([]) == ("N/A", None)

    def test_paid_shipping(self, client):
        opts = [{"shippingCost": {"value": "4.50", "currency": "EUR"}}]
        assert client._parse_shipping(opts) == ("€4.50", 4.5)


class TestQueryParams:
    def _search_params(self, client, method="search"):
        mock_resp = MagicMock(ok=True, status_code=200)
        mock_resp.json.return_value = {"total": 0, "itemSummaries": []}
        with (
            patch.object(client, "_get_access_token", return_value="tok"),
            patch.object(client.session, "get", return_value=mock_resp) as mock_get,
        ):
            getattr(client, method)("xbox 360 (sammlung, konvolut)")
        return mock_get.call_args.kwargs["params"]

    @pytest.mark.parametrize("method", ["search", "search_auctions"])
    def test_query_sent_verbatim_without_undocumented_minus_terms(self, client, method):
        """The Browse API caps q at 100 chars and documents no "-word"
        exclusion; ~85 chars of "-skylanders -lego …" used to be appended to
        every query, pushing it past the cap."""
        assert self._search_params(client, method)["q"] == "xbox 360 (sammlung, konvolut)"

    @pytest.mark.parametrize("method", ["search", "search_auctions"])
    def test_requests_extended_fieldgroup_for_descriptions(self, client, method):
        """shortDescription only arrives with fieldgroups=EXTENDED."""
        assert "EXTENDED" in self._search_params(client, method)["fieldgroups"]


class TestListingIdentity:
    def test_legacy_item_id(self, client):
        deal = client._normalize_item(
            {"title": "T", "itemWebUrl": "https://www.ebay.de/itm/206580175564", "legacyItemId": "206580175564"}
        )
        assert deal["listing_id"] == "ebay:206580175564"
        assert deal["source"] == "ebay"

    def test_item_id_v1_format(self, client):
        deal = client._normalize_item({"title": "T", "itemWebUrl": "https://x", "itemId": "v1|206580175564|0"})
        assert deal["listing_id"] == "ebay:206580175564"

    def test_listing_type_from_buying_options(self, client):
        auction = client._normalize_item({"title": "T", "itemWebUrl": "https://x", "buyingOptions": ["AUCTION"]})
        fixed = client._normalize_item({"title": "T", "itemWebUrl": "https://x", "buyingOptions": ["FIXED_PRICE"]})
        assert auction["listing_type"] == "auction"
        assert fixed["listing_type"] == "fixed"

    def test_auction_search_skips_unusable_items(self, client):
        """An item with neither title nor URL used to crash search_auctions
        (it tagged fields onto the None that _normalize_item returned)."""
        mock_resp = MagicMock(ok=True, status_code=200)
        mock_resp.json.return_value = {"itemSummaries": [{"itemId": "x"}, {"title": "Ok", "itemWebUrl": "https://x"}]}
        with (
            patch.object(client, "_get_access_token", return_value="tok"),
            patch.object(client.session, "get", return_value=mock_resp),
        ):
            deals, errors = client.search_auctions("xbox")
        assert len(deals) == 1
        assert deals[0]["listing_type"] == "auction"


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
