"""Tests for scraper.py — eBay.de HTML results scraper.

Parser tests run against real captured eBay markup (fixtures/ebay_srp_scard.html),
not hand-written HTML: the previous hand-written fixture kept these tests green
for months after eBay's real markup had moved on.
"""

import copy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from bs4 import BeautifulSoup

from models import Condition
from scraper import EbayScraper, _parse_eur_amount

_FIXTURE = (Path(__file__).parent / "fixtures" / "ebay_srp_scard.html").read_bytes()


@pytest.fixture
def scraper():
    return EbayScraper()


@pytest.fixture
def deals(scraper):
    deals, errors = scraper.parse_results_page(_FIXTURE, max_results=50)
    assert errors == []
    return {d["listing_id"]: d for d in deals}


def _page_with_card(mutate) -> bytes:
    """A copy of the real fixture page whose first card was edited by *mutate*
    — for edge cases the captured page doesn't happen to contain."""
    soup = BeautifulSoup(_FIXTURE, "html.parser")
    mutate(soup.select_one("ul.srp-results > li.s-card"))
    return str(soup).encode()


class TestParseRealResultsPage:
    def test_all_real_cards_parsed_and_placeholders_excluded(self, deals):
        # 6 real listings; the 2 "Shop on eBay" placeholder cards (outside
        # the results list) must not appear.
        assert len(deals) == 6
        assert all(d["title"] != "Shop on eBay" for d in deals.values())

    def test_buy_it_now_card_fields(self, deals):
        d = deals["ebay:206580175564"]
        assert d["title"] == "DJ Hero 2 Turntable Bundle inkl. DJ Hero 1 & 2 - Xbox 360 - Mischpult"
        assert d["price"] == 94.99
        assert d["condition"] == "Gebraucht | Gewerblich"
        assert d["condition_normalized"] == Condition.USED
        assert d["listing_type"] == "fixed"
        assert d["seller_rating"] == 100.0
        assert d["url"] == "https://www.ebay.de/itm/206580175564"
        assert d["image_urls"] == ["https://i.ebayimg.com/images/g/2jgAAeSwKRRp5W3g/s-l500.jpg"]
        assert d["image_issues"] == []

    def test_title_excludes_screen_reader_noise(self, deals):
        assert all("neuem Fenster" not in d["title"] for d in deals.values())

    def test_free_shipping_not_confused_with_free_returns(self, deals):
        """Card has both "Gratis 2-3 Tage Lieferung" and "Kostenloser Rückversand"."""
        d = deals["ebay:206580175564"]
        assert d["shipping"] == "Free"
        assert d["shipping_cost"] == 0.0

    def test_paid_shipping_and_german_decimal_seller_rating(self, deals):
        d = deals["ebay:407243441249"]  # "+EUR 7,69 · 2-3 Tage Lieferung", "95,6% positiv"
        assert d["shipping"] == "€7.69"
        assert d["shipping_cost"] == 7.69
        assert d["seller_rating"] == 95.6

    def test_condition_skips_platform_subtitle_row(self, deals):
        """The first subtitle row is the platform ("Microsoft · Microsoft Xbox 360"),
        not the condition — the old scraper reported it as the condition."""
        assert deals["ebay:407243441249"]["condition"] == "Gebraucht | Privat"

    def test_auction_detected_from_bid_row(self, deals):
        assert deals["ebay:318911799276"]["listing_type"] == "auction"
        assert deals["ebay:318911799276"]["price"] == 1.0

    def test_auction_with_buy_it_now_uses_current_bid(self, deals):
        d = deals["ebay:800701145155"]  # "EUR 7,03 / 0 Gebote / EUR 15,00 / Sofort-Kaufen"
        assert d["listing_type"] == "auction"
        assert d["price"] == 7.03

    def test_thumbnail_upgraded_to_full_size_image(self, deals):
        """Raw server HTML carries a 140px src; the full-size URL is in data-defer-load."""
        assert deals["ebay:318911799276"]["image_urls"] == [
            "https://i.ebayimg.com/images/g/R4wAAeSw3T1qtOtq/s-l500.jpg"
        ]

    def test_listing_date_from_relative_age(self, deals):
        """ "Vor 2 Std. eingestellt" → roughly two hours ago (coarse, but real)."""
        listed = datetime.fromisoformat(deals["ebay:206580175564"]["listing_date"])
        assert abs((datetime.now(UTC) - listed) - timedelta(hours=2)) < timedelta(minutes=5)
        days_old = datetime.fromisoformat(deals["ebay:800701145155"]["listing_date"])  # "Vor 2 T."
        assert abs((datetime.now(UTC) - days_old) - timedelta(days=2)) < timedelta(minutes=5)

    def test_german_location_left_blank(self, deals):
        """German items show no location row (LH_PrefLoc=1 already restricts to Germany)."""
        assert all(d["item_location"] == "" for d in deals.values())

    def test_max_results_respected(self, scraper):
        deals, _ = scraper.parse_results_page(_FIXTURE, max_results=2)
        assert len(deals) == 2


class TestEdgeCases:
    def test_price_range_listing_skipped(self, scraper):
        """A price range is a multi-variation listing (buyer picks one item)."""

        def to_range(card):
            price_row = card.select_one(".s-card__price").parent
            price_row.append(BeautifulSoup('<span class="s-card__price"> bis </span>', "html.parser"))
            price_row.append(BeautifulSoup('<span class="s-card__price">EUR 186,09</span>', "html.parser"))

        deals, _ = scraper.parse_results_page(_page_with_card(to_range))
        assert "ebay:206580175564" not in {d["listing_id"] for d in deals}
        assert len(deals) == 5

    def test_foreign_location_row(self, scraper):
        def add_location(card):
            row = copy.copy(card.select(".s-card__attribute-row")[1])
            row.span.string = "aus Großbritannien"
            card.select_one(".su-card-container__attributes__primary").append(row)

        deals, _ = scraper.parse_results_page(_page_with_card(add_location))
        assert deals[0]["item_location"] == "Großbritannien"

    def test_sold_count_marks_trending_and_feeds_scam_detection(self, scraper):
        """ "11+ verkauft" on a bundle listing is the bait-and-switch signal the
        scam detector looks for in seller_count."""

        def add_sold(card):
            row = copy.copy(card.select(".s-card__attribute-row")[1])
            row.span.string = "11+ verkauft"
            card.select_one(".su-card-container__attributes__primary").append(row)

        deals, _ = scraper.parse_results_page(_page_with_card(add_sold))
        assert deals[0]["is_trending"] is True
        assert deals[0]["seller_count"] == "11+ verkauft"

    def test_title_badge_stripped(self, scraper):
        """Some card layouts eBay serves put a "Neues Angebot" badge inside the title."""

        def add_badge(card):
            title = card.select_one(".s-card__title")
            title.insert(0, BeautifulSoup('<span class="su-styled-text">Neues Angebot</span>', "html.parser"))

        deals, _ = scraper.parse_results_page(_page_with_card(add_badge))
        assert deals[0]["title"] == "DJ Hero 2 Turntable Bundle inkl. DJ Hero 1 & 2 - Xbox 360 - Mischpult"

    def test_absolute_listing_date_for_older_listings(self, scraper):
        def listed_on(card):
            age = next(s for s in card.find_all(string=True) if "eingestellt" in s)
            age.replace_with("Eingestellt am Sep 20")

        deals, _ = scraper.parse_results_page(_page_with_card(listed_on))
        listed = datetime.fromisoformat(deals[0]["listing_date"])
        assert (listed.month, listed.day) in ((9, 19), (9, 20))  # German midnight, stored in UTC

    def test_listing_age_found_outside_the_usual_container(self, scraper):
        """The age text is located anywhere in the card, not only under one wrapper."""

        def move_age(card):
            age_row = next(r for r in card.select(".s-card__attribute-row") if "eingestellt" in r.get_text())
            age_row.extract()
            card.select_one(".su-card-container__header").append(age_row)

        deals, _ = scraper.parse_results_page(_page_with_card(move_age))
        assert deals[0]["listing_date"] is not None

    def test_page_without_results_list_is_reported(self, scraper):
        deals, errors = scraper.parse_results_page(b"<html><title>Pardon Our Interruption</title></html>")
        assert deals == []
        assert "unexpected page" in errors[0]
        assert "EBAY_CLIENT_ID" in errors[0]


class TestSearchRequest:
    def _ok_response(self):
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.content = _FIXTURE
        return resp

    def test_request_params(self, scraper):
        with (
            patch.object(scraper.session, "get", return_value=self._ok_response()) as mock_get,
            patch.object(scraper, "_rate_limit"),
        ):
            deals, errors = scraper.search("xbox 360 (sammlung,konvolut) -fifa", max_results=100)
        params = mock_get.call_args.kwargs["params"]
        assert params["_nkw"] == "xbox 360 (sammlung,konvolut) -fifa"  # eBay's OR/exclusion syntax passed through
        assert params["_sop"] == "10"  # newly listed ("12" is best match)
        assert params["LH_PrefLoc"] == "1"  # located in Germany
        assert params["_sacat"] == "1249"  # Videospiele & Konsolen
        assert params["_ipg"] == "120"
        assert len(deals) == 6
        assert errors == []

    def test_one_retry_after_a_403(self, scraper):
        """eBay often refuses a request while setting cookies, then serves it
        once they come back — observed live, roughly every other request."""
        refused = MagicMock(ok=False, status_code=403, reason="Forbidden")
        with (
            patch.object(scraper.session, "get", side_effect=[refused, self._ok_response()]) as mock_get,
            patch.object(scraper, "_rate_limit"),
        ):
            deals, errors = scraper.search("xbox")
        assert mock_get.call_count == 2
        assert len(deals) == 6
        assert errors == []

    def test_only_one_retry(self, scraper):
        refused = MagicMock(ok=False, status_code=403, reason="Forbidden")
        with (
            patch.object(scraper.session, "get", return_value=refused) as mock_get,
            patch.object(scraper, "_rate_limit"),
        ):
            deals, errors = scraper.search("xbox")
        assert mock_get.call_count == 2
        assert deals == []

    @pytest.mark.parametrize("status", [403, 429])
    def test_bot_protection_refusal_explains_the_api_alternative(self, scraper, status):
        resp = MagicMock(ok=False, status_code=status, reason="Forbidden")
        with patch.object(scraper.session, "get", return_value=resp), patch.object(scraper, "_rate_limit"):
            deals, errors = scraper.search("test")
        assert deals == []
        assert "EBAY_CLIENT_ID" in errors[0]

    def test_timeout(self, scraper):
        with (
            patch.object(scraper.session, "get", side_effect=requests.exceptions.Timeout()),
            patch.object(scraper, "_rate_limit"),
        ):
            deals, errors = scraper.search("test")
        assert deals == []
        assert "timed out" in errors[0]

    def test_connection_error(self, scraper):
        with (
            patch.object(scraper.session, "get", side_effect=requests.exceptions.ConnectionError("refused")),
            patch.object(scraper, "_rate_limit"),
        ):
            deals, errors = scraper.search("test")
        assert deals == []
        assert "connection" in errors[0].lower()


class TestParseEurAmount:
    def test_formats(self):
        assert _parse_eur_amount("EUR 94,99") == 94.99
        assert _parse_eur_amount("EUR 1.234,56") == 1234.56
        assert _parse_eur_amount("$20.00") == 20.0
        assert _parse_eur_amount("9,68") == 9.68
        assert _parse_eur_amount("no digits") is None


@pytest.mark.live
def test_live_ebay_search():
    """Opt-in drift detector (`pytest -m live`): hits the real ebay.de."""
    deals, errors = EbayScraper().search("xbox 360 (sammlung,konvolut,paket,bundle)", max_results=20)
    if not deals and errors and "EBAY_CLIENT_ID" in errors[0]:
        pytest.skip(f"eBay's bot protection blocks this network: {errors[0]}")
    assert deals, errors
    assert all(d["title"] and d["title"] != "Unknown" for d in deals)
    assert all(d["price"] > 0 for d in deals)
    assert sum(d["condition"] != "Unknown" for d in deals) >= len(deals) // 2
    assert all(d["listing_date"] for d in deals)
