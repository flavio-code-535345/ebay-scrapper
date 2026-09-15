"""Tests for kleinanzeigen_scraper.py — Kleinanzeigen.de HTML scraper.

Closes the project's biggest test-coverage gap (kleinanzeigen_scraper.py was
at 15% coverage with no dedicated test file at all).
"""

from unittest.mock import MagicMock, patch

import pytest

from kleinanzeigen_scraper import KleinanzeigenScraper

_SAMPLE_HTML = """\
<html><body>
<ul id="srp-results">
<article class="aditem">
  <a class="ellipsis" href="/s-anzeige/xbox-360-spielesammlung/123">
    <h2 class="ellipsis">Xbox 360 Spielesammlung 10 Spiele</h2>
  </a>
  <p class="aditem-main--middle--price">25 € VB</p>
  <div class="aditem-main--top--left">Berlin</div>
  <div class="aditem-main--top--right">Heute, 19:30</div>
  <p class="aditem-main--middle--description">Sehr guter Zustand, alle Spiele funktionieren.</p>
  <img src="https://img.kleinanzeigen.de/api/v1/prod-ads/images/aa/aa1.jpg" />
</article>
<article class="aditem">
  <a class="ellipsis" href="/s-anzeige/ps4-konsole/456">
    <h2 class="ellipsis">PS4 Konsole mit Controller</h2>
  </a>
  <p class="aditem-main--middle--price">120 €</p>
  <div class="aditem-main--top--left">Hamburg</div>
  <div class="aditem-main--top--right">05.09.2025</div>
  <p class="aditem-main--middle--description">Neu, originalverpackt.</p>
</article>
</ul>
</body></html>
"""


@pytest.fixture
def scraper():
    return KleinanzeigenScraper()


class TestSearch:
    def test_empty_query_returns_error(self, scraper):
        deals, errors = scraper.search("")
        assert deals == []
        assert "query is required" in errors[0]

    def test_blocked_returns_error(self, scraper):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch.object(scraper._session, "get", return_value=mock_resp):
            deals, errors = scraper.search("test")
        assert deals == []
        assert "blocked or captcha" in errors[0]

    def test_rate_limited_returns_error(self, scraper):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        with patch.object(scraper._session, "get", return_value=mock_resp):
            deals, errors = scraper.search("test")
        assert deals == []
        assert "429" in errors[0]

    def test_http_error_returns_error(self, scraper):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.ok = False
        with patch.object(scraper._session, "get", return_value=mock_resp):
            deals, errors = scraper.search("test")
        assert deals == []
        assert "500" in errors[0]

    def test_request_exception_returns_error(self, scraper):
        import requests

        with patch.object(scraper._session, "get", side_effect=requests.RequestException("boom")):
            deals, errors = scraper.search("test")
        assert deals == []
        assert "boom" in errors[0]

    def test_no_articles_found_returns_empty_no_error(self, scraper):
        """A page with no matching article elements returns cleanly, not as an error."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>no results</body></html>"
        with patch.object(scraper._session, "get", return_value=mock_resp):
            deals, errors = scraper.search("test")
        assert deals == []
        assert errors == []

    def test_parses_articles_into_deals(self, scraper):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_HTML
        with patch.object(scraper._session, "get", return_value=mock_resp):
            deals, errors = scraper.search("xbox", max_results=10)
        assert len(deals) == 2
        assert deals[0]["title"] == "Xbox 360 Spielesammlung 10 Spiele"
        assert deals[0]["price"] == 25.0
        assert deals[0]["url"].endswith("/s-anzeige/xbox-360-spielesammlung/123")
        assert deals[0]["url"].startswith("https://www.kleinanzeigen.de")

    def test_respects_max_results(self, scraper):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.text = _SAMPLE_HTML
        with patch.object(scraper._session, "get", return_value=mock_resp):
            deals, _errors = scraper.search("xbox", max_results=1)
        assert len(deals) == 1


class TestDealSchema:
    """Every deal must expose the full shared schema, including the fields
    added by this rewrite (condition_normalized, shipping_note, a properly
    parsed listing_date instead of raw page text)."""

    def _search(self, scraper, html=_SAMPLE_HTML):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.text = html
        with patch.object(scraper._session, "get", return_value=mock_resp):
            deals, _errors = scraper.search("xbox", max_results=10)
        return deals

    def test_vb_price_sets_shipping_note_not_shipping(self, scraper):
        """The 'VB' (Verhandlungsbasis / negotiable) flag is a price
        attribute, not shipping info — it must not be stuffed into the
        'shipping' field (which stays empty; Kleinanzeigen exposes no real
        shipping-cost data on the results page)."""
        deals = self._search(scraper)
        assert deals[0]["shipping"] == ""
        assert deals[0]["shipping_note"] == "VB"

    def test_non_vb_price_has_no_shipping_note(self, scraper):
        deals = self._search(scraper)
        assert deals[1]["shipping"] == ""
        assert deals[1]["shipping_note"] == ""

    def test_condition_normalized_present(self, scraper):
        from models import Condition

        deals = self._search(scraper)
        assert "condition_normalized" in deals[0]
        # deals[1]'s description says "Neu, originalverpackt."
        assert deals[1]["condition_normalized"] == Condition.NEW

    def test_listing_date_relative_text_parsed_to_iso(self, scraper):
        """'Heute, 19:30' must be parsed into a real ISO-8601 timestamp, not
        left as raw page text — the old behavior silently broke the
        newest-first sort for every Kleinanzeigen deal."""
        deals = self._search(scraper)
        from datetime import datetime

        parsed = datetime.fromisoformat(deals[0]["listing_date"])
        assert parsed.hour == 19
        assert parsed.minute == 30

    def test_listing_date_absolute_text_parsed_to_iso(self, scraper):
        deals = self._search(scraper)
        from datetime import datetime

        parsed = datetime.fromisoformat(deals[1]["listing_date"])
        assert parsed.year == 2025
        assert parsed.month == 9
        assert parsed.day == 5

    def test_unparsable_date_is_none_not_raw_text(self, scraper):
        html = _SAMPLE_HTML.replace('<div class="aditem-main--top--right">Heute, 19:30</div>', "")
        deals = self._search(scraper, html)
        assert deals[0]["listing_date"] is None

    def test_image_issues_reflects_presence_of_images(self, scraper):
        deals = self._search(scraper)
        assert deals[0]["image_issues"] == []  # has an <img>
        assert deals[1]["image_issues"] == ["no_images"]  # no <img> in fixture

    def test_seller_count_and_is_trending_are_documented_stubs(self, scraper):
        """Kleinanzeigen exposes neither on its results page — these stay
        fixed stubs rather than guessed-at values."""
        deals = self._search(scraper)
        assert deals[0]["seller_count"] == ""
        assert deals[0]["is_trending"] is False


class TestRateLimit:
    def test_rate_limit_serializes_concurrent_callers(self, scraper):
        """_rate_limit must hold its lock for the whole check-sleep-update
        sequence so concurrent callers can't both pass the elapsed-time
        check together and burst the target site."""
        import threading
        import time

        call_times = []
        lock_calls = threading.Barrier(2, timeout=2)

        original_rate_limit = scraper._rate_limit

        def tracked_rate_limit():
            lock_calls.wait()
            original_rate_limit()
            call_times.append(time.monotonic())

        threads = [threading.Thread(target=tracked_rate_limit) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(call_times) == 2
        assert abs(call_times[1] - call_times[0]) >= 1.0  # _REQUEST_DELAY = 1.5s, minus scheduling slack


class TestExtractPrice:
    def test_german_format_with_vb(self, scraper):
        html = '<p class="aditem-main--middle--price">1.234,56 € VB</p>'
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser")
        price, is_vb = scraper._extract_price(article)
        assert price == 1234.56
        assert is_vb is True

    def test_plain_integer_price(self, scraper):
        html = '<p class="aditem-main--middle--price">35 €</p>'
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser")
        price, is_vb = scraper._extract_price(article)
        assert price == 35.0
        assert is_vb is False

    def test_zu_verschenken_has_no_numeric_price(self, scraper):
        html = '<p class="aditem-main--middle--price">Zu verschenken</p>'
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser")
        price, _is_vb = scraper._extract_price(article)
        assert price == 0.0

    def test_no_price_element(self, scraper):
        html = "<article></article>"
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser")
        price, is_vb = scraper._extract_price(article)
        assert price == 0.0
        assert is_vb is False

    def test_out_of_range_price_rejected(self, scraper):
        html = '<p class="aditem-main--middle--price">999999999 €</p>'
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser")
        price, _is_vb = scraper._extract_price(article)
        assert price == 0.0


class TestExtractCondition:
    def test_neu_keyword(self, scraper):
        assert scraper._extract_condition("Neu, OVP", "") == "Neu"

    def test_sehr_gut_keyword(self, scraper):
        assert scraper._extract_condition("Sehr guter Zustand", "") == "Sehr gut"

    def test_gebraucht_keyword(self, scraper):
        assert scraper._extract_condition("gebraucht, voll funktionsfähig", "") == "Gebraucht"

    def test_defekt_keyword(self, scraper):
        assert scraper._extract_condition("Defekt, für Bastler", "") == "Defekt"

    def test_no_keyword_returns_empty(self, scraper):
        assert scraper._extract_condition("", "") == ""


class TestExtractRating:
    def test_top_text_badge(self, scraper):
        html = '<article><span class="rating">TOP Anbieter</span></article>'
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser")
        assert scraper._extract_rating(article) == 100.0

    def test_no_rating_signal(self, scraper):
        html = "<article></article>"
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser")
        assert scraper._extract_rating(article) == 0.0


class TestParseArticle:
    def test_missing_link_returns_none(self, scraper):
        html = "<article><h2>No link here</h2></article>"
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser").find("article")
        assert scraper._parse_article(article) is None

    def test_missing_title_returns_none(self, scraper):
        html = '<article><a class="ellipsis" href="/s-anzeige/x/1"></a></article>'
        from bs4 import BeautifulSoup

        article = BeautifulSoup(html, "html.parser").find("article")
        assert scraper._parse_article(article) is None

    def test_exception_in_parsing_returns_none_not_raises(self, scraper):
        """A malformed/unexpected article element must not blow up the whole
        search — _parse_article swallows exceptions and returns None."""
        assert scraper._parse_article(object()) is None
