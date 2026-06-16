"""Tests for scraper.py — legacy eBay HTML scraper."""

from unittest.mock import MagicMock, patch

import pytest

from scraper import _LOW_RES_URL_RE, EbayScraper


@pytest.fixture
def scraper():
    return EbayScraper()


# ── Sample eBay search results HTML ────────────────────────────────────────

_SAMPLE_HTML = """\
<html><body>
<ul class="srp-results">
<li class="s-item">
  <a class="s-item__link" href="http://ebay.de/itm/123">
    <div class="s-item__title">
      <span>Neues Angebot</span>
      <span>Xbox 360 Bundle 5 Spiele</span>
    </div>
  </a>
  <span class="s-item__price">€25,00</span>
  <span class="SECONDARY_INFO">Gebraucht | Privat</span>
  <span class="s-item__seller-info-text">98.5% positive</span>
  <span class="s-item__shipping">EUR 4.00 Versand</span>
  <div class="s-item__location">Berlin, Deutschland</div>
  <div class="s-item__image-wrapper">
    <img src="https://i.ebayimg.com/s-l500.jpg" />
  </div>
</li>
<li class="s-item">
  <a class="s-item__link" href="http://ebay.de/itm/456">
    <div class="s-item__title">
      <span>Gesponsert</span>
      <span>PS4 Spielesammlung 10 Spiele</span>
    </div>
  </a>
  <span class="s-item__price">EUR 45,00</span>
  <span class="SECONDARY_INFO">Neu | Gewerblich</span>
  <span class="s-item__seller-info-text">99.8% positive</span>
  <span class="s-item__shipping">Kostenloser Versand</span>
  <div class="s-item__location">Hamburg, DE</div>
  <div class="s-item__image-wrapper">
    <img src="https://i.ebayimg.com/s-l1600.jpg" />
  </div>
</li>
</ul>
</body></html>
"""


class TestEbayScraper:
    def test_search_timeout(self, scraper):
        """HTTP timeout returns empty deals with an error."""
        with patch.object(scraper.session, "get") as mock_get:
            mock_get.side_effect = TimeoutError("timed out")
            deals, errors = scraper.search("test")
        assert deals == []
        assert any("timed out" in e.lower() for e in errors)

    def test_search_connection_error(self, scraper):
        """Connection error returns empty deals with an error."""
        with patch.object(scraper.session, "get") as mock_get:
            mock_get.side_effect = ConnectionError("refused")
            deals, errors = scraper.search("test")
        assert deals == []
        assert any("connection" in e.lower() for e in errors)

    def test_search_403_error(self, scraper):
        """403 response returns empty deals with access-denied error."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 403
        mock_resp.reason = "Forbidden"
        with patch.object(scraper.session, "get", return_value=mock_resp):
            deals, errors = scraper.search("test")
        assert deals == []
        assert any("access denied" in e.lower() for e in errors)

    def test_search_429_error(self, scraper):
        """429 response returns rate-limit error."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.reason = "Too Many Requests"
        with patch.object(scraper.session, "get", return_value=mock_resp):
            deals, errors = scraper.search("test")
        assert deals == []
        assert any("rate limited" in e.lower() for e in errors)

    def test_search_parses_items(self, scraper):
        """Successful response parses listing items correctly."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_HTML.encode("utf-8")
        with patch.object(scraper.session, "get", return_value=mock_resp):
            deals, errors = scraper.search("xbox", max_results=10)
        assert len(deals) >= 1
        # First deal
        assert "Xbox 360 Bundle" in deals[0]["title"]
        assert deals[0]["price"] == 25.0
        assert deals[0]["condition"].startswith("Gebraucht")
        assert deals[0]["seller_rating"] == 98.5
        assert deals[0]["url"] == "http://ebay.de/itm/123"
        assert "DE" in deals[0]["item_location"].upper()
        assert len(deals[0]["image_urls"]) >= 1

    def test_search_strips_badge_spans(self, scraper):
        """Badge text like 'Neues Angebot' and 'Gesponsert' is stripped from title."""
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.content = _SAMPLE_HTML.encode("utf-8")
        with patch.object(scraper.session, "get", return_value=mock_resp):
            deals, _ = scraper.search("test", max_results=10)
        for d in deals:
            assert "Neues Angebot" not in d["title"]
            assert "Gesponsert" not in d["title"]

    def test_parse_price_german_format(self, scraper):
        """German price format '1.234,56' is parsed correctly."""
        assert scraper._parse_price("EUR 1.234,56") == 1234.56
        assert scraper._parse_price("€12,99") == 12.99
        assert scraper._parse_price("EUR 5,00 bis EUR 10,00") == 5.0

    def test_parse_price_english_format(self, scraper):
        """English price format '1,234.56' is parsed correctly."""
        assert scraper._parse_price("$1,234.56") == 1234.56
        assert scraper._parse_price("99.99") == 99.99

    def test_parse_seller_rating(self, scraper):
        assert scraper._parse_seller_rating("98.5% positive") == 98.5
        assert scraper._parse_seller_rating("99.9%") == 99.9
        assert scraper._parse_seller_rating("No rating") == 0.0

    def test_low_res_url_regex(self):
        """Regex matches low-res eBay image URLs."""
        assert _LOW_RES_URL_RE.search("s-l140.jpg")
        assert _LOW_RES_URL_RE.search("s-l225.jpg")
        assert not _LOW_RES_URL_RE.search("s-l500.jpg")
        assert not _LOW_RES_URL_RE.search("s-l1600.jpg")

    def test_extract_condition_with_text(self, scraper):
        """_extract_condition falls back to keyword matching."""
        from bs4 import BeautifulSoup
        html = '<span class="x">Gebraucht - Akzeptabler Zustand</span>'
        soup = BeautifulSoup(html, "html.parser")
        condition = scraper._extract_condition(soup)
        assert "Gebraucht" in condition

    def test_get_item_details(self, scraper):
        """get_item_details returns a dict even on error."""
        with patch.object(scraper.session, "get") as mock_get:
            mock_get.side_effect = Exception("boom")
            details = scraper.get_item_details("http://ebay.de/itm/999")
        assert details == {}
