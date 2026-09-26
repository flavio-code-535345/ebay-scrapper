"""Tests for kleinanzeigen_scraper.py — Kleinanzeigen.de search scraper.

Parser tests run against a real captured results page
(fixtures/kleinanzeigen_srp.html). The previous tests used hand-written HTML
built from the scraper's own (by then dead) selectors, so they stayed green
while the live scraper returned nothing at all.
"""

import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
import requests
from bs4 import BeautifulSoup

import kleinanzeigen_scraper as ka
from kleinanzeigen_scraper import KleinanzeigenScraper, _devalue, _parse_price
from models import Condition

_FIXTURE = (Path(__file__).parent / "fixtures" / "kleinanzeigen_srp.html").read_text(encoding="utf-8")
_BERLIN = ZoneInfo("Europe/Berlin")


@pytest.fixture
def scraper():
    return KleinanzeigenScraper()


@pytest.fixture
def deals(scraper):
    deals, errors = scraper.parse_results_page(_FIXTURE)
    assert errors == []
    return {d["listing_id"]: d for d in deals}


def _without_structured_data(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for island in soup.select("astro-island"):
        island.decompose()
    return str(soup)


class TestStructuredDataParsing:
    def test_real_ads_parsed_and_sponsored_slot_skipped(self, deals):
        # The fixture holds 7 result entries, one of them an empty sponsored-ad slot.
        assert len(deals) == 6

    def test_ad_fields(self, deals):
        d = deals["kleinanzeigen:3521619922"]
        assert d["title"] == "Resident Evil Revelations Xbox 360 Sealed Sammlung in OVP FSK16"
        assert d["url"] == (
            "https://www.kleinanzeigen.de/s-anzeige/resident-evil-revelations-xbox-360-sealed-sammlung-in-ovp-fsk16/"
            "3521619922-227-5751"
        )
        assert d["price"] == 45.0
        assert d["shipping_note"] == "VB"
        assert d["item_location"] == "92421 Schwandorf"
        assert d["description"].startswith("Biete hier diesen Top Titel an.")
        assert d["condition_normalized"] == Condition.NEW  # "OVP" in the title

    def test_umlauts_decoded(self, deals):
        assert deals["kleinanzeigen:3521611226"]["title"].startswith("Große Sammlung")

    def test_german_thousands_separator(self, deals):
        assert deals["kleinanzeigen:3522355374"]["price"] == 1200.0  # "1.200 €"

    def test_negotiable_without_amount(self, deals):
        d = deals["kleinanzeigen:3521611226"]  # price text is just "VB"
        assert d["price"] == 0.0
        assert d["shipping_note"] == "VB"

    def test_shipping_is_real_availability_info(self, deals):
        assert deals["kleinanzeigen:3521619922"]["shipping"] == "Versand möglich"
        assert deals["kleinanzeigen:3521143176"]["shipping"] == "Nur Abholung"
        assert all(d["shipping_cost"] is None for d in deals.values())

    def test_relative_date_is_german_local_time(self, deals):
        """ "Gestern, 23:18" on the page means 23:18 in Germany."""
        listed = datetime.fromisoformat(deals["kleinanzeigen:3522355374"]["listing_date"]).astimezone(_BERLIN)
        assert (listed.hour, listed.minute) == (23, 18)

    def test_absolute_date(self, deals):
        listed = datetime.fromisoformat(deals["kleinanzeigen:3521143176"]["listing_date"]).astimezone(_BERLIN)
        assert (listed.year, listed.month, listed.day, listed.hour) == (2026, 9, 23, 0)

    def test_full_size_images(self, deals):
        urls = deals["kleinanzeigen:3522355374"]["image_urls"]
        assert len(urls) == 2
        assert all(u.startswith("https://img.kleinanzeigen.de/") and u.endswith("?rule=$_59.AUTO") for u in urls)

    def test_no_fabricated_seller_rating(self, deals):
        """Search results show no seller rating; the old scraper invented 85/90/100."""
        assert all(d["seller_rating"] == 0.0 for d in deals.values())

    def test_max_results(self, scraper):
        deals, _ = scraper.parse_results_page(_FIXTURE, max_results=2)
        assert len(deals) == 2


class TestHtmlFallback:
    def test_html_cards_match_structured_data(self, scraper, deals):
        """Without the embedded data, the HTML cards must yield the same deals."""
        fallback, errors = scraper.parse_results_page(_without_structured_data(_FIXTURE))
        assert errors == []
        fields = ("title", "price", "shipping", "shipping_note", "listing_date", "item_location", "url")
        for d in fallback:
            assert {f: d[f] for f in fields} == {f: deals[d["listing_id"]][f] for f in fields}
        assert len(fallback) == len(deals)

    def test_wanted_ads_skipped(self, scraper):
        soup = BeautifulSoup(_without_structured_data(_FIXTURE), "html.parser")
        first = soup.select_one("article[data-adid]")
        first.append(BeautifulSoup("<span>Gesuch</span>", "html.parser"))
        deals, _ = scraper.parse_results_page(str(soup))
        assert first["data-adid"] not in {d["listing_id"].split(":")[1] for d in deals}

    def test_unparseable_page_with_results_is_reported(self, scraper):
        html = '<span id="srp-breadcrumb-summary">1 - 25 von 127 Ergebnissen</span><div>new markup</div>'
        deals, errors = scraper.parse_results_page(html)
        assert deals == []
        assert "markup has likely changed" in errors[0]

    def test_genuinely_empty_results_are_not_an_error(self, scraper):
        deals, errors = scraper.parse_results_page("<html><body>Keine Ergebnisse</body></html>")
        assert deals == []
        assert errors == []


class TestSearchRequest:
    def _ok(self, content: str = _FIXTURE):
        return MagicMock(ok=True, status_code=200, content=content.encode("utf-8"))

    def test_searches_video_games_category(self, scraper):
        with (
            patch.object(scraper._session, "get", return_value=self._ok()) as get,
            patch.object(scraper, "_rate_limit"),
        ):
            deals, errors = scraper.search("Xbox 360 Spiele Sammlung")
        assert get.call_args.args[0] == "https://www.kleinanzeigen.de/s-pc-videospiele/xbox-360-spiele-sammlung/k0c227"
        assert len(deals) == 6
        assert errors == []

    def test_umlaut_query_is_url_encoded(self, scraper):
        with (
            patch.object(scraper._session, "get", return_value=self._ok()) as get,
            patch.object(scraper, "_rate_limit"),
        ):
            scraper.search("Spiele für PS4")
        assert "spiele-f%C3%BCr-ps4" in get.call_args.args[0]

    def test_empty_query(self, scraper):
        assert scraper.search("  ") == ([], ["query is required"])

    def test_ip_ban_pauses_the_source(self, scraper):
        """After a 403 no further requests go out until the cooldown passes —
        retrying into an IP ban only extends it."""
        blocked = MagicMock(ok=False, status_code=403)
        with patch.object(scraper._session, "get", return_value=blocked) as get, patch.object(scraper, "_rate_limit"):
            _, first_errors = scraper.search("xbox")
            deals, second_errors = scraper.search("xbox")
        assert get.call_count == 1
        assert deals == []
        assert "pausing" in first_errors[0]
        assert "paused" in second_errors[0]

    def test_request_exception(self, scraper):
        with (
            patch.object(scraper._session, "get", side_effect=requests.RequestException("boom")),
            patch.object(scraper, "_rate_limit"),
        ):
            deals, errors = scraper.search("xbox")
        assert deals == []
        assert "boom" in errors[0]


_AD_URL = "https://www.kleinanzeigen.de/s-anzeige/10-playstation-3-spiele-sammlung-6-ps-4-spiele/3522392115-227-26832"
_AD_PAGE = (Path(__file__).parent / "fixtures" / "kleinanzeigen_ad.html").read_text(encoding="utf-8")
_AD_PAGE_NEW_LAYOUT = (Path(__file__).parent / "fixtures" / "kleinanzeigen_ad_new_layout.html").read_text(
    encoding="utf-8"
)


class TestAdDescription:
    """Search results cut descriptions at ~100 characters; the ad page has the rest."""

    def test_both_page_layouts(self):
        classic = ka.parse_ad_description(_AD_PAGE)
        assert classic.startswith("Ich biete hier eine Sammlung von 10 gebrauchten PlayStation 3 Spielen")
        assert "\n\nStück preis 7euro VB oder komplett paket 120 euro inkl versand\n\n" in classic
        assert classic.endswith("schreib mir gerne eine Nachricht.")
        newer = ka.parse_ad_description(_AD_PAGE_NEW_LAYOUT)
        assert newer.startswith("Ich verkaufe das Spiel Battlefield 1 für die PlayStation 4.")
        assert "\n- Plattform: PlayStation 4\n" in newer
        assert ka.parse_ad_description("<html><body>no ad</body></html>") is None

    def test_fetched_once_then_cached(self, scraper):
        page = MagicMock(ok=True, status_code=200, content=_AD_PAGE.encode("utf-8"))
        with patch.object(scraper._session, "get", return_value=page) as get, patch.object(scraper, "_rate_limit"):
            assert scraper.fetch_description(_AD_URL, cached_only=True) == (None, [])  # miss: no request
            text, errors = scraper.fetch_description(_AD_URL)
            again, _ = scraper.fetch_description(_AD_URL, cached_only=True)
        assert get.call_count == 1
        assert errors == []
        assert "komplett paket 120 euro" in text
        assert again == text

    def test_only_kleinanzeigen_ad_urls(self, scraper):
        with patch.object(scraper._session, "get") as get:
            text, errors = scraper.fetch_description("https://evil.example/s-anzeige/x/1-2-3")
        assert text is None
        assert "Not a Kleinanzeigen ad URL" in errors[0]
        get.assert_not_called()

    def test_ip_ban_pauses_searches_too(self, scraper):
        blocked = MagicMock(ok=False, status_code=403)
        with patch.object(scraper._session, "get", return_value=blocked) as get, patch.object(scraper, "_rate_limit"):
            _, first = scraper.fetch_description(_AD_URL)
            _, second = scraper.search("xbox")
        assert get.call_count == 1
        assert "pausing" in first[0]
        assert "paused" in second[0]

    def test_page_without_description_is_reported(self, scraper):
        page = MagicMock(ok=True, status_code=200, content=b"<html><body>new markup</body></html>")
        with patch.object(scraper._session, "get", return_value=page), patch.object(scraper, "_rate_limit"):
            text, errors = scraper.fetch_description(_AD_URL)
        assert text is None
        assert "markup has likely changed" in errors[0]


class TestRateLimit:
    def test_concurrent_callers_are_spaced_apart(self, scraper, monkeypatch):
        monkeypatch.setattr(ka, "_REQUEST_SPACING_S", 0.3)
        stamps = []
        barrier = threading.Barrier(2, timeout=2)

        def call():
            barrier.wait()
            scraper._rate_limit()
            stamps.append(time.monotonic())

        threads = [threading.Thread(target=call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(stamps) == 2
        assert abs(stamps[1] - stamps[0]) >= 0.25


class TestHelpers:
    def test_parse_price(self):
        assert _parse_price("1.200 €") == (1200.0, False)
        assert _parse_price("45 € VB") == (45.0, True)
        assert _parse_price("VB") == (0.0, True)
        assert _parse_price("1.234,50 €") == (1234.5, False)
        assert _parse_price("Zu verschenken") == (0.0, False)
        assert _parse_price("999999999 €") == (0.0, False)

    def test_devalue(self):
        raw = [0, {"a": [0, 1], "b": [1, [[0, "x"], [0, "y"]]], "c": [0, {"d": [0, None]}]}]
        assert _devalue(raw) == {"a": 1, "b": ["x", "y"], "c": {"d": None}}


@pytest.mark.live
def test_live_kleinanzeigen_search():
    """Opt-in drift detector (`pytest -m live`): hits the real kleinanzeigen.de."""
    deals, errors = KleinanzeigenScraper().search("xbox 360 spiele sammlung", max_results=10)
    assert deals, errors
    assert all(d["title"] and d["url"].startswith("https://www.kleinanzeigen.de/s-anzeige/") for d in deals)
    assert sum(d["price"] > 0 for d in deals) >= len(deals) // 2
    assert all(d["listing_date"] for d in deals)
    newest = max(datetime.fromisoformat(d["listing_date"]) for d in deals)
    assert datetime.now(UTC) - newest < timedelta(days=30)
