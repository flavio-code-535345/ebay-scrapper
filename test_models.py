"""Tests for models.py — shared deal schema, condition normalization, date parsing, sort key."""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from models import Condition, canonical_listing_id, normalize_condition, parse_listing_date, sort_key_for_deal

_BERLIN = ZoneInfo("Europe/Berlin")
# A fixed "now" (summer time, UTC+2) so date tests don't depend on the clock.
_NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


class TestNormalizeCondition:
    def test_empty_or_none_is_unknown(self):
        assert normalize_condition(None) == Condition.UNKNOWN
        assert normalize_condition("") == Condition.UNKNOWN

    def test_exact_api_labels(self):
        assert normalize_condition("New") == Condition.NEW
        assert normalize_condition("Very Good") == Condition.VERY_GOOD
        assert normalize_condition("Good") == Condition.GOOD
        assert normalize_condition("Acceptable") == Condition.ACCEPTABLE
        assert normalize_condition("Used") == Condition.USED
        assert normalize_condition("For parts or not working") == Condition.FOR_PARTS
        assert normalize_condition("Manufacturer Refurbished") == Condition.REFURBISHED
        assert normalize_condition("Seller Refurbished") == Condition.REFURBISHED
        assert normalize_condition("New with defects") == Condition.NEW_WITH_DEFECTS
        assert normalize_condition("New – Other") == Condition.NEW_OTHER

    def test_exact_kleinanzeigen_labels(self):
        assert normalize_condition("Neu") == Condition.NEW
        assert normalize_condition("Sehr gut") == Condition.VERY_GOOD
        assert normalize_condition("Gebraucht") == Condition.USED
        assert normalize_condition("Defekt") == Condition.FOR_PARTS

    def test_scraper_free_text_keyword_fallback(self):
        """The HTML scraper's raw text doesn't exactly match any fixed label set."""
        assert normalize_condition("Gebraucht - Akzeptabler Zustand") == Condition.ACCEPTABLE
        assert normalize_condition("Sehr guter Zustand") == Condition.VERY_GOOD
        assert normalize_condition("Für Ersatzteile") == Condition.FOR_PARTS
        assert normalize_condition("Unknown") == Condition.UNKNOWN

    def test_case_insensitive(self):
        assert normalize_condition("NEU") == Condition.NEW
        assert normalize_condition("very good") == Condition.VERY_GOOD

    def test_ebay_card_condition_strings(self):
        """eBay's current result cards: '<condition> | <seller type>'."""
        assert normalize_condition("Gebraucht | Privat") == Condition.USED
        assert normalize_condition("Neu (Sonstige) | Gewerblich") == Condition.NEW_OTHER
        assert normalize_condition("Nur Ersatzteile | Privat") == Condition.FOR_PARTS
        assert normalize_condition("Gut - Refurbished") == Condition.REFURBISHED


class TestParseListingDate:
    def test_none_or_empty(self):
        assert parse_listing_date(None, "api") is None
        assert parse_listing_date("", "kleinanzeigen") is None

    def test_scraper_relative_age(self):
        """eBay's newest-first result cards show a coarse relative age."""
        assert parse_listing_date("Vor 2 Std. eingestellt", "scraper", now=_NOW) == _NOW - timedelta(hours=2)
        assert parse_listing_date("Vor 30 Min. eingestellt", "scraper", now=_NOW) == _NOW - timedelta(minutes=30)
        assert parse_listing_date("Vor 1 T. eingestellt", "scraper", now=_NOW) == _NOW - timedelta(days=1)

    def test_scraper_absolute_date_for_older_listings(self):
        """ "Eingestellt am Sep 20" — German local midnight; eBay mixes English
        and German month abbreviations."""
        assert parse_listing_date("Eingestellt am Sep 20", "scraper", now=_NOW) == datetime(2026, 9, 20, tzinfo=_BERLIN)
        assert parse_listing_date("Eingestellt am Okt 3", "scraper", now=_NOW) == datetime(2025, 10, 3, tzinfo=_BERLIN)

    def test_scraper_absolute_date_rolls_back_a_year(self):
        """No year is shown: a date later than today means last year."""
        january = datetime(2027, 1, 5, 12, 0, tzinfo=UTC)
        assert parse_listing_date("Eingestellt am Dez 30", "scraper", now=january) == datetime(
            2026, 12, 30, tzinfo=_BERLIN
        )

    def test_scraper_non_age_text_is_none(self):
        """Anything that isn't a relative age is never turned into a date."""
        assert parse_listing_date("2024-01-01T00:00:00Z", "scraper") is None
        assert parse_listing_date("Sofort-Kaufen", "scraper") is None

    def test_api_iso8601(self):
        dt = parse_listing_date("2024-03-01T10:00:00.000Z", "api")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 1

    def test_api_unparsable_returns_none(self):
        assert parse_listing_date("not a date", "api") is None

    def test_kleinanzeigen_today_is_german_local_time(self):
        """'Heute, 19:30' is 19:30 in Germany — 17:30 UTC in summer, not 19:30 UTC."""
        dt = parse_listing_date("Heute, 19:30", "kleinanzeigen", now=_NOW)
        assert dt == datetime(2026, 9, 25, 19, 30, tzinfo=_BERLIN)
        assert dt.utcoffset() == timedelta(0)
        assert dt.hour == 17

    def test_kleinanzeigen_yesterday(self):
        dt = parse_listing_date("Gestern, 10:15", "kleinanzeigen", now=_NOW)
        assert dt == datetime(2026, 9, 24, 10, 15, tzinfo=_BERLIN)

    def test_kleinanzeigen_today_uses_german_calendar_day(self):
        """Just after midnight in Germany it's still the previous day in UTC —
        'Heute' must mean the German day."""
        just_after_german_midnight = datetime(2026, 9, 24, 22, 30, tzinfo=UTC)  # 00:30 on the 25th in Berlin
        dt = parse_listing_date("Heute, 00:10", "kleinanzeigen", now=just_after_german_midnight)
        assert dt == datetime(2026, 9, 25, 0, 10, tzinfo=_BERLIN)

    def test_kleinanzeigen_absolute_date(self):
        dt = parse_listing_date("05.09.2025", "kleinanzeigen")
        assert dt == datetime(2025, 9, 5, tzinfo=_BERLIN)

    def test_kleinanzeigen_unparsable_returns_none(self):
        assert parse_listing_date("some weird text", "kleinanzeigen") is None


class TestSortKeyForDeal:
    def test_must_have_sorts_before_others(self):
        must_have = {"ai_deal_rating": "Must Have", "listing_date": None}
        good = {"ai_deal_rating": "Good", "listing_date": None}
        assert sort_key_for_deal(must_have) < sort_key_for_deal(good)

    def test_must_buy_treated_same_as_must_have(self):
        a = {"ai_deal_rating": "Must Buy", "listing_date": None}
        b = {"ai_deal_rating": "Must Have", "listing_date": None}
        assert sort_key_for_deal(a)[0] == sort_key_for_deal(b)[0] == 0

    def test_newer_dated_deal_sorts_before_older_dated_deal(self):
        newer = {"ai_deal_rating": "Good", "listing_date": "2024-06-01T00:00:00+00:00"}
        older = {"ai_deal_rating": "Good", "listing_date": "2024-01-01T00:00:00+00:00"}
        assert sort_key_for_deal(newer) < sort_key_for_deal(older)

    def test_dated_deal_sorts_before_undated_deal_regardless_of_recency(self):
        """The bug this fixes: an undated deal must never rank above a dated
        one just because the old sort collapsed undated to datetime.min —
        the important direction is dated-before-undated within a tier, and
        this asserts exactly that ordering."""
        dated_old = {"ai_deal_rating": "Good", "listing_date": "2020-01-01T00:00:00+00:00"}
        undated = {"ai_deal_rating": "Good", "listing_date": None}
        assert sort_key_for_deal(dated_old) < sort_key_for_deal(undated)

    def test_unparsable_date_treated_as_undated(self):
        bad = {"ai_deal_rating": "Good", "listing_date": "not a date"}
        undated = {"ai_deal_rating": "Good", "listing_date": None}
        assert sort_key_for_deal(bad) == sort_key_for_deal(undated)


class TestCanonicalListingId:
    def test_same_ebay_listing_across_url_shapes(self):
        """A Browse-API itemWebUrl and a search-results link with tracking
        parameters must resolve to the same identity."""
        api_url = "https://www.ebay.de/itm/206580175564"
        web_url = "https://www.ebay.de/itm/206580175564?_skw=xbox+360&hash=item30192352cc"
        slug_url = "https://www.ebay.de/itm/DJ-Hero-2-Turntable/206580175564"
        assert canonical_listing_id(api_url) == "ebay:206580175564"
        assert canonical_listing_id(web_url) == "ebay:206580175564"
        assert canonical_listing_id(slug_url) == "ebay:206580175564"

    def test_kleinanzeigen_ad(self):
        url = "https://www.kleinanzeigen.de/s-anzeige/xbox-360-spiele-sammlung/3521619922-227-5751"
        assert canonical_listing_id(url) == "kleinanzeigen:3521619922"

    def test_unrecognised_url(self):
        assert canonical_listing_id("https://example.com/whatever") is None
        assert canonical_listing_id("") is None
        assert canonical_listing_id(None) is None
