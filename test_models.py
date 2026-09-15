"""Tests for models.py — shared deal schema, condition normalization, date parsing, sort key."""

from datetime import UTC, datetime, timedelta

from models import Condition, normalize_condition, parse_listing_date, sort_key_for_deal


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


class TestParseListingDate:
    def test_none_or_empty(self):
        assert parse_listing_date(None, "api") is None
        assert parse_listing_date("", "kleinanzeigen") is None

    def test_scraper_source_always_none(self):
        """The HTML scraper's search page exposes no listing date at all —
        never fabricate one even if a raw value somehow showed up."""
        assert parse_listing_date("2024-01-01T00:00:00Z", "scraper") is None

    def test_api_iso8601(self):
        dt = parse_listing_date("2024-03-01T10:00:00.000Z", "api")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 1

    def test_api_unparsable_returns_none(self):
        assert parse_listing_date("not a date", "api") is None

    def test_kleinanzeigen_today(self):
        dt = parse_listing_date("Heute, 19:30", "kleinanzeigen")
        assert dt is not None
        now = datetime.now(UTC)
        assert dt.year == now.year and dt.month == now.month and dt.day == now.day
        assert dt.hour == 19
        assert dt.minute == 30

    def test_kleinanzeigen_yesterday(self):
        dt = parse_listing_date("Gestern, 10:15", "kleinanzeigen")
        assert dt is not None
        expected_day = (datetime.now(UTC) - timedelta(days=1)).day
        assert dt.day == expected_day
        assert dt.hour == 10
        assert dt.minute == 15

    def test_kleinanzeigen_absolute_date(self):
        dt = parse_listing_date("05.09.2025", "kleinanzeigen")
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 9
        assert dt.day == 5

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
