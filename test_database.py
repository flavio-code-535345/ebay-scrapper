"""Tests for database.py — SQLite persistence layer."""

import pytest

import database


@pytest.fixture(autouse=True)
def _temp_db(tmp_path):
    """Use a temporary database for each test."""
    db_file = tmp_path / "test.db"
    old_path = database.DB_PATH
    database.DB_PATH = str(db_file)
    database.init_db()
    yield
    database.DB_PATH = old_path


class TestInitDB:
    def test_tables_created(self):
        """After init_db the expected tables exist."""
        with database.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row["name"] for row in cursor.fetchall()}
        assert "searches" in tables
        assert "deals" in tables
        assert "settings" in tables
        assert "user_saved_deals" in tables
        assert "user_skipped_deals" in tables

    def test_re_init_is_idempotent(self):
        """Calling init_db twice does not raise."""
        database.init_db()

    def test_deals_indexes_exist(self):
        """search_id (FK, filtered constantly) and created_at (ORDER BY in
        CSV export) must be indexed — both were previously unindexed."""
        with database.get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='deals'")
            indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_deals_search_id" in indexes
        assert "idx_deals_created_at" in indexes


class TestSettings:
    def test_get_setting_default(self):
        assert database.get_setting("nonexistent") is None

    def test_get_setting_custom_default(self):
        assert database.get_setting("nonexistent", "fallback") == "fallback"

    def test_set_and_get(self):
        database.set_setting("theme", "dark")
        assert database.get_setting("theme") == "dark"

    def test_upsert_updates_value(self):
        database.set_setting("key", "v1")
        database.set_setting("key", "v2")
        assert database.get_setting("key") == "v2"


class TestSaveSearch:
    def test_save_search_returns_id(self):
        search_id = database.save_search("test query", [])
        assert isinstance(search_id, int)
        assert search_id > 0

    def test_save_search_with_deals(self):
        deals = [
            {
                "title": "Game A",
                "price": 10.0,
                "condition": "Used",
                "seller_rating": 98.5,
                "url": "http://ebay.com/itm/1",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "Berlin, DE",
                "image_urls": ["http://example.com/img.jpg"],
                "image_issues": [],
                "listing_date": "2024-01-01T00:00:00Z",
            }
        ]
        database.save_search("query", deals)
        history = database.get_history()
        assert len(history) == 1
        assert history[0]["query"] == "query"
        assert history[0]["result_count"] == 1

    def test_get_deals_by_search(self):
        deals = [
            {
                "title": "Game X",
                "price": 5.0,
                "condition": "New",
                "seller_rating": 99.0,
                "url": "http://ebay.com/itm/2",
                "shipping": "€3.00",
                "is_trending": True,
                "item_location": "DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            }
        ]
        sid = database.save_search("xyz", deals)
        fetched = database.get_deals_by_search(sid)
        assert len(fetched) == 1
        assert fetched[0]["title"] == "Game X"
        assert fetched[0]["price"] == 5.0

    def test_price_note_is_kept(self):
        """History shows the same "listed price was per game" note the live search did."""
        note = "Listed €7.00 is the price per game ('Stück preis'); the whole lot costs €120.00."
        sid = database.save_search("q", [{"title": "Bundle", "price": 120.0, "url": "http://x/1", "price_note": note}])
        assert database.get_deals_by_search(sid)[0]["price_note"] == note

    def test_list_fields_round_trip_as_lists(self):
        """List-typed fields must always come back as lists after a
        save/load round trip — never a raw string, regardless of what shape
        they arrived in (a None-vs-list encoding inconsistency previously
        made this depend on the input type)."""
        deals = [
            {
                "title": "Bundle Game",
                "price": 20.0,
                "url": "http://ebay.com/itm/3",
                "ai_visual_findings": ["blurry photo"],
                "ai_red_flags": [],
                "image_issues": ["no_images"],
                "image_urls": ["http://x/1.jpg", "http://x/2.jpg"],
                "ai_itemized_resale_estimates": [{"game": "Halo 3", "price_eur": 12.0}],
            }
        ]
        sid = database.save_search("list-fields", deals)
        fetched = database.get_deals_by_search(sid)[0]
        assert fetched["ai_visual_findings"] == ["blurry photo"]
        assert fetched["ai_red_flags"] == []
        assert fetched["image_issues"] == ["no_images"]
        assert fetched["image_urls"] == ["http://x/1.jpg", "http://x/2.jpg"]
        assert fetched["ai_itemized_resale_estimates"] == [{"game": "Halo 3", "price_eur": 12.0}]

    def test_missing_list_fields_are_none_not_raw_garbage(self):
        """A deal that never sets a list-typed field at all stores/loads as
        None, not some other inconsistent representation."""
        deals = [{"title": "No AI data", "price": 1.0, "url": "http://ebay.com/itm/4"}]
        sid = database.save_search("missing-fields", deals)
        fetched = database.get_deals_by_search(sid)[0]
        assert fetched["ai_visual_findings"] is None
        assert fetched["image_urls"] is None

    def test_deals_with_ai_fields(self):
        deals = [
            {
                "title": "Bundle",
                "price": 20.0,
                "condition": "Used",
                "seller_rating": 95.0,
                "url": "http://ebay.com/itm/3",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "DE",
                "image_urls": ["http://ex.com/a.jpg"],
                "image_issues": [],
                "listing_date": "2024-06-01T00:00:00Z",
                "ai_deal_rating": "Must Have",
                "ai_confidence_score": 90,
                "ai_visual_findings": ["Minor scratches"],
                "ai_red_flags": [],
                "ai_fair_market_estimate": "~€30-40",
                "ai_verdict_summary": "Great deal!",
                "ai_assessed": True,
                "ai_potential_scam": False,
                "ai_scam_warning": "",
                "ai_itemized_resale_estimates": [{"game": "Game1", "price_eur": 15.0, "price_source": "ebay_sold"}],
                "ai_estimated_total_cost": 20.0,
                "ai_estimated_gross_profit": 10.0,
            }
        ]
        sid = database.save_search("bundle", deals)
        fetched = database.get_deals_by_search(sid)
        assert fetched[0]["ai_deal_rating"] == "Must Have"
        assert fetched[0]["ai_estimated_gross_profit"] == 10.0
        estimates = fetched[0]["ai_itemized_resale_estimates"]
        assert estimates[0]["game"] == "Game1"


class TestSaveUnsaveDeal:
    def test_save_and_get_saved(self):
        database.save_deal("http://ex.com/d1", "Game A", 10.0)
        saved = database.get_saved_deals()
        assert len(saved) == 1
        assert saved[0]["url"] == "http://ex.com/d1"

    def test_unsave_removes(self):
        database.save_deal("http://ex.com/d2", "Game B", 5.0)
        database.unsave_deal("http://ex.com/d2")
        assert database.get_saved_deals() == []

    def test_upsert_updates(self):
        database.save_deal("http://ex.com/u1", "Old", 1.0)
        database.save_deal("http://ex.com/u1", "New", 2.0)
        saved = database.get_saved_deals()
        assert len(saved) == 1
        assert saved[0]["title"] == "New"


class TestSkipUnskipDeal:
    def test_skip_and_get_skipped(self):
        database.skip_deal("http://ex.com/s1", "Bad Deal", 50.0)
        skipped = database.get_skipped_deals()
        assert len(skipped) == 1
        assert skipped[0]["url"] == "http://ex.com/s1"

    def test_get_skipped_urls(self):
        database.skip_deal("http://ex.com/s2", "Meh", 10.0)
        database.skip_deal("http://ex.com/s3", "Nah", 5.0)
        urls = database.get_skipped_deal_urls()
        assert len(urls) == 2
        assert "http://ex.com/s2" in urls

    def test_unskip_removes(self):
        database.skip_deal("http://ex.com/s4", "Skip", 1.0)
        database.unskip_deal("http://ex.com/s4")
        assert database.get_skipped_deals() == []


class TestExportCSV:
    def test_export_empty(self):
        csv = database.export_csv()
        assert csv == ""

    def test_export_with_data(self):
        deals = [
            {
                "title": "Ex Game",
                "price": 15.0,
                "condition": "Used",
                "seller_rating": 97.0,
                "url": "http://ex.com/export1",
                "shipping": "Free",
                "is_trending": False,
                "item_location": "DE",
                "image_urls": [],
                "image_issues": [],
                "listing_date": None,
            }
        ]
        database.save_search("export query", deals)
        csv = database.export_csv()
        assert "Ex Game" in csv
        assert "export query" in csv


class TestStats:
    def test_stats_empty(self):
        stats = database.get_stats()
        assert stats["total_searches"] == 0
        assert stats["total_deals"] == 0

    def test_stats_with_data(self):
        database.save_search(
            "s1",
            [
                {
                    "title": "A",
                    "price": 1.0,
                    "condition": "U",
                    "seller_rating": 90.0,
                    "url": "u1",
                    "shipping": "F",
                    "is_trending": False,
                    "item_location": "DE",
                    "image_urls": [],
                    "image_issues": [],
                    "listing_date": None,
                }
            ],
        )
        stats = database.get_stats()
        assert stats["total_searches"] == 1
        assert stats["total_deals"] == 1
