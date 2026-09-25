"""Tests for search/pipeline.py — fetch, dedupe, filter, rank and select."""

import time
from datetime import UTC, datetime, timedelta

from search.pipeline import SourceJob, dedupe, fetch_all, game_count, is_german_location, run_search, select
from search.query import plan_search

_NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
_XBOX_PLAN = plan_search(["Xbox 360 Spiele Sammlung"])


def _deal(title="Xbox 360 Spiele Sammlung", url="https://www.ebay.de/itm/100000001", **extra):
    return {"title": title, "url": url, "price": 20.0, "item_location": "", **extra}


def _fn(deals, errors=(), delay=0.0):
    def search(query, max_results=50):
        time.sleep(delay)
        return [dict(d) for d in deals], list(errors)

    return search


class TestFetchBudget:
    def test_stalled_source_does_not_delay_the_search(self):
        """A source that overruns the budget is reported and dropped — it must
        not hold the response (and the AI's share of the deadline) hostage."""
        jobs = [
            SourceJob("ebay", _fn([_deal()]), "q", 10),
            SourceJob("kleinanzeigen", _fn([_deal(url="https://x/2")], delay=3.0), "q", 10),
        ]
        t0 = time.monotonic()
        results, reports = fetch_all(jobs, budget_s=0.5)
        assert time.monotonic() - t0 < 1.5
        assert [job.source for job, _ in results] == ["ebay"]
        stalled = next(r for r in reports if r.source == "kleinanzeigen")
        assert stalled.timed_out is True
        assert "search budget" in stalled.errors[0]

    def test_source_exception_becomes_an_error_not_a_crash(self):
        def boom(query, max_results=50):
            raise RuntimeError("kaputt")

        results, reports = fetch_all([SourceJob("ebay", boom, "q", 10)], budget_s=2)
        assert results == []
        assert "kaputt" in reports[0].errors[0]

    def test_reports_aggregate_per_source(self):
        jobs = [
            SourceJob("kleinanzeigen", _fn([_deal(url="https://x/1")], errors=["warn"]), "a", 10),
            SourceJob("kleinanzeigen", _fn([_deal(url="https://x/2"), _deal(url="https://x/3")]), "b", 10),
        ]
        _, reports = fetch_all(jobs, budget_s=2)
        (report,) = reports
        assert report.queries == ["a", "b"]
        assert report.count == 3
        assert report.errors == ["warn"]

    def test_results_keep_job_order_regardless_of_finish_order(self):
        jobs = [
            SourceJob("ebay", _fn([_deal(url="https://x/slow")], delay=0.3), "q", 10),
            SourceJob("kleinanzeigen", _fn([_deal(url="https://x/fast")]), "q", 10),
        ]
        results, _ = fetch_all(jobs, budget_s=2)
        assert [job.source for job, _ in results] == ["ebay", "kleinanzeigen"]


class TestDedupe:
    def _job(self, source="ebay"):
        return SourceJob(source, _fn([]), "q", 10)

    def test_same_ebay_listing_via_api_and_web_urls(self):
        api = _deal(url="https://www.ebay.de/itm/206580175564", title="A")
        web = _deal(url="https://www.ebay.de/itm/206580175564?_skw=xbox&hash=item30", title="A (web)")
        merged = dedupe([(self._job(), [api]), (self._job(), [web])])
        assert len(merged) == 1
        assert merged[0]["listing_id"] == "ebay:206580175564"

    def test_cross_posted_listing_same_title_and_price(self):
        ebay = _deal(title="Xbox 360 Spiele Sammlung!", url="https://www.ebay.de/itm/100000001")
        ka = _deal(title="xbox 360 spiele sammlung", url="https://www.kleinanzeigen.de/s-anzeige/x/123-227-1")
        merged = dedupe([(self._job(), [ebay]), (self._job("kleinanzeigen"), [ka])])
        assert len(merged) == 1

    def test_same_title_different_price_kept(self):
        a = _deal(url="https://www.ebay.de/itm/100000001")
        b = _deal(url="https://www.ebay.de/itm/100000002", price=35.0)
        assert len(dedupe([(self._job(), [a, b])])) == 2

    def test_source_and_listing_type_defaults(self):
        (ka,) = dedupe([(self._job("kleinanzeigen"), [_deal(url="https://x/1")])])
        (auction,) = dedupe([(self._job("ebay_auctions"), [_deal(url="https://x/2")])])
        assert ka["source"] == "kleinanzeigen"
        assert auction["source"] == "ebay"
        assert auction["listing_type"] == "auction"


class TestFilters:
    def _run(self, deals, plan=_XBOX_PLAN, skipped=frozenset()):
        return run_search(
            plan,
            ebay_search=_fn(deals),
            ebay_is_api=True,
            auction_search=None,
            kleinanzeigen_search=None,
            skipped=set(skipped),
            budget_s=2,
        )

    def test_platform_guard(self):
        deals = [
            _deal(title="PS4 Spiele Sammlung", url="https://www.ebay.de/itm/100000001"),  # other platform → dropped
            _deal(title="Xbox 360 und PS4 Sammlung", url="https://www.ebay.de/itm/100000002"),  # includes ours
            _deal(title="Konvolut Videospiele", url="https://www.ebay.de/itm/100000003"),  # no platform named
            _deal(title="Xbox Spiele Paket", url="https://www.ebay.de/itm/100000004"),  # generic family
        ]
        outcome = self._run(deals)
        titles = {d["title"] for d in outcome.selected}
        assert titles == {"Xbox 360 und PS4 Sammlung", "Konvolut Videospiele", "Xbox Spiele Paket"}
        assert outcome.removed["other_platform"] == 1

    def test_no_platform_in_query_means_no_guard(self):
        outcome = self._run([_deal(title="PS4 Spiele")], plan=plan_search(["Spiele Sammlung"]))
        assert len(outcome.selected) == 1

    def test_skipped_by_listing_id_even_with_a_different_url_shape(self):
        deal = _deal(url="https://www.ebay.de/itm/206580175564")
        outcome = self._run([deal], skipped={"ebay:206580175564"})
        assert outcome.selected == []
        assert outcome.removed["skipped"] == 1

    def test_non_german_ebay_item_dropped(self):
        outcome = self._run(
            [_deal(item_location="Großbritannien"), _deal(url="https://x/2", price=25.0, item_location="DE")]
        )
        assert len(outcome.selected) == 1
        assert outcome.removed["not_germany"] == 1

    def test_sports_only_dropped_but_mixed_bundle_kept(self):
        deals = [
            _deal(title="FIFA 19 Xbox 360", url="https://x/1"),
            _deal(title="Xbox 360 Sammlung FIFA Halo Gears Fable Mass Effect", url="https://x/2"),
        ]
        outcome = self._run(deals)
        assert [d["url"] for d in outcome.selected] == ["https://x/2"]

    def test_german_location_helper(self):
        assert is_german_location("") and is_german_location("Berlin, DE") and is_german_location("Deutschland")
        assert not is_german_location("New York, US")


class TestSelection:
    def test_each_source_is_represented(self):
        """50 strong eBay results must not crowd two Kleinanzeigen ads out of the AI slots."""
        ebay = [
            _deal(
                url=f"https://www.ebay.de/itm/2000000{i:02d}",
                source="ebay",
                listing_type="fixed",
                listing_date=_NOW.isoformat(),
            )
            for i in range(50)
        ]
        ka = [
            _deal(
                url=f"https://www.kleinanzeigen.de/s-anzeige/x/{i}-227-1",
                source="kleinanzeigen",
                title="Konvolut",
                listing_date=(_NOW - timedelta(days=20)).isoformat(),
            )
            for i in range(2)
        ]
        chosen = select(ebay + ka, _XBOX_PLAN, limit=30, now=_NOW)
        assert len(chosen) == 30
        assert sum(d["source"] == "kleinanzeigen" for d in chosen) == 2

    def test_fresher_and_cheaper_per_game_ranks_higher(self):
        old = _deal(url="https://x/old", listing_date=(_NOW - timedelta(days=30)).isoformat())
        fresh = _deal(url="https://x/fresh", listing_date=_NOW.isoformat())
        assert [d["url"] for d in select([old, fresh], _XBOX_PLAN, limit=2, now=_NOW)] == [
            "https://x/fresh",
            "https://x/old",
        ]

        pricey = _deal(url="https://x/pricey", title="Xbox 360 Sammlung 10 Spiele", price=100.0)
        cheap = _deal(url="https://x/cheap", title="Xbox 360 Sammlung 50 Spiele", price=50.0)
        assert select([pricey, cheap], _XBOX_PLAN, limit=2, now=_NOW)[0]["url"] == "https://x/cheap"

    def test_deterministic_ties_keep_merge_order(self):
        deals = [_deal(url=f"https://x/{i}") for i in range(5)]
        assert [d["url"] for d in select(deals, _XBOX_PLAN, limit=5, now=_NOW)] == [d["url"] for d in deals]

    def test_limit_respected(self):
        deals = [_deal(url=f"https://x/{i}") for i in range(40)]
        assert len(select(deals, _XBOX_PLAN, limit=30, now=_NOW)) == 30


class TestGameCount:
    def test_counts(self):
        assert game_count("XXL Xbox Spiele Sammlung/ Konvolut 26 Spiele Xbox360") == 26
        assert game_count("ca. 94 Xbox 360 Spiele Sammlung Konvolut") == 94
        assert game_count("Xbox 360 und wii spiel Sammlung 13x Xbox 360") is None  # "13x Xbox" isn't "13 Spiele"
        assert game_count("Xbox 360 Spiele Sammlung 7 Spiele") == 7
        assert game_count("Xbox 360 Konsole") is None
        # A platform's own number is not a game count.
        assert game_count("Xbox 360 Spiele Sammlung") is None
        assert game_count("PS4 Spiele Konvolut") is None
        assert game_count("PS4 Konvolut 12 Spiele") == 12
