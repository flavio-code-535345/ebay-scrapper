"""Tests for search/pipeline.py — fetch, dedupe, filter, rank and select."""

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kleinanzeigen_scraper import parse_ad_description
from search.pipeline import (
    AUCTION_MAX_TIME_LEFT,
    SourceJob,
    apply_filters,
    check_prices,
    complete_descriptions,
    dedupe,
    fetch_all,
    game_count,
    is_german_location,
    run_search,
    select,
)
from search.query import plan_search

_NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
_XBOX_PLAN = plan_search(["Xbox 360 Spiele Sammlung"])
_PS_PLAN = plan_search(["PS3 Spiele Sammlung"])

# A real Kleinanzeigen ad: 16 games listed at "7 € VB" — per game. Its search
# result carried only the first ~100 characters of this description.
_EXAMPLE_TITLE = "10 PlayStation 3 Spiele Sammlung / 6 PS 4 spiele"
_EXAMPLE_DESCRIPTION = parse_ad_description(
    (Path(__file__).parent / "fixtures" / "kleinanzeigen_ad.html").read_text(encoding="utf-8")
)
_EXAMPLE_PREVIEW = _EXAMPLE_DESCRIPTION[:97] + "..."


def _deal(title="Xbox 360 Spiele Sammlung", url="https://www.ebay.de/itm/100000001", **extra):
    return {"title": title, "url": url, "price": 20.0, "item_location": "", **extra}


def _fn(deals, errors=(), delay=0.0):
    def search(query, max_results=50, **kwargs):
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

    def test_end_time_reported_by_another_leg_is_kept(self):
        """An auction with a Buy-It-Now option also shows up in the Buy-It-Now
        leg, whose sort order hides the time left; the auction leg has it."""
        bin_leg = _deal(url="https://www.ebay.de/itm/800701145155", listing_type="auction")
        auction_leg = dict(bin_leg, auction_end=_NOW.isoformat())
        (merged,) = dedupe([(self._job(), [bin_leg]), (self._job("ebay_auctions"), [auction_leg])])
        assert merged["auction_end"] == _NOW.isoformat()

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
            # Seen live: bundle words and platform names aren't "other games".
            _deal(title="FIFA Sammlung Konvolut Xbox 360 X360", url="https://x/3", price=32.0),
        ]
        outcome = self._run(deals)
        assert [d["url"] for d in outcome.selected] == ["https://x/2"]
        assert outcome.removed["sports_only"] == 2

    def test_auctions_must_end_within_two_days(self):
        def auction(n, ends_in):
            end = None if ends_in is None else (_NOW + ends_in).isoformat()
            return _deal(url=f"https://www.ebay.de/itm/30000000{n}", listing_type="auction", auction_end=end)

        deals = [
            _deal(url="https://www.ebay.de/itm/300000000", listing_type="fixed"),
            auction(1, timedelta(hours=5)),
            auction(2, timedelta(hours=47)),
            auction(3, timedelta(hours=49)),  # ends too late
            auction(4, timedelta(days=6)),  # ends too late
            auction(5, timedelta(hours=-1)),  # already over
            auction(6, None),  # end time unknown
        ]
        kept, removed = apply_filters(deals, _XBOX_PLAN, set(), now=_NOW)
        assert [d["url"][-1] for d in kept] == ["0", "1", "2"]
        assert removed["auction_ends_late"] == 4

    def test_api_end_date_format_accepted(self):
        deal = _deal(listing_type="auction", auction_end="2026-09-26T09:00:00.000Z")
        kept, _ = apply_filters([deal], _XBOX_PLAN, set(), now=_NOW)
        assert kept == [deal]

    def test_german_location_helper(self):
        assert is_german_location("") and is_german_location("Berlin, DE") and is_german_location("Deutschland")
        assert not is_german_location("New York, US")


class TestAuctionLeg:
    def _calls(self, ebay_is_api):
        calls = []

        def auctions(query, max_results=50, **kwargs):
            calls.append((query, kwargs))
            return [], []

        run_search(
            _XBOX_PLAN,
            ebay_search=_fn([]),
            ebay_is_api=ebay_is_api,
            auction_search=auctions,
            kleinanzeigen_search=None,
            skipped=set(),
            budget_s=2,
        )
        return calls

    def test_each_engine_searches_auctions_with_its_own_query_and_the_cutoff(self):
        assert self._calls(ebay_is_api=True) == [(_XBOX_PLAN.ebay_api[0], {"ends_within": AUCTION_MAX_TIME_LEFT})]
        assert self._calls(ebay_is_api=False) == [(_XBOX_PLAN.ebay_web[0], {"ends_within": AUCTION_MAX_TIME_LEFT})]

    def test_cutoff_is_two_days(self):
        assert timedelta(days=2) == AUCTION_MAX_TIME_LEFT


def _ka(n, price, title=_EXAMPLE_TITLE, description=_EXAMPLE_PREVIEW, **extra):
    return {
        "title": title,
        "url": f"https://www.kleinanzeigen.de/s-anzeige/x/{n}-227-1",
        "price": price,
        "source": "kleinanzeigen",
        "description": description,
        "shipping_cost": None,
        **extra,
    }


class TestCheckPrices:
    def test_per_game_price_replaced_by_the_whole_lot_price_the_text_states(self):
        deal = _ka(1, 7.0, description=_EXAMPLE_DESCRIPTION, shipping_note="VB")
        assert check_prices([deal]) == 1
        assert (deal["price"], deal["listed_price"], deal["price_basis"]) == (120.0, 7.0, "lot_from_text")
        assert (deal["shipping"], deal["shipping_cost"]) == ("inkl. Versand", 0.0)
        assert deal["shipping_note"] == ""  # the "VB" belonged to the 7 € piece price
        assert deal["price_note"].startswith("Listed €7.00 is the price per game ('Stück preis')")
        assert check_prices([deal]) == 0  # idempotent
        assert deal["price"] == 120.0

    def test_per_game_price_without_a_lot_price(self):
        deal = _deal(title="Xbox 360 Spiele Sammlung 20 Spiele", description="Stückpreis 3 €", price=3.0)
        check_prices([deal])
        assert (deal["price"], deal["price_basis"]) == (3.0, "per_item")
        assert deal["price_note"] == (
            "Price is per game, not for the bundle ('Stückpreis') — all 20 would cost about €60."
        )

    def test_make_an_offer_placeholder(self):
        deal = _ka(1, 1.0, description="Sammlung", shipping_note="VB")
        check_prices([deal])
        assert deal["price_basis"] == "offer"
        assert "placeholder" in deal["price_note"]

    def test_per_game_listing_no_longer_outranks_a_real_bundle(self):
        """3 € "for 20 games" looked like 0.15 €/game; it is 3 € per game."""
        title = "Xbox 360 Spiele Sammlung 20 Spiele"
        per_game = _deal(url="https://x/per-game", title=title, description="Stückpreis 3 €", price=3.0)
        real = _deal(url="https://x/real", title=title, price=40.0)
        assert select([per_game, real], _XBOX_PLAN, limit=2, now=_NOW)[0] is per_game
        check_prices([per_game, real])
        assert select([per_game, real], _XBOX_PLAN, limit=2, now=_NOW)[0] is real


class TestNotABundle:
    def test_single_game_dropped_from_bundle_searches_only(self):
        padded = _deal(title="Battlefield 1 PS4 Spiel Sammlung PS2 PS3 PS5 Konvolut Bundle Top")
        kept, removed = apply_filters([padded], plan_search(["PS4 Sammlung"]), set(), now=_NOW)
        assert kept == []
        assert removed["not_a_bundle"] == 1
        kept, _ = apply_filters([padded], plan_search(["Battlefield 1 PS4"]), set(), now=_NOW)
        assert kept == [padded]


class _Describe:
    """Stands in for KleinanzeigenScraper.fetch_description."""

    def __init__(self, texts=None, cached=None, errors=(), delay=0.0):
        self.texts, self.cached, self.errors, self.delay = texts or {}, cached or {}, list(errors), delay
        self.fetched = []

    def __call__(self, url, cached_only=False):
        if cached_only:
            return self.cached.get(url), []
        self.fetched.append(url)
        time.sleep(self.delay)
        return self.texts.get(url), self.errors


class TestCompleteDescriptions:
    def test_only_the_most_suspicious_truncated_bundles_are_fetched(self):
        deals = [
            _ka(1, 7.0),  # 16 games for 7 € — 0.44 €/game
            _ka(2, 30.0),  # 1.88 €/game
            _ka(3, 16.0),  # 1.00 €/game
            _ka(4, 120.0),  # 7.50 €/game: plausible
            _ka(5, 7.0, description="Kurze Beschreibung."),  # complete already
            _deal(price=5.0, title="Xbox 360 Spiele Sammlung 20 Spiele", description="..."),  # not Kleinanzeigen
        ]
        describe = _Describe()
        complete_descriptions(deals, _PS_PLAN, describe, budget_s=2)
        assert describe.fetched == [deals[0]["url"], deals[2]["url"]]  # cheapest per game first, at most two

    def test_full_text_reprices_the_real_listing(self):
        deal = _ka(1, 7.0, shipping_note="VB")
        describe = _Describe(texts={deal["url"]: _EXAMPLE_DESCRIPTION})
        (kept,), errors = complete_descriptions([deal], _PS_PLAN, describe, budget_s=2)
        assert errors == []
        assert kept["description"] == _EXAMPLE_DESCRIPTION
        assert (kept["price"], kept["listed_price"]) == (120.0, 7.0)

    def test_cached_text_is_used_for_any_truncated_deal_without_a_request(self):
        deal = _ka(4, 120.0)  # not suspicious — never fetched
        describe = _Describe(cached={deal["url"]: _EXAMPLE_DESCRIPTION})
        complete_descriptions([deal], _PS_PLAN, describe, budget_s=2)
        assert describe.fetched == []
        assert deal["description"] == _EXAMPLE_DESCRIPTION

    def test_single_game_revealed_by_the_full_text_is_dropped(self):
        deal = _ka(1, 5.0, title="PS4 Konvolut", description="Ich verkaufe das...")
        describe = _Describe(texts={deal["url"]: "Ich verkaufe das Spiel Battlefield 1 für die PlayStation 4."})
        kept, _ = complete_descriptions([deal], _PS_PLAN, describe, budget_s=2)
        assert kept == []

    def test_fetch_errors_are_returned(self):
        describe = _Describe(errors=["Kleinanzeigen HTTP 403"])
        _, errors = complete_descriptions([_ka(1, 7.0)], _PS_PLAN, describe, budget_s=2)
        assert errors == ["Kleinanzeigen HTTP 403"]

    def test_slow_fetch_does_not_hold_the_search(self):
        deal = _ka(1, 7.0)
        describe = _Describe(texts={deal["url"]: _EXAMPLE_DESCRIPTION}, delay=2.0)
        t0 = time.monotonic()
        (kept,), _ = complete_descriptions([deal], _PS_PLAN, describe, budget_s=0.2)
        assert time.monotonic() - t0 < 1.0
        assert kept["price"] == 7.0  # unverified this time; the scraper caches it for the next search

    def test_run_search_wires_it_in(self):
        describe = _Describe(texts={_ka(1, 7.0)["url"]: _EXAMPLE_DESCRIPTION})
        outcome = run_search(
            _PS_PLAN,
            ebay_search=_fn([]),
            ebay_is_api=True,
            auction_search=None,
            kleinanzeigen_search=_fn([_ka(1, 7.0)]),
            kleinanzeigen_describe=describe,
            skipped=set(),
            budget_s=5,
        )
        assert [d["price"] for d in outcome.selected] == [120.0]


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
