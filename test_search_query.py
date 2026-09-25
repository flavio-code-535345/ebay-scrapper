"""Tests for search/query.py — the query planner."""

import pytest

from ebay_api_client import MAX_QUERY_LENGTH
from search.query import BUNDLE_TERMS, MAX_KLEINANZEIGEN_QUERIES, detect_platform, plan_search, platforms_named

_CHIP_PHRASES = [
    "Xbox 360 Spiele Sammlung Konvolut",
    "Xbox 360 Spielesammlung Konvolut",
    "Xbox 360 Spiele Konvolut",
]


class TestBundleQueries:
    def test_one_or_group_covers_every_bundle_synonym(self):
        plan = plan_search(["Xbox 360 Spiele Sammlung"])
        assert plan.ebay_api == ("xbox 360 (sammlung,konvolut,paket,bundle,spielesammlung,lot,spielepaket)",)
        assert plan.bundle_intent is True
        assert plan.core_terms == ("xbox", "360")

    def test_quick_chip_phrases_fold_into_one_request(self):
        """The old design sent every phrasing (plus up to 8 synonym variants)
        to every source; near-identical phrasings now collapse."""
        plan = plan_search(_CHIP_PHRASES)
        assert len(plan.ebay_api) == 1
        assert len(plan.ebay_web) == 1
        assert plan.label == _CHIP_PHRASES[0]

    def test_api_queries_fit_the_browse_api_cap(self):
        long_core = "nintendo switch oled super mario party jamboree deluxe"
        plan = plan_search([f"{long_core} sammlung"])
        (api_q,) = plan.ebay_api
        assert len(api_q) <= MAX_QUERY_LENGTH
        assert api_q.startswith(long_core)
        assert "(sammlung,konvolut" in api_q  # trimmed from the end, most productive terms kept

    def test_api_queries_have_no_minus_terms(self):
        """The Browse API documents no exclusion syntax."""
        for phrases in (_CHIP_PHRASES, ["PS4 Konvolut"], ["Zelda"]):
            assert all(" -" not in q for q in plan_search(phrases).ebay_api)

    def test_web_queries_carry_exclusions(self):
        (web_q,) = plan_search(["Xbox 360 Sammlung"]).ebay_web
        assert web_q.startswith("xbox 360 (sammlung,")
        assert "-skylanders" in web_q and "-amiibo" in web_q
        # Mixed bundles containing FIFA/Kinect titles are judged by the
        # pipeline's sports filter, not excluded up front.
        assert "-fifa" not in web_q and "-kinect" not in web_q

    def test_all_bundle_terms_are_single_words(self):
        """OR alternatives must be single words; multi-word alternatives
        aren't reliably honored inside eBay's (a,b,c) groups."""
        assert all(" " not in t for t in BUNDLE_TERMS)


class TestKleinanzeigenQueries:
    def test_at_most_two_plain_variants(self):
        plan = plan_search(_CHIP_PHRASES)
        assert len(plan.kleinanzeigen) <= MAX_KLEINANZEIGEN_QUERIES
        assert all("(" not in q for q in plan.kleinanzeigen)  # no OR support there

    def test_variants_use_distinct_bundle_words(self):
        assert plan_search(["Xbox 360 Spiele Sammlung"]).kleinanzeigen == ("xbox 360 sammlung", "xbox 360 konvolut")
        assert plan_search(["PS4 Konvolut"]).kleinanzeigen == ("ps4 konvolut", "ps4 sammlung")


class TestOtherQueries:
    def test_games_intent_without_bundle_word(self):
        plan = plan_search(["Xbox 360 Spiele"])
        assert plan.ebay_api == ("xbox 360 (spiele,games,videospiele)",)
        assert plan.bundle_intent is False
        assert plan.kleinanzeigen == ("xbox 360",)

    def test_plain_title_search_passes_through(self):
        plan = plan_search(["Zelda Breath of the Wild"])
        assert plan.ebay_api == ("zelda breath of the wild",)
        assert plan.platform is None

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            plan_search(["", "   "])


class TestPlatforms:
    def test_detect_platform(self):
        assert plan_search(["PlayStation 4 Konvolut"]).platform == "Sony PlayStation 4"
        assert detect_platform("xbox360 spiele") == "Microsoft Xbox 360"
        assert detect_platform("Spielesammlung") is None

    def test_platforms_named_ignores_generic_family_names(self):
        assert platforms_named("Xbox 360 und PS4 Spiele") == {"Microsoft Xbox 360", "Sony PlayStation 4"}
        assert platforms_named("Xbox Spiele Sammlung") == set()
