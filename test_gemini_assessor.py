#!/usr/bin/env python3
"""
Unit tests for gemini_assessor.py — focusing on the deterministic
bundle bait-and-switch scam detector introduced to catch the canonical
'Spielesammlung + Stückzahl + verfügbar/verkauft > 1' pattern, and
the sports/Kinect deal detector that filters out low-resale-value listings.
"""

import json
import time
import unittest.mock as mock

import pytest

from ai_providers.base import (
    _ASSESS_TOTAL_BUDGET_S,
    _BATCH_SIZE,
    _apply_scam_override,
    _apply_sports_kinect_override,
    _build_single_game_search_query,
    _detect_bundle_individual_sale_scam,
    _detect_sports_kinect_deal,
    _extract_platform_name,
    _extract_potential_game_titles,
    _is_aggregate_placeholder,
)
from ai_providers.enrichment import _ENRICH_MAX_BUDGET_S, EnrichedDeal, Enricher, _query_jobs_for_deal
from ai_providers.gemini import _GEMINI_REQUEST_TIMEOUT, GeminiAssessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ASSESSMENT = {
    "ai_deal_rating": "Must Buy",
    "ai_confidence_score": 85,
    "ai_potential_scam": False,
    "ai_scam_warning": "",
    "ai_visual_findings": [],
    "ai_red_flags": [],
    "ai_fair_market_estimate": "~€25–35",
    "ai_verdict_summary": "Great bundle deal.",
    "ai_assessed": True,
}


def _make_assessment(**overrides):
    result = dict(_BASE_ASSESSMENT)
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# _detect_bundle_individual_sale_scam tests
# ---------------------------------------------------------------------------


class TestDetectBundleIndividualSaleScam:
    """Tests for the deterministic scam-detection helper."""

    # ------------------------------------------------------------------
    # Canonical positive cases (should detect scam)
    # ------------------------------------------------------------------

    def test_canonical_spielesammlung_4_verfuegbar_1_verkauft(self):
        """Canonical case from the bug report: Spielesammlung + 4 verfügbar, 1 verkauft."""
        deal = {
            "title": "Nintendo DS Spielesammlung 20 Spiele",
            "seller_count": "4 verfügbar, 1 verkauft",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None
        assert "BAIT-AND-SWITCH" in result
        assert "Spielesammlung" in result

    def test_sammlung_with_multiple_verfuegbar(self):
        """Sammlung keyword + multiple available."""
        deal = {
            "title": "PS4 Spiele Sammlung - 10 Spiele",
            "seller_count": "5 verfügbar",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None

    def test_lot_keyword_with_verkauft_gt_1(self):
        """'Lot' keyword + sold > 1."""
        deal = {
            "title": "Game Boy Lot 15 Spiele",
            "seller_count": "3 verkauft",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None

    def test_bundle_keyword_mixed_case(self):
        """Bundle keyword is case-insensitive."""
        deal = {
            "title": "SNES BUNDLE 8 Games",
            "seller_count": "2 verfügbar, 2 verkauft",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None

    def test_konvolut_keyword(self):
        """Konvolut keyword triggers detection."""
        deal = {
            "title": "Mega Drive Konvolut 12 Spiele",
            "seller_count": "6 verfügbar",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None

    def test_paket_keyword(self):
        """Paket keyword triggers detection."""
        deal = {
            "title": "PS2 Spielepaket Rarität",
            "seller_count": "10 verfügbar, 5 verkauft",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None

    def test_large_quantity_detected(self):
        """High availability count (e.g. 50 verfügbar) is flagged."""
        deal = {
            "title": "Switch Spielesammlung günstig",
            "seller_count": "50 verfügbar",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None

    def test_collection_keyword_english(self):
        """English 'collection' keyword is also detected."""
        deal = {
            "title": "NES Game Collection 20 cartridges",
            "seller_count": "4 verfügbar, 2 verkauft",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is not None

    # ------------------------------------------------------------------
    # True-negative cases (should NOT detect scam)
    # ------------------------------------------------------------------

    def test_genuine_bundle_quantity_1(self):
        """A listing with exactly 1 available and 0 sold is NOT flagged."""
        deal = {
            "title": "Nintendo DS Spielesammlung 20 Spiele",
            "seller_count": "1 verfügbar",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is None

    def test_genuine_bundle_no_seller_count(self):
        """No seller_count data — cannot determine, so not flagged."""
        deal = {
            "title": "PS3 Spielesammlung 15 Spiele",
            "seller_count": "",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is None

    def test_single_game_listing(self):
        """A plain single-game listing is not flagged (no bundle keyword)."""
        deal = {
            "title": "Zelda Breath of the Wild Switch",
            "seller_count": "4 verfügbar, 2 verkauft",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is None

    def test_bundle_keyword_but_quantity_zero(self):
        """Bundle keyword present but seller_count has no numbers — not flagged."""
        deal = {
            "title": "PC Spielesammlung groß",
            "seller_count": "verfügbar",  # no numeric value
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is None

    def test_missing_title(self):
        """Empty title — cannot detect, not flagged."""
        deal = {
            "title": "",
            "seller_count": "5 verfügbar",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is None

    def test_empty_deal(self):
        """Completely empty deal dict — not flagged."""
        result = _detect_bundle_individual_sale_scam({})
        assert result is None

    def test_bundle_keyword_only_1_sold(self):
        """Bundle keyword + exactly 1 sold (and nothing else) — not flagged."""
        deal = {
            "title": "GameCube Spielesammlung 8 Spiele",
            "seller_count": "1 verkauft",
        }
        result = _detect_bundle_individual_sale_scam(deal)
        assert result is None


# ---------------------------------------------------------------------------
# _apply_scam_override tests
# ---------------------------------------------------------------------------


class TestApplyScamOverride:
    """Tests for the assessment-override function."""

    def test_override_forces_avoid_and_scam_flag(self):
        """Canonical scam pattern overrides Must Buy to Avoid with scam=True."""
        deal = {
            "title": "Nintendo DS Spielesammlung 20 Spiele",
            "seller_count": "4 verfügbar, 1 verkauft",
        }
        assessment = _make_assessment(ai_deal_rating="Must Buy", ai_potential_scam=False)
        result = _apply_scam_override(deal, assessment)

        assert result["ai_deal_rating"] == "Avoid"
        assert result["ai_potential_scam"] is True
        assert "BAIT-AND-SWITCH" in result["ai_scam_warning"]
        assert "SCAM RISK" in result["ai_verdict_summary"]

    def test_override_prepends_to_existing_summary(self):
        """Existing verdict_summary is preserved after the scam prefix."""
        deal = {
            "title": "PS4 Spielesammlung 15 Spiele",
            "seller_count": "3 verfügbar",
        }
        assessment = _make_assessment(
            ai_deal_rating="Fair",
            ai_verdict_summary="Good condition, potential profit.",
        )
        result = _apply_scam_override(deal, assessment)

        assert result["ai_verdict_summary"].startswith("⚠️ **SCAM RISK")
        assert "Good condition, potential profit." in result["ai_verdict_summary"]

    def test_override_appends_to_existing_warning(self):
        """Existing scam_warning from Gemini is preserved alongside the new warning."""
        deal = {
            "title": "GBA Bundle 10 Spiele",
            "seller_count": "8 verfügbar, 3 verkauft",
        }
        assessment = _make_assessment(
            ai_deal_rating="Fair",
            ai_potential_scam=True,
            ai_scam_warning="Gemini flagged: some pick-one wording.",
        )
        result = _apply_scam_override(deal, assessment)

        assert "BAIT-AND-SWITCH" in result["ai_scam_warning"]
        assert "Gemini flagged: some pick-one wording." in result["ai_scam_warning"]

    def test_no_override_for_single_game_listing(self):
        """Non-bundle listing with high seller_count is not touched."""
        deal = {
            "title": "Mario Kart 8 Nintendo Switch",
            "seller_count": "10 verfügbar",
        }
        assessment = _make_assessment(ai_deal_rating="Must Buy")
        result = _apply_scam_override(deal, assessment)

        assert result["ai_deal_rating"] == "Must Buy"
        assert result["ai_potential_scam"] is False

    def test_no_override_when_genuine_bundle(self):
        """Genuine bundle (quantity 1) is not overridden."""
        deal = {
            "title": "Nintendo DS Spielesammlung 20 Spiele",
            "seller_count": "1 verfügbar",
        }
        assessment = _make_assessment(ai_deal_rating="Must Buy")
        result = _apply_scam_override(deal, assessment)

        assert result["ai_deal_rating"] == "Must Buy"
        assert result["ai_potential_scam"] is False

    def test_override_sets_verdict_when_summary_missing(self):
        """If ai_verdict_summary is empty the scam prefix becomes the full summary."""
        deal = {
            "title": "Switch Spielesammlung 30 Spiele",
            "seller_count": "20 verfügbar, 10 verkauft",
        }
        assessment = _make_assessment(ai_verdict_summary="")
        result = _apply_scam_override(deal, assessment)

        assert result["ai_verdict_summary"].startswith("⚠️ **SCAM RISK")
        verdict_lower = result["ai_verdict_summary"].lower()
        assert "collection" in verdict_lower or "scam" in verdict_lower

    def test_override_returns_same_dict(self):
        """_apply_scam_override mutates and returns the same dict object."""
        deal = {
            "title": "Xbox Spielesammlung 5 Spiele",
            "seller_count": "3 verfügbar",
        }
        assessment = _make_assessment()
        result = _apply_scam_override(deal, assessment)
        assert result is assessment


# ---------------------------------------------------------------------------
# _detect_sports_kinect_deal tests
# ---------------------------------------------------------------------------


class TestDetectSportsKinectDeal:
    """Tests for the deterministic sports/Kinect content detector."""

    # ------------------------------------------------------------------
    # Positive cases (should detect sports/Kinect content)
    # ------------------------------------------------------------------

    def test_detects_kinect_in_title(self):
        """'Kinect' keyword triggers detection."""
        deal = {"title": "Xbox 360 Kinect Sensor + 3 Spiele Bundle"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None
        assert "SPORTS/KINECT" in result

    def test_detects_fifa_in_title(self):
        """'FIFA' keyword triggers detection."""
        deal = {"title": "PS4 Spielesammlung FIFA 22 FIFA 21 FIFA 20 5 Spiele"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None
        assert "SPORTS/KINECT CONTENT DETECTED" in result
        assert "FIFA" in result

    def test_detects_topspin_in_title(self):
        """'TopSpin' keyword triggers detection."""
        deal = {"title": "Xbox 360 Bundle TopSpin 4 + Forza 3 + FIFA 18"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None

    def test_detects_forza_in_title(self):
        """'Forza' keyword triggers detection."""
        deal = {"title": "Xbox 360 Lot Forza Motorsport 4 + FIFA 14"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None

    def test_detects_nba_in_title(self):
        """'NBA 2K' keyword triggers detection."""
        deal = {"title": "PS4 NBA 2K22 + FIFA 22 Spielesammlung"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None

    def test_detects_pes_in_title(self):
        """'PES' keyword triggers detection."""
        deal = {"title": "PS2 Spielesammlung PES 6 PES 5 Konvolut"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None

    def test_detects_just_dance_in_title(self):
        """'Just Dance' keyword triggers detection."""
        deal = {"title": "Wii Just Dance 2019 + Just Dance 2020 Bundle"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None

    def test_detects_kinect_case_insensitive(self):
        """Detection is case-insensitive."""
        deal = {"title": "XBOX 360 KINECT ADVENTURES BUNDLE"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None

    def test_detects_wii_sports_in_title(self):
        """'Wii Sports' keyword triggers detection."""
        deal = {"title": "Wii Sports + Wii Sports Resort Bundle"}
        result = _detect_sports_kinect_deal(deal)
        assert result is not None

    # ------------------------------------------------------------------
    # Negative cases (should NOT detect sports/Kinect content)
    # ------------------------------------------------------------------

    def test_no_detection_for_halo_bundle(self):
        """A non-sports bundle like Halo is NOT flagged."""
        deal = {"title": "Xbox 360 Spielesammlung Halo 3 Halo 4 Gears of War"}
        result = _detect_sports_kinect_deal(deal)
        assert result is None

    def test_no_detection_for_zelda_bundle(self):
        """Nintendo first-party bundle is not flagged."""
        deal = {"title": "Nintendo Switch Bundle Zelda Breath of the Wild + Mario Kart"}
        result = _detect_sports_kinect_deal(deal)
        assert result is None

    def test_no_detection_for_empty_title(self):
        """Empty title is not flagged."""
        deal = {"title": ""}
        result = _detect_sports_kinect_deal(deal)
        assert result is None

    def test_no_detection_for_missing_title(self):
        """Missing title key is not flagged."""
        deal = {}
        result = _detect_sports_kinect_deal(deal)
        assert result is None

    def test_no_detection_for_cod_bundle(self):
        """Call of Duty bundle (non-sports) is not flagged."""
        deal = {"title": "PS4 Bundle Call of Duty Black Ops 3 + GTA V 10 Spiele"}
        result = _detect_sports_kinect_deal(deal)
        assert result is None

    def test_no_detection_for_rpg_bundle(self):
        """RPG-heavy bundle is not flagged."""
        deal = {"title": "PS3 Spielesammlung Final Fantasy Dark Souls Skyrim"}
        result = _detect_sports_kinect_deal(deal)
        assert result is None


# ---------------------------------------------------------------------------
# _apply_sports_kinect_override tests
# ---------------------------------------------------------------------------


class TestApplySportsKinectOverride:
    """Tests for the sports/Kinect assessment-override function."""

    def test_override_forces_avoid_for_kinect_deal(self):
        """Kinect deal overrides Must Buy to Avoid."""
        deal = {"title": "Xbox 360 Kinect Sensor + Adventures Bundle"}
        assessment = _make_assessment(ai_deal_rating="Must Buy")
        result = _apply_sports_kinect_override(deal, assessment)
        assert result["ai_deal_rating"] == "Avoid"

    def test_override_forces_avoid_for_fifa_deal(self):
        """FIFA bundle overrides Fair to Avoid."""
        deal = {"title": "PS4 Spielesammlung FIFA 22 FIFA 21 FIFA 20"}
        assessment = _make_assessment(ai_deal_rating="Fair")
        result = _apply_sports_kinect_override(deal, assessment)
        assert result["ai_deal_rating"] == "Avoid"

    def test_override_prepends_to_verdict_summary(self):
        """Existing verdict summary is preserved after the sports/Kinect prefix."""
        deal = {"title": "Xbox 360 Bundle Forza Motorsport + FIFA"}
        assessment = _make_assessment(
            ai_deal_rating="Fair",
            ai_verdict_summary="Cheap lot.",
        )
        result = _apply_sports_kinect_override(deal, assessment)
        assert result["ai_verdict_summary"].startswith("⛔ **SPORTS/KINECT")
        assert "Cheap lot." in result["ai_verdict_summary"]

    def test_override_adds_red_flag(self):
        """Sports/Kinect flag is added to red_flags list."""
        deal = {"title": "Wii Just Dance 2020 + Wii Sports Bundle"}
        assessment = _make_assessment(ai_red_flags=[])
        result = _apply_sports_kinect_override(deal, assessment)
        assert any("Sports/Kinect" in f for f in result["ai_red_flags"])

    def test_override_does_not_duplicate_red_flag(self):
        """Running override twice does not add duplicate red flags."""
        deal = {"title": "Xbox 360 Kinect Bundle"}
        assessment = _make_assessment(ai_red_flags=[])
        _apply_sports_kinect_override(deal, assessment)
        _apply_sports_kinect_override(deal, assessment)
        sports_flags = [f for f in assessment["ai_red_flags"] if "Sports/Kinect" in f]
        assert len(sports_flags) == 1

    def test_no_override_for_non_sports_deal(self):
        """Non-sports deal is not overridden."""
        deal = {"title": "Xbox 360 Bundle Halo 3 + Gears of War + Mass Effect"}
        assessment = _make_assessment(ai_deal_rating="Must Buy")
        result = _apply_sports_kinect_override(deal, assessment)
        assert result["ai_deal_rating"] == "Must Buy"

    def test_override_returns_same_dict(self):
        """_apply_sports_kinect_override mutates and returns the same dict."""
        deal = {"title": "PS3 Bundle FIFA 22 + TopSpin 4"}
        assessment = _make_assessment()
        result = _apply_sports_kinect_override(deal, assessment)
        assert result is assessment

    def test_override_with_empty_summary(self):
        """If verdict summary is empty, sports prefix becomes the full summary."""
        deal = {"title": "Xbox Kinect Sports Bundle"}
        assessment = _make_assessment(ai_verdict_summary="")
        result = _apply_sports_kinect_override(deal, assessment)
        assert result["ai_verdict_summary"].startswith("⛔ **SPORTS/KINECT")


# ---------------------------------------------------------------------------
# _extract_potential_game_titles tests
# ---------------------------------------------------------------------------


class TestExtractPotentialGameTitles:
    """Tests for the bundle game-title extractor."""

    def test_comma_separated_titles(self):
        """Comma-separated titles in a bundle listing are extracted."""
        title = "PS4 Bundle: God of War, Spider-Man, Horizon Zero Dawn"
        result = _extract_potential_game_titles(title)
        assert len(result) >= 2
        assert any("God of War" in t for t in result)

    def test_plus_separated_titles(self):
        """Plus-sign separated titles are extracted."""
        title = "Switch Bundle Zelda + Mario Odyssey + Kirby"
        result = _extract_potential_game_titles(title)
        assert len(result) >= 2
        assert any("Zelda" in t for t in result)

    def test_generic_bundle_no_titles(self):
        """Generic 'N Spiele' bundle with no individual titles returns empty."""
        title = "10 PS4 Spiele Sammlung Lot"
        result = _extract_potential_game_titles(title)
        # May return empty or only generic words stripped — count should be low
        # (no real game names can be extracted)
        for t in result:
            # Should not contain generic platform words alone
            assert t not in {"PS4", "Sammlung", "Lot", "Spiele"}

    def test_empty_title_returns_empty(self):
        """Empty title returns empty list."""
        assert _extract_potential_game_titles("") == []

    def test_respects_max_games_limit(self):
        """Never returns more than _MAX_GAMES_PER_BUNDLE titles."""
        from ai_providers.base import _MAX_GAMES_PER_BUNDLE

        many = ", ".join([f"Game {i}" for i in range(20)])
        title = f"Bundle: {many}"
        result = _extract_potential_game_titles(title)
        assert len(result) <= _MAX_GAMES_PER_BUNDLE

    def test_short_tokens_filtered(self):
        """Tokens shorter than 3 characters are excluded."""
        title = "PS4 Bundle: A + B + God of War"
        result = _extract_potential_game_titles(title)
        for t in result:
            assert len(t) >= 3

    def test_numeric_only_tokens_filtered(self):
        """Pure numeric tokens are excluded."""
        title = "Bundle: 22 + FIFA 22 + 21"
        result = _extract_potential_game_titles(title)
        for t in result:
            assert not t.isdigit()


# ---------------------------------------------------------------------------
# _extract_platform_name tests
# ---------------------------------------------------------------------------


class TestExtractPlatformName:
    def test_xbox_360(self):
        assert _extract_platform_name("10 Xbox 360 Spiele Bundle") == "Microsoft Xbox 360"

    def test_xbox_360_no_space(self):
        assert _extract_platform_name("Xbox360 Spielesammlung") == "Microsoft Xbox 360"

    def test_ps4(self):
        assert _extract_platform_name("PS4 Spielesammlung 5 Spiele") == "Sony PlayStation 4"

    def test_playstation_4_full(self):
        assert _extract_platform_name("PlayStation 4 Bundle") == "Sony PlayStation 4"

    def test_ps3(self):
        assert _extract_platform_name("PS3 Lot 8 Spiele") == "Sony PlayStation 3"

    def test_wii(self):
        assert _extract_platform_name("Nintendo Wii Spiele Lot") == "Nintendo Wii"

    def test_nintendo_switch(self):
        assert _extract_platform_name("Nintendo Switch Bundle 5 Games") == "Nintendo Switch"

    def test_xbox_one(self):
        assert _extract_platform_name("Xbox One Bundle COD + FIFA") == "Microsoft Xbox One"

    def test_no_platform(self):
        assert _extract_platform_name("5 Spiele Bundle Lot") == ""

    def test_case_insensitive(self):
        assert _extract_platform_name("XBOX 360 BUNDLE") == "Microsoft Xbox 360"

    def test_more_specific_before_generic(self):
        """Xbox 360 must be matched before bare Xbox."""
        assert _extract_platform_name("Xbox 360 Sammlung") == "Microsoft Xbox 360"


# ---------------------------------------------------------------------------
# _build_single_game_search_query tests
# ---------------------------------------------------------------------------


class TestBuildSingleGameSearchQuery:
    """Tests for the single-game eBay search query builder."""

    def test_appends_platform_in_parentheses(self):
        """Platform is appended in the required '(PLATFORM)' format."""
        query = _build_single_game_search_query("Halo 3 Xbox 360 gebraucht")
        assert query.endswith("(Microsoft Xbox 360)")

    def test_strips_condition_words(self):
        """Common condition words (gebraucht, neu, OVP, etc.) are stripped."""
        query = _build_single_game_search_query("Batman Arkham Knight PS4 gebraucht OVP")
        assert "gebraucht" not in query.lower()
        assert "ovp" not in query.lower()

    def test_strips_platform_from_body(self):
        """Platform keywords are removed from the game-name part of the query."""
        query = _build_single_game_search_query("Halo 3 Xbox 360 gebraucht")
        # Platform name should appear exactly once, inside the parentheses.
        assert query.count("Xbox 360") == 1
        assert query.count("Microsoft Xbox 360") == 1

    def test_ps4_platform(self):
        """PlayStation 4 is recognised and formatted correctly."""
        query = _build_single_game_search_query("God of War PS4")
        assert "(Sony PlayStation 4)" in query

    def test_switch_platform(self):
        """Nintendo Switch is recognised and formatted correctly."""
        query = _build_single_game_search_query("Zelda Breath of the Wild Nintendo Switch")
        assert "(Nintendo Switch)" in query

    def test_no_platform_returns_cleaned_title(self):
        """Without a detected platform the cleaned title is returned as-is."""
        query = _build_single_game_search_query("Cyberpunk 2077")
        assert "(" not in query
        assert "Cyberpunk" in query

    def test_fallback_on_over_cleaning(self):
        """If cleaning removes too much the original title is used as fallback."""
        # A title that is entirely composed of condition/platform words
        query = _build_single_game_search_query("Xbox 360 gebraucht neu")
        # Result should still be non-empty
        assert len(query) >= 3

    def test_empty_title_returns_empty(self):
        """Empty input returns empty string."""
        assert _build_single_game_search_query("") == ""

    def test_xbox_one_platform(self):
        """Xbox One is correctly detected and formatted."""
        query = _build_single_game_search_query("Forza Horizon 4 Xbox One")
        assert "(Microsoft Xbox One)" in query

    def test_ps3_platform(self):
        """PlayStation 3 is correctly detected."""
        query = _build_single_game_search_query("Dark Souls PS3 sehr gut")
        assert "(Sony PlayStation 3)" in query


# ---------------------------------------------------------------------------
# Tests: GOOD / MUST HAVE rated bundles are not blocked by overrides
# ---------------------------------------------------------------------------


class TestGoodMustHaveBundlesNotBlocked:
    """Verify that legitimate non-sports, non-scam bundles can receive
    GOOD or MUST HAVE ratings — i.e. the deterministic overrides do NOT
    fire for them and must not convert those ratings to 'Avoid'."""

    # ── Must Have scenarios ───────────────────────────────────────────

    def test_must_have_stays_must_have_for_genuine_bundle(self):
        """A genuine bundle (qty=1, no sports) keeps a 'Must Have' AI rating."""
        deal = {
            "title": "Xbox 360 Bundle: Halo 3, Gears of War, Mass Effect",
            "seller_count": "1 verfügbar",
        }
        assessment = _make_assessment(ai_deal_rating="Must Have")
        result = _apply_scam_override(deal, assessment)
        result = _apply_sports_kinect_override(deal, result)
        assert result["ai_deal_rating"] == "Must Have"
        assert result["ai_potential_scam"] is False

    def test_must_have_stays_must_have_ps3_rpg_bundle(self):
        """PS3 RPG bundle at low price: both overrides leave 'Must Have' intact."""
        deal = {
            "title": "PS3 Spielesammlung: Final Fantasy XIII, Dark Souls, Skyrim",
            "seller_count": "1 verfügbar",
        }
        assessment = _make_assessment(ai_deal_rating="Must Have")
        result = _apply_scam_override(deal, assessment)
        result = _apply_sports_kinect_override(deal, result)
        assert result["ai_deal_rating"] == "Must Have"

    def test_must_have_stays_for_single_game_no_bundle_keyword(self):
        """Single-game listing (no bundle keyword) is never scam-flagged."""
        deal = {
            "title": "Red Dead Redemption 2 PS4",
            "seller_count": "5 verfügbar",
        }
        assessment = _make_assessment(ai_deal_rating="Must Have")
        result = _apply_scam_override(deal, assessment)
        result = _apply_sports_kinect_override(deal, result)
        assert result["ai_deal_rating"] == "Must Have"
        assert result["ai_potential_scam"] is False

    # ── Good scenarios ───────────────────────────────────────────────

    def test_good_stays_good_for_genuine_bundle(self):
        """A genuine adventure-game bundle keeps a 'Good' AI rating."""
        deal = {
            "title": "Switch Bundle: Mario Odyssey, Zelda, Kirby",
            "seller_count": "1 verfügbar",
        }
        assessment = _make_assessment(ai_deal_rating="Good")
        result = _apply_scam_override(deal, assessment)
        result = _apply_sports_kinect_override(deal, result)
        assert result["ai_deal_rating"] == "Good"
        assert result["ai_potential_scam"] is False

    def test_good_stays_good_for_single_rpg_game(self):
        """Non-sports single-game listing keeps a 'Good' rating."""
        deal = {
            "title": "God of War Ragnarok PS5 wie neu",
            "seller_count": "3 verfügbar",
        }
        assessment = _make_assessment(ai_deal_rating="Good")
        result = _apply_scam_override(deal, assessment)
        result = _apply_sports_kinect_override(deal, result)
        assert result["ai_deal_rating"] == "Good"

    # ── Sports/Kinect still blocked ──────────────────────────────────

    def test_sports_bundle_is_avoided_even_if_ai_said_good(self):
        """Sports/Kinect override correctly demotes a would-be 'Good' rating."""
        deal = {"title": "Xbox 360 Bundle FIFA 22 + Forza 4 + Kinect Adventures"}
        assessment = _make_assessment(ai_deal_rating="Good")
        result = _apply_sports_kinect_override(deal, assessment)
        assert result["ai_deal_rating"] == "Avoid"

    def test_scam_bundle_is_avoided_even_if_ai_said_must_have(self):
        """Scam override correctly demotes a would-be 'Must Have' rating."""
        deal = {
            "title": "Xbox 360 Spielesammlung 20 Spiele",
            "seller_count": "10 verfügbar, 5 verkauft",
        }
        assessment = _make_assessment(ai_deal_rating="Must Have")
        result = _apply_scam_override(deal, assessment)
        assert result["ai_deal_rating"] == "Avoid"
        assert result["ai_potential_scam"] is True


# ---------------------------------------------------------------------------
# _parse_batch_response correctly preserves GOOD / MUST HAVE ratings
# ---------------------------------------------------------------------------


class TestParseBatchResponseGoodMustHave:
    """Verify _parse_batch_response faithfully forwards 'Good' and
    'Must Have' deal_rating values from the AI JSON response."""

    def _parse(self, payload):
        return GeminiAssessor._parse_batch_response(json.dumps(payload), len(payload))

    def test_must_have_rating_preserved(self):
        payload = [
            {
                "deal_rating": "Must Have",
                "confidence_score": 95,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€40",
                "itemized_resale_estimates": [
                    {"game": "Halo 3", "price_eur": 12.0, "price_source": "ebay_sold"},
                    {"game": "Gears of War", "price_eur": 10.0, "price_source": "ebay_sold"},
                ],
                "estimated_total_cost": 8.99,
                "estimated_gross_profit": 13.01,
                "verdict_summary": "Excellent profit potential.",
            }
        ]
        result = self._parse(payload)
        assert len(result) == 1
        assert result[0]["ai_deal_rating"] == "Must Have"
        assert result[0]["ai_assessed"] is True
        assert result[0]["ai_estimated_gross_profit"] == pytest.approx(13.01)

    def test_good_rating_preserved(self):
        payload = [
            {
                "deal_rating": "Good",
                "confidence_score": 80,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€20",
                "itemized_resale_estimates": [
                    {
                        "game": "Batman Arkham Knight",
                        "price_eur": 18.0,
                        "price_source": "ebay_active",
                    },
                ],
                "estimated_total_cost": 10.99,
                "estimated_gross_profit": 7.01,
                "verdict_summary": "Good profit potential.",
            }
        ]
        result = self._parse(payload)
        assert len(result) == 1
        assert result[0]["ai_deal_rating"] == "Good"
        assert result[0]["ai_assessed"] is True
        assert result[0]["ai_estimated_gross_profit"] == pytest.approx(7.01)

    def test_mixed_batch_ratings_preserved(self):
        """Batch with Must Have, Good, Okay, Avoid all preserved correctly."""
        payload = [
            {
                "deal_rating": "Must Have",
                "confidence_score": 90,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€50",
                "itemized_resale_estimates": [
                    {"game": "Halo 3 Xbox 360", "price_eur": 50.0, "price_source": "ebay_sold"}
                ],
                "estimated_total_cost": 10.0,
                "estimated_gross_profit": 40.0,
                "verdict_summary": "Amazing.",
            },
            {
                "deal_rating": "Good",
                "confidence_score": 75,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€25",
                "itemized_resale_estimates": [
                    {
                        "game": "Batman Arkham Knight PS4",
                        "price_eur": 25.0,
                        "price_source": "ebay_active",
                    }
                ],
                "estimated_total_cost": 15.0,
                "estimated_gross_profit": 10.0,
                "verdict_summary": "Good.",
            },
            {
                "deal_rating": "Okay",
                "confidence_score": 60,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€12",
                "itemized_resale_estimates": [
                    {"game": "Minecraft Xbox 360", "price_eur": 12.0, "price_source": "ai_estimate"}
                ],
                "estimated_total_cost": 10.0,
                "estimated_gross_profit": 2.0,
                "verdict_summary": "Decent.",
            },
            {
                "deal_rating": "Avoid",
                "confidence_score": 85,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€8",
                "itemized_resale_estimates": [{"game": "FIFA 22 PS4", "price_eur": 3.0, "price_source": "ebay_sold"}],
                "estimated_total_cost": 10.0,
                "estimated_gross_profit": -7.0,
                "verdict_summary": "Loss.",
            },
        ]
        results = self._parse(payload)
        assert len(results) == 4
        ratings = [r["ai_deal_rating"] for r in results]
        assert ratings == ["Must Have", "Good", "Okay", "Avoid"]
        for r in results:
            assert r["ai_assessed"] is True

    def test_itemized_resale_estimates_for_single_game(self):
        """Single-game itemized_resale_estimates with one entry is stored correctly."""
        payload = [
            {
                "deal_rating": "Good",
                "confidence_score": 78,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€15",
                "itemized_resale_estimates": [
                    {"game": "Halo 3 Xbox 360", "price_eur": 15.0, "price_source": "ebay_sold"},
                ],
                "estimated_total_cost": 8.99,
                "estimated_gross_profit": 6.01,
                "verdict_summary": "Good single-game flip.",
            }
        ]
        result = self._parse(payload)
        assert result[0]["ai_deal_rating"] == "Good"
        assert len(result[0]["ai_itemized_resale_estimates"]) == 1
        assert result[0]["ai_itemized_resale_estimates"][0]["price_eur"] == 15.0


# ---------------------------------------------------------------------------
# eBay price cache helpers
# ---------------------------------------------------------------------------


class TestEbayPriceCache:
    """Tests for Enricher's in-memory eBay price cache (moved out of the
    assessor itself as part of splitting network I/O into its own phase)."""

    def test_cache_miss_returns_none(self):
        """A fresh Enricher returns None for any query."""
        e = Enricher()
        assert e._cached_price("Halo 3 (Microsoft Xbox 360)") is None

    def test_store_and_retrieve(self):
        """Stored price is returned on the next lookup."""
        e = Enricher()
        e._store_price("Halo 3 (Microsoft Xbox 360)", 12.50, "sold_listings")
        result = e._cached_price("Halo 3 (Microsoft Xbox 360)")
        assert result is not None
        price, source = result
        assert price == pytest.approx(12.50)
        assert source == "sold_listings"

    def test_store_none_price_is_cached(self):
        """A None price (no eBay result) is also cached to avoid retrying."""
        e = Enricher()
        e._store_price("Unknown Game (Nintendo Switch)", None, "no_result")
        result = e._cached_price("Unknown Game (Nintendo Switch)")
        assert result is not None
        price, source = result
        assert price is None
        assert source == "no_result"

    def test_cache_entry_expires_after_ttl(self):
        """Cache entries are evicted when their TTL has elapsed."""
        import time as _time

        e = Enricher()
        e._store_price("God of War (Sony PlayStation 4)", 20.0, "active_listings")
        # Manually expire the entry by backdating its timestamp.
        query = "God of War (Sony PlayStation 4)"
        price, source, expire_at = e._price_cache[query]
        e._price_cache[query] = (price, source, _time.monotonic() - 1.0)
        assert e._cached_price(query) is None
        # Evicted entry should be removed from the dict.
        assert query not in e._price_cache

    def test_separate_queries_do_not_collide(self):
        """Different queries are stored and retrieved independently."""
        e = Enricher()
        e._store_price("Halo 3 (Microsoft Xbox 360)", 12.0, "sold_listings")
        e._store_price("Zelda (Nintendo Switch)", 35.0, "active_listings")
        r1 = e._cached_price("Halo 3 (Microsoft Xbox 360)")
        r2 = e._cached_price("Zelda (Nintendo Switch)")
        assert r1 is not None and r1[0] == pytest.approx(12.0)
        assert r2 is not None and r2[0] == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# _query_jobs_for_deal — pure computation, no I/O
# ---------------------------------------------------------------------------


class TestQueryJobsForDeal:
    """Tests for the pure per-deal eBay-query-job builder."""

    def test_single_game_xbox360(self):
        """Single-game Xbox 360 listing produces a 'GAME (Microsoft Xbox 360)' query."""
        is_bundle, jobs = _query_jobs_for_deal({"title": "Halo 3 Xbox 360 gebraucht"})
        assert is_bundle is False
        assert len(jobs) == 1
        _label, query = jobs[0]
        assert query.endswith("(Microsoft Xbox 360)")
        assert "Halo" in query

    def test_single_game_ps4(self):
        """Single-game PS4 listing produces a 'GAME (Sony PlayStation 4)' query."""
        is_bundle, jobs = _query_jobs_for_deal({"title": "God of War PS4"})
        assert is_bundle is False
        assert len(jobs) == 1
        assert "(Sony PlayStation 4)" in jobs[0][1]

    def test_bundle_produces_multiple_queries(self):
        """Bundle listing produces one (game, query) job per extracted title."""
        is_bundle, jobs = _query_jobs_for_deal({"title": "Xbox 360 Bundle: Halo 3, Gears of War, Mass Effect"})
        assert is_bundle is True
        assert len(jobs) >= 2
        for _label, q in jobs:
            assert "(Microsoft Xbox 360)" in q

    def test_bundle_no_titles_falls_back_to_single_listing(self):
        """A bundle-keyword title with no extractable game names falls back to
        treating the whole title as one single-listing query — matching the
        original behavior (empty bundle_prices used to trigger a single-
        listing price fetch on the whole title instead)."""
        assert _extract_potential_game_titles("Xbox 360 Konvolut") == []  # sanity-check the premise
        is_bundle, jobs = _query_jobs_for_deal({"title": "Xbox 360 Konvolut"})
        assert is_bundle is False
        for _label, q in jobs:
            assert len(q.strip()) >= 3

    def test_empty_title_returns_empty(self):
        _is_bundle, jobs = _query_jobs_for_deal({"title": ""})
        assert jobs == []

    def test_missing_title_returns_empty(self):
        _is_bundle, jobs = _query_jobs_for_deal({})
        assert jobs == []


# ---------------------------------------------------------------------------
# Enricher.enrich_deals — price lookups
# ---------------------------------------------------------------------------


class TestEnricherPriceLookups:
    """Tests for Enricher.enrich_deals's eBay price-lookup half."""

    def _make_enricher_with_mock_client(self, price_map=None):
        """Return an Enricher with a mock eBay client.

        *price_map* maps query substrings to (price, source) tuples so tests
        can simulate different eBay API outcomes.
        """
        price_map = price_map or {}

        def _mock_get_median(query, max_results=10):
            for key, (price, source) in price_map.items():
                if key in query:
                    return price, source, []
            return None, "no_result", []

        mock_client = mock.MagicMock()
        mock_client.get_lowest_market_price.side_effect = _mock_get_median
        e = Enricher()
        e.ebay_client = mock_client
        return e

    def test_populates_single_price_for_single_game_deal(self):
        e = self._make_enricher_with_mock_client({"Halo": (12.0, "sold_listings")})
        results = e.enrich_deals([{"title": "Halo 3 Xbox 360"}], deadline=time.monotonic() + 5)
        assert results[0].single_price == pytest.approx(12.0)

    def test_populates_bundle_prices_for_bundle_deal(self):
        e = self._make_enricher_with_mock_client(
            {
                "Halo": (12.0, "sold_listings"),
                "Gears": (9.0, "sold_listings"),
            }
        )
        results = e.enrich_deals([{"title": "Xbox 360 Bundle: Halo 3, Gears of War"}], deadline=time.monotonic() + 5)
        assert len(results[0].bundle_prices) >= 1
        assert any(g["price_eur"] is not None for g in results[0].bundle_prices)

    def test_deduplicates_queries_across_deals(self):
        """Same game appearing in multiple deals triggers only one eBay call."""
        call_count = {"n": 0}

        def _mock_get_median(query, max_results=10):
            call_count["n"] += 1
            return 10.0, "sold_listings", []

        e = Enricher()
        mock_client = mock.MagicMock()
        mock_client.get_lowest_market_price.side_effect = _mock_get_median
        e.ebay_client = mock_client

        # Two deals with the same title → same query → should deduplicate.
        deals = [{"title": "Halo 3 Xbox 360"}, {"title": "Halo 3 Xbox 360"}]
        e.enrich_deals(deals, deadline=time.monotonic() + 5)
        assert call_count["n"] == 1, "Duplicate queries should only be fetched once"

    def test_uses_cache_on_second_call(self):
        """A second enrich_deals call for the same deals makes no new eBay API calls."""
        call_count = {"n": 0}

        def _mock_get_median(query, max_results=10):
            call_count["n"] += 1
            return 10.0, "sold_listings", []

        e = Enricher()
        mock_client = mock.MagicMock()
        mock_client.get_lowest_market_price.side_effect = _mock_get_median
        e.ebay_client = mock_client

        deals = [{"title": "Halo 3 Xbox 360"}]
        e.enrich_deals(deals, deadline=time.monotonic() + 5)
        first_count = call_count["n"]
        e.enrich_deals(deals, deadline=time.monotonic() + 5)
        assert call_count["n"] == first_count, "Second call should hit cache, not call eBay again"

    def test_no_ebay_client_returns_no_prices(self):
        """When no eBay client is registered, enrichment still succeeds — just
        with no price data — rather than raising."""
        e = Enricher()
        results = e.enrich_deals([{"title": "Halo 3 Xbox 360"}], deadline=time.monotonic() + 5)
        assert results[0].single_price is None
        assert results[0].bundle_prices == []

    def test_failed_ebay_call_does_not_raise(self):
        """A failing eBay API call is silently absorbed; the cache is not poisoned."""
        e = Enricher()
        mock_client = mock.MagicMock()
        mock_client.get_lowest_market_price.side_effect = RuntimeError("connection refused")
        e.ebay_client = mock_client
        e.enrich_deals([{"title": "Halo 3 Xbox 360"}], deadline=time.monotonic() + 5)  # Should not raise

    def test_slow_uncached_price_lookups_respect_the_deadline(self):
        """The actual bug this rewrite fixes: a bundle listing with several
        extractable game titles used to trigger SEQUENTIAL, blocking,
        non-deadline-aware eBay calls inside content-building — enough of
        them could alone burn more time than a whole search's deadline,
        before Gemini was ever called. Here, 5 uncached queries each taking
        2s (10s if still sequential) must not make enrich_deals itself run
        anywhere near that long."""

        def _slow_get_median(query, max_results=10):
            time.sleep(2.0)
            return 10.0, "sold_listings", []

        e = Enricher()
        mock_client = mock.MagicMock()
        mock_client.get_lowest_market_price.side_effect = _slow_get_median
        e.ebay_client = mock_client

        deal = {"title": "Xbox 360 Konvolut: Halo 3, Gears of War, Mass Effect, Fable 2, Dead Space"}
        deadline = time.monotonic() + 3
        t0 = time.monotonic()
        e.enrich_deals([deal], deadline)
        elapsed = time.monotonic() - t0

        assert elapsed < 4.0, f"enrich_deals blocked past its deadline (took {elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# Batch-size / timeout constants – regression guard
# ---------------------------------------------------------------------------


class TestBatchTimeoutConstants:
    """Regression tests that verify the batch-size and timeout constants stay
    within safe limits relative to the Gunicorn worker timeout (180 s)."""

    # Gunicorn timeout from Dockerfile
    _GUNICORN_TIMEOUT = 180

    def test_batch_size_reduced_for_lower_latency(self):
        """_BATCH_SIZE must be ≤ 5 so per-call prompts stay small."""
        assert _BATCH_SIZE <= 5, f"_BATCH_SIZE={_BATCH_SIZE} is too large; keep ≤ 5 for low per-call latency"

    def test_batch_size_positive(self):
        assert _BATCH_SIZE >= 1

    def test_per_call_timeout_fits_in_budget(self):
        """Each individual call timeout must be < total budget."""
        assert _GEMINI_REQUEST_TIMEOUT < _ASSESS_TOTAL_BUDGET_S

    def test_total_budget_leaves_gunicorn_headroom(self):
        """Enrichment's own budget ceiling + total Gemini budget must stay
        below the Gunicorn timeout."""
        assert _ENRICH_MAX_BUDGET_S + _ASSESS_TOTAL_BUDGET_S < self._GUNICORN_TIMEOUT, (
            "Combined enrichment + Gemini budget exceeds Gunicorn worker timeout"
        )


# ---------------------------------------------------------------------------
# assess_deals_batch deadline handling — regression tests for the Cloudflare
# 524 fix: /api/search now hands assess_deals_batch an absolute deadline
# (request-start + a total request budget) instead of the provider always
# getting its own independent ~145s allowance stacked on top of however
# long the search phase already took.
# ---------------------------------------------------------------------------


class TestAssessDealsBatchDeadline:
    def _make_enabled_assessor(self):
        from google.genai import types

        a = GeminiAssessor()
        a.enabled = True
        a.user_enabled = True
        a._client = mock.MagicMock()
        a._types = types  # real (network-free) request-builder classes
        return a

    def test_expired_deadline_skips_all_batches_without_calling_api(self):
        """An already-passed deadline must short-circuit every batch — no
        Gemini call is made, every deal comes back as None."""
        a = self._make_enabled_assessor()
        deals = [{"title": f"Game {i}", "url": f"http://x/{i}"} for i in range(7)]  # 2 batches
        results = a.assess_deals_batch(deals, deadline=time.monotonic() - 1)
        assert results == [None] * len(deals)
        a._client.models.generate_content.assert_not_called()

    def test_no_deadline_falls_back_to_default_budget(self):
        """Omitting deadline (e.g. a direct/standalone caller) must not
        crash — it falls back to _ASSESS_TOTAL_BUDGET_S from 'now'."""
        a = self._make_enabled_assessor()
        a._client.models.generate_content.side_effect = RuntimeError("boom")
        deals = [{"title": "Game", "url": "http://x/1"}]
        results = a.assess_deals_batch(deals)
        assert results == [None]

    def test_deadline_survives_text_only_model_fallback_recursion(self):
        """_assess_batch_with_retry re-invokes itself when it discovers the
        model is text-only mid-call; the deadline must be threaded through
        that recursive call too, not silently dropped (it originally was —
        the recursive call used to read ``self._assess_batch_with_retry(deals)``
        with no deadline argument at all)."""
        a = self._make_enabled_assessor()
        a._images_supported = True
        # First call: the SDK rejects the image part. Second call (after
        # images get disabled) fails a different, unrelated way so we can
        # tell the recursive call actually happened and isn't just a repeat.
        a._client.models.generate_content.side_effect = [
            Exception("does not support image input"),
            RuntimeError("second call"),
        ]
        deadline = time.monotonic() + 30  # plenty of time for both attempts
        deals = [{"title": "Game", "url": "http://x/1"}]
        enriched_batch = [EnrichedDeal()]
        with mock.patch.object(a, "_assess_batch_with_retry", wraps=a._assess_batch_with_retry) as spy:
            results = a._assess_batch_with_retry(deals, enriched_batch, deadline)
        assert results == [None]
        assert a._images_supported is False
        # Called twice: the original attempt, then the images-disabled retry.
        assert spy.call_count == 2
        # Both calls — including the recursive one — must carry the same
        # deadline rather than it being silently dropped.
        for call in spy.call_args_list:
            assert call.args[2] == deadline

    def test_slow_uncached_bundle_price_lookups_do_not_starve_the_batch(self):
        """The actual triggering bug, reproduced end-to-end and proven fixed:
        a bundle listing with several extractable game titles used to
        trigger SEQUENTIAL, blocking, non-deadline-aware eBay price lookups
        inside content-building — enough of them could alone burn more time
        than the whole batch's deadline, before Gemini was ever called. Here
        a real (mocked, slow) eBay client with 5 uncached queries (10s if
        still sequential) must not prevent the batch from completing well
        within its deadline, and the Gemini call must still happen."""
        a = self._make_enabled_assessor()
        mock_client = mock.MagicMock()

        def _slow_get_median(query, max_results=10):
            time.sleep(2.0)
            return 10.0, "sold_listings", []

        mock_client.get_lowest_market_price.side_effect = _slow_get_median
        a._enricher.ebay_client = mock_client

        def fast_generate_content(*, model, contents, config):
            resp = mock.MagicMock()
            resp.text = json.dumps([{"deal_rating": "Okay"}] * 1)
            return resp

        a._client.models.generate_content.side_effect = fast_generate_content

        deal = {
            "title": "Xbox 360 Konvolut: Halo 3, Gears of War, Mass Effect, Fable 2, Dead Space",
            "url": "http://x/1",
        }
        deadline = time.monotonic() + 4
        t0 = time.monotonic()
        results = a.assess_deals_batch([deal], deadline=deadline)
        elapsed = time.monotonic() - t0

        assert elapsed < 5.0, f"batch was starved by slow price lookups (took {elapsed:.2f}s)"
        assert results[0] is not None
        a._client.models.generate_content.assert_called()

    def test_batches_run_concurrently_not_sequentially(self, monkeypatch):
        """Empirical proof batches overlap in flight instead of running one
        at a time: 6 batches (30 deals — the app's _MAX_DISPLAY cap) each
        taking ~0.3s must finish well under the ~1.9s a fully sequential
        design (6 calls + 5 staggers) would take. This is what lets a full
        set of search results actually get AI ratings within a search's
        overall deadline instead of only the first batch or two."""
        import ai_providers.gemini as gemini_module

        monkeypatch.setattr(gemini_module, "_BATCH_DELAY_SECONDS", 0.02)
        a = self._make_enabled_assessor()

        def slow_generate_content(*, model, contents, config):
            time.sleep(0.3)
            resp = mock.MagicMock()
            resp.text = json.dumps([{"deal_rating": "Okay"}] * _BATCH_SIZE)
            return resp

        a._client.models.generate_content.side_effect = slow_generate_content
        deals = [{"title": f"Game {i}", "url": f"http://x/{i}"} for i in range(30)]  # 6 batches

        t0 = time.monotonic()
        results = a.assess_deals_batch(deals, deadline=time.monotonic() + 30)
        elapsed = time.monotonic() - t0

        assert len(results) == 30
        assert all(r is not None for r in results)
        assert elapsed < 1.2, f"batches do not appear to run concurrently (took {elapsed:.2f}s)"


# ---------------------------------------------------------------------------
# Enricher.enrich_deals — image fetching. Regression tests for a real
# latency bug: a batch's images (up to 3 per deal x 5 deals = 15) used to be
# fetched one at a time inside _build_batch_contents, entirely unbounded by
# the batch's own deadline/timeout since it all happened before the API call
# even started. Worst case (15 x the 5s per-image timeout = 75s) was large
# enough to eat most of a search's whole deadline on a single batch before
# it ever reached Gemini — surfacing in production as specific batches
# (whichever one drew the slow-to-fetch images) coming back with no rating
# at all. Image fetching now happens inside Enricher.enrich_deals, for the
# WHOLE filtered deal set at once, not per Gemini sub-batch.
# ---------------------------------------------------------------------------


class TestEnricherImageFetching:
    def test_fetches_concurrently_not_sequentially(self):
        """15 images (5 deals x 3 each), each taking 0.2s, must finish in
        well under the 3.0s a fully sequential fetch would take."""
        e = Enricher()
        deals = [{"image_urls": [f"http://x/{i}-{j}.jpg" for j in range(3)]} for i in range(5)]

        def slow_fetch(url):
            time.sleep(0.2)
            return (f"bytes:{url}".encode(), "image/jpeg")

        with mock.patch.object(e, "_fetch_one_image", side_effect=slow_fetch):
            t0 = time.monotonic()
            results = e.enrich_deals(deals, deadline=time.monotonic() + 5)
            elapsed = time.monotonic() - t0

        assert all(len(r.images) == 3 for r in results)
        assert elapsed < 1.0, f"images do not appear to be fetched concurrently (took {elapsed:.2f}s)"

    def test_preserves_per_deal_image_order_despite_concurrent_completion(self):
        """Deal N's images must come back in their original URL order even
        though different deals'/images' fetches complete in whatever order
        the thread pool happens to finish them in."""
        deals = [
            {"image_urls": ["http://x/a1.jpg", "http://x/a2.jpg", "http://x/a3.jpg"]},
            {"image_urls": ["http://x/b1.jpg"]},
        ]
        # Deliberately make earlier URLs slower so completion order is
        # scrambled relative to submission order.
        delays = {"http://x/a1.jpg": 0.15, "http://x/a2.jpg": 0.05, "http://x/a3.jpg": 0.10, "http://x/b1.jpg": 0.01}

        def fetch(url):
            time.sleep(delays[url])
            return (url.encode(), "image/jpeg")

        e = Enricher()
        with mock.patch.object(e, "_fetch_one_image", side_effect=fetch):
            results = e.enrich_deals(deals, deadline=time.monotonic() + 5)

        assert [data.decode() for data, _mime in results[0].images] == [
            "http://x/a1.jpg",
            "http://x/a2.jpg",
            "http://x/a3.jpg",
        ]
        assert [data.decode() for data, _mime in results[1].images] == ["http://x/b1.jpg"]

    def test_skips_failed_fetches(self):
        """A None return (fetch failed) is dropped, not kept as a placeholder."""
        deals = [{"image_urls": ["http://x/good.jpg", "http://x/bad.jpg"]}]

        def fetch(url):
            return None if "bad" in url else (url.encode(), "image/jpeg")

        e = Enricher()
        with mock.patch.object(e, "_fetch_one_image", side_effect=fetch):
            results = e.enrich_deals(deals, deadline=time.monotonic() + 5)

        assert len(results[0].images) == 1
        assert results[0].images[0][0] == b"http://x/good.jpg"

    def test_fetch_images_false_skips_fetch_entirely(self):
        """The caller (GeminiAssessor, when its active model is text-only)
        can skip image fetching entirely rather than fetching images it
        knows it will discard."""
        deals = [{"image_urls": ["http://x/a.jpg"]}]
        e = Enricher()
        with mock.patch.object(e, "_fetch_one_image") as mock_fetch:
            results = e.enrich_deals(deals, deadline=time.monotonic() + 5, fetch_images=False)
        assert results[0].images == []
        mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# Phase B (content-building) must make zero network calls of its own — all
# eBay/image I/O happens in Phase A (Enricher) beforehand. This is what
# makes Phase C's existing deadline-checked timeout actually sufficient;
# verified here by making any live network call raise.
# ---------------------------------------------------------------------------


class TestPhaseBIsNetworkFree:
    def _make_assessor(self):
        from google.genai import types

        a = GeminiAssessor()
        a._images_supported = True
        a._types = types
        return a

    def test_build_batch_contents_makes_no_network_calls(self, monkeypatch):
        import requests

        def _boom(*args, **kwargs):
            raise AssertionError("Phase B must not make network calls")

        monkeypatch.setattr(requests, "get", _boom)
        a = self._make_assessor()
        deals = [
            {
                "title": "Xbox 360 Konvolut: Halo 3, Gears of War",
                "image_urls": ["http://x/1.jpg"],
                "description": "",
            }
        ]
        enriched = [
            EnrichedDeal(
                bundle_prices=[{"game": "Halo 3", "price_eur": 12.0, "price_source": "ebay_sold"}],
                images=[(b"fake-bytes", "image/jpeg")],
            )
        ]
        a._build_batch_contents(deals, enriched)  # must not raise

    def test_build_contents_makes_no_network_calls(self, monkeypatch):
        import requests

        def _boom(*args, **kwargs):
            raise AssertionError("Phase B must not make network calls")

        monkeypatch.setattr(requests, "get", _boom)
        a = self._make_assessor()
        deal = {"title": "Halo 3 Xbox 360", "image_urls": ["http://x/1.jpg"], "description": ""}
        enriched = EnrichedDeal(single_price=12.0, images=[(b"fake-bytes", "image/jpeg")])
        a._build_contents(deal, enriched)  # must not raise


# ---------------------------------------------------------------------------
# Top-3 value games logic (frontend helper parity test)
# ---------------------------------------------------------------------------


class TestTopValueGamesSelection:
    """Verify the top-3 selection logic that the frontend uses is correct.

    The frontend selects items by:
      1. Filtering to price_eur > 0 and excluding aggregate placeholders.
      2. Sorting by price_eur descending (with Number() coercion for strings).
      3. Walking the sorted list and collecting up to 3 *unique* game names
         (first occurrence of each name wins — no duplicates fill a slot).

    These tests mirror that logic in Python to ensure the algorithm stays
    correct across edge cases.
    """

    @staticmethod
    def _top_value_games(itemized, max_top=3):
        """Python mirror of the updated JS top-value-games selection.

        Mirrors the JS algorithm exactly:
        - Excludes aggregate/placeholder entries via _is_aggregate_placeholder.
        - Coerces price_eur to float to handle string values.
        - Deduplicates by game name so a repeated title only occupies one slot.
        """
        eligible = [
            i
            for i in itemized
            if i.get("game")
            and not _is_aggregate_placeholder(i.get("game"))
            and i.get("price_eur") is not None
            and float(i["price_eur"]) > 0
        ]
        eligible.sort(key=lambda i: float(i["price_eur"]), reverse=True)
        seen: set = set()
        result = []
        for item in eligible:
            if len(seen) >= max_top:
                break
            if item["game"] not in seen:
                seen.add(item["game"])
                result.append(item)
        return result

    @staticmethod
    def _sort_itemized(itemized):
        """Python mirror of the sortedItemized JS sort.

        Null prices sort last (mapped to -1 so they appear at the bottom).
        """

        def sort_key(i):
            p = i.get("price_eur")
            return float(p) if p is not None else -1.0

        return sorted(itemized, key=sort_key, reverse=True)

    @staticmethod
    def _highlight_set(top_games):
        """Return the set of game names that should be highlighted."""
        return {i["game"] for i in top_games}

    # ── Core selection tests ──────────────────────────────────────────────────

    def test_top_3_selected_correctly(self):
        itemized = [
            {"game": "Halo 3", "price_eur": 12.0},
            {"game": "Mass Effect 2", "price_eur": 25.0},
            {"game": "Gears of War", "price_eur": 9.0},
            {"game": "Dead Space", "price_eur": 18.0},
            {"game": "Bioshock", "price_eur": 15.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 3
        assert [g["game"] for g in top] == ["Mass Effect 2", "Dead Space", "Bioshock"]
        assert top[0]["price_eur"] == 25.0

    def test_fewer_than_3_games_returns_all(self):
        itemized = [
            {"game": "Game A", "price_eur": 10.0},
            {"game": "Game B", "price_eur": 5.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 2

    def test_games_without_price_excluded(self):
        itemized = [
            {"game": "No Data", "price_eur": None},
            {"game": "Game A", "price_eur": 8.0},
            {"game": "Zero Price", "price_eur": 0},
            {"game": "Game B", "price_eur": 15.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 2
        assert top[0]["game"] == "Game B"
        assert top[1]["game"] == "Game A"

    def test_empty_itemized_returns_empty(self):
        assert self._top_value_games([]) == []

    def test_single_game_included(self):
        itemized = [{"game": "Halo 3", "price_eur": 12.0}]
        top = self._top_value_games(itemized)
        assert len(top) == 1
        assert top[0]["game"] == "Halo 3"

    def test_block_shown_only_for_good_or_better_with_2_plus_priced_games(self):
        """Top-value block requires ≥ 2 priced games to be shown (rating check
        is handled in the frontend; here we verify the price-count threshold)."""
        # Three priced games → block qualifies
        itemized = [
            {"game": "Game A", "price_eur": 12.0},
            {"game": "Game B", "price_eur": 8.0},
            {"game": "Game C", "price_eur": 5.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) >= 2, "Three priced games should produce ≥ 2 top results"

        # With only 1 priced game the block should not be shown (< 2 threshold)
        single = [{"game": "Game A", "price_eur": 12.0}]
        top = self._top_value_games(single)
        assert len(top) < 2, "Single priced game must not meet the ≥2 threshold"

        # Mix: 3 games but only 1 with a valid price → block should not be shown
        mostly_no_price = [
            {"game": "Game A", "price_eur": None},
            {"game": "Game B", "price_eur": 0},
            {"game": "Game C", "price_eur": 10.0},
        ]
        top = self._top_value_games(mostly_no_price)
        assert len(top) < 2, "Only one priced game — block threshold not met"

    # ── Edge cases: price ties ────────────────────────────────────────────────

    def test_ties_on_price_all_three_selected(self):
        """When multiple games share the same price, all are eligible; the first
        3 encountered after sorting (stable sort) fill the top-3 slots."""
        itemized = [
            {"game": "Game A", "price_eur": 10.0},
            {"game": "Game B", "price_eur": 10.0},
            {"game": "Game C", "price_eur": 10.0},
            {"game": "Game D", "price_eur": 10.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 3
        # All selected games must have price 10.0
        assert all(g["price_eur"] == 10.0 for g in top)

    def test_ties_exactly_three_unique_names(self):
        """Exactly 3 unique games with equal price → all 3 selected."""
        itemized = [
            {"game": "Alpha", "price_eur": 5.0},
            {"game": "Beta", "price_eur": 5.0},
            {"game": "Gamma", "price_eur": 5.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 3
        names = {g["game"] for g in top}
        assert names == {"Alpha", "Beta", "Gamma"}

    # ── Edge cases: duplicate game names ────────────────────────────────────

    def test_duplicate_game_name_counts_as_one_slot(self):
        """A game appearing twice in itemized must only fill ONE top-3 slot,
        so that the highlight count never exceeds min(3, unique_eligible)."""
        itemized = [
            {"game": "Halo 3", "price_eur": 25.0},
            {"game": "Halo 3", "price_eur": 25.0},  # duplicate
            {"game": "Gears", "price_eur": 18.0},
            {"game": "Mass Effect", "price_eur": 15.0},
            {"game": "Dead Space", "price_eur": 12.0},
        ]
        top = self._top_value_games(itemized)
        # Despite the duplicate, only 3 unique slots should be filled.
        assert len(top) == 3
        names = [g["game"] for g in top]
        assert names.count("Halo 3") == 1, "Duplicate title must not fill two slots"

    def test_all_duplicates_produce_one_result(self):
        """If all entries share the same game name, only 1 item is returned."""
        itemized = [
            {"game": "Halo 3", "price_eur": 20.0},
            {"game": "Halo 3", "price_eur": 18.0},
            {"game": "Halo 3", "price_eur": 15.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 1
        assert top[0]["game"] == "Halo 3"
        assert top[0]["price_eur"] == 20.0  # highest price wins

    def test_two_unique_names_with_duplicates(self):
        """Two unique names with duplicates → only 2 items returned, not 3."""
        itemized = [
            {"game": "Alpha", "price_eur": 10.0},
            {"game": "Alpha", "price_eur": 9.0},  # duplicate
            {"game": "Beta", "price_eur": 7.0},
            {"game": "Beta", "price_eur": 6.0},  # duplicate
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 2
        assert top[0]["game"] == "Alpha"
        assert top[1]["game"] == "Beta"

    # ── Edge cases: aggregate / placeholder exclusion ────────────────────────

    def test_aggregate_entries_excluded_from_top3(self):
        """Aggregate placeholder entries must never fill a top-3 slot."""
        itemized = [
            {"game": "Halo 3", "price_eur": 12.0},
            {"game": "Gears of War", "price_eur": 9.0},
            {"game": "Additional Titles", "price_eur": 50.0},  # aggregate — skip
            {"game": "Remaining Titles", "price_eur": 40.0},  # aggregate — skip
            {"game": "Dead Space", "price_eur": 6.0},
        ]
        top = self._top_value_games(itemized)
        names = [g["game"] for g in top]
        assert "Additional Titles" not in names
        assert "Remaining Titles" not in names
        assert "Halo 3" in names
        assert len(top) == 3

    def test_german_aggregate_excluded(self):
        """German aggregate placeholders (Weitere Spiele, Sonstige Titel) must
        be excluded even when they have a high estimated price."""
        itemized = [
            {"game": "Weitere Spiele", "price_eur": 100.0},  # aggregate — skip
            {"game": "Sonstige Titel", "price_eur": 80.0},  # aggregate — skip
            {"game": "Game A", "price_eur": 12.0},
            {"game": "Game B", "price_eur": 9.0},
            {"game": "Game C", "price_eur": 6.0},
        ]
        top = self._top_value_games(itemized)
        names = [g["game"] for g in top]
        assert "Weitere Spiele" not in names
        assert "Sonstige Titel" not in names
        assert "Game A" in names

    def test_bare_token_aggregate_excluded(self):
        """Bare tokens like '...' and 'etc.' must not appear in top-3."""
        itemized = [
            {"game": "...", "price_eur": 99.0},  # placeholder
            {"game": "etc.", "price_eur": 88.0},  # placeholder
            {"game": "Game A", "price_eur": 5.0},
        ]
        top = self._top_value_games(itemized)
        names = [g["game"] for g in top]
        assert "..." not in names
        assert "etc." not in names
        assert "Game A" in names

    def test_only_aggregates_returns_empty(self):
        """A list of only aggregate entries should return an empty top selection."""
        itemized = [
            {"game": "Additional Titles", "price_eur": 20.0},
            {"game": "Remaining Titles", "price_eur": 15.0},
        ]
        top = self._top_value_games(itemized)
        assert top == []

    # ── Edge cases: sorting ──────────────────────────────────────────────────

    def test_sorted_descending_by_price(self):
        """sortedItemized must always produce highest price first."""
        itemized = [
            {"game": "C", "price_eur": 5.0},
            {"game": "A", "price_eur": 30.0},
            {"game": "B", "price_eur": 15.0},
        ]
        sorted_list = self._sort_itemized(itemized)
        prices = [i["price_eur"] for i in sorted_list]
        assert prices == sorted(prices, reverse=True)

    def test_null_prices_sort_last(self):
        """Items with price_eur=None must appear after all priced items."""
        itemized = [
            {"game": "A", "price_eur": None},
            {"game": "B", "price_eur": 10.0},
            {"game": "C", "price_eur": None},
            {"game": "D", "price_eur": 5.0},
        ]
        sorted_list = self._sort_itemized(itemized)
        # Priced items come first
        assert sorted_list[0]["price_eur"] == 10.0
        assert sorted_list[1]["price_eur"] == 5.0
        # Null-price items are last
        assert sorted_list[2]["price_eur"] is None
        assert sorted_list[3]["price_eur"] is None

    def test_string_prices_coerced_correctly(self):
        """String price_eur values (e.g. from JSON) must sort correctly."""
        itemized = [
            {"game": "A", "price_eur": "5.0"},
            {"game": "B", "price_eur": "25.0"},
            {"game": "C", "price_eur": "12.5"},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 3
        assert top[0]["game"] == "B"
        assert top[1]["game"] == "C"
        assert top[2]["game"] == "A"

    # ── Edge cases: one-game list ────────────────────────────────────────────

    def test_one_game_list(self):
        """A single-game list returns exactly 1 item."""
        itemized = [{"game": "Only Game", "price_eur": 7.5}]
        top = self._top_value_games(itemized)
        assert len(top) == 1
        assert top[0]["game"] == "Only Game"

    def test_one_game_with_no_price(self):
        """A single game with no price → empty result."""
        itemized = [{"game": "Only Game", "price_eur": None}]
        top = self._top_value_games(itemized)
        assert top == []

    # ── Highlight count validation ────────────────────────────────────────────

    def test_highlight_count_never_exceeds_3(self):
        """The top3Set must never cause more than 3 rows to be highlighted,
        even when the same game name appears multiple times in the full list."""
        itemized = [
            {"game": "Halo 3", "price_eur": 20.0},
            {"game": "Halo 3", "price_eur": 18.0},  # duplicate
            {"game": "Gears", "price_eur": 15.0},
            {"game": "Gears", "price_eur": 14.0},  # duplicate
            {"game": "BioShock", "price_eur": 10.0},
            {"game": "FIFA 20", "price_eur": 2.0},
        ]
        top = self._top_value_games(itemized)
        highlight_set = self._highlight_set(top)
        # Simulate the frontend highlight pass: count rows that would be starred.
        highlighted_count = 0
        already_highlighted: set = set()
        sorted_list = self._sort_itemized(itemized)
        for item in sorted_list:
            if item["game"] in highlight_set and item["game"] not in already_highlighted:
                already_highlighted.add(item["game"])
                highlighted_count += 1
        assert highlighted_count <= 3, f"Expected ≤ 3 highlights, got {highlighted_count}"

    def test_highlight_count_equals_unique_eligible(self):
        """Highlighted count should equal min(3, unique games with price > 0)."""
        itemized = [
            {"game": "A", "price_eur": 10.0},
            {"game": "B", "price_eur": 8.0},
        ]
        top = self._top_value_games(itemized)
        assert len(top) == 2  # only 2 unique eligible → exactly 2 highlighted


# ---------------------------------------------------------------------------
# _is_aggregate_placeholder — detect grouped/bundled placeholder game entries
# ---------------------------------------------------------------------------


class TestIsAggregatePlaceholder:
    """Verify _is_aggregate_placeholder correctly identifies entries that
    should never appear in per-game resale breakdowns."""

    def test_additional_titles_is_placeholder(self):
        assert _is_aggregate_placeholder("Additional Titles") is True

    def test_remaining_titles_is_placeholder(self):
        assert _is_aggregate_placeholder("Remaining Titles") is True

    def test_other_games_is_placeholder(self):
        assert _is_aggregate_placeholder("Other Games") is True

    def test_more_games_is_placeholder(self):
        assert _is_aggregate_placeholder("More Games") is True

    def test_rest_of_games_is_placeholder(self):
        assert _is_aggregate_placeholder("Rest of Games") is True

    def test_weitere_spiele_is_placeholder(self):
        assert _is_aggregate_placeholder("Weitere Spiele") is True

    def test_sonstige_titel_is_placeholder(self):
        assert _is_aggregate_placeholder("Sonstige Titel") is True

    def test_etc_is_placeholder(self):
        assert _is_aggregate_placeholder("etc.") is True

    def test_ellipsis_is_placeholder(self):
        assert _is_aggregate_placeholder("...") is True

    def test_and_more_is_placeholder(self):
        assert _is_aggregate_placeholder("and more") is True

    def test_real_game_title_is_not_placeholder(self):
        assert _is_aggregate_placeholder("Halo 3") is False

    def test_zelda_is_not_placeholder(self):
        assert _is_aggregate_placeholder("The Legend of Zelda: Breath of the Wild") is False

    def test_batman_is_not_placeholder(self):
        assert _is_aggregate_placeholder("Batman Arkham Knight") is False

    def test_empty_string_is_not_placeholder(self):
        assert _is_aggregate_placeholder("") is False

    def test_non_string_is_not_placeholder(self):
        assert _is_aggregate_placeholder(None) is False
        assert _is_aggregate_placeholder(42) is False


class TestParseBatchResponseFiltersAggregates:
    """Verify _parse_batch_response removes aggregate placeholder entries
    from itemized_resale_estimates automatically."""

    def _parse(self, payload):
        return GeminiAssessor._parse_batch_response(json.dumps(payload), len(payload))

    def test_additional_titles_entry_removed(self):
        """An 'Additional Titles' aggregate entry must be stripped from results."""
        payload = [
            {
                "deal_rating": "Good",
                "confidence_score": 80,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€30",
                "itemized_resale_estimates": [
                    {"game": "Halo 3", "price_eur": 12.0, "price_source": "ebay_sold", "is_exceptional": False},
                    {"game": "Gears of War", "price_eur": 10.0, "price_source": "ebay_sold", "is_exceptional": False},
                    {
                        "game": "Additional Titles",
                        "price_eur": 8.0,
                        "price_source": "ai_estimate",
                        "is_exceptional": False,
                    },
                ],
                "estimated_total_cost": 15.0,
                "estimated_gross_profit": 15.0,
                "verdict_summary": "Good bundle.",
            }
        ]
        result = self._parse(payload)
        games = [e["game"] for e in result[0]["ai_itemized_resale_estimates"]]
        assert "Additional Titles" not in games
        assert "Halo 3" in games
        assert "Gears of War" in games
        assert len(games) == 2

    def test_remaining_titles_entry_removed(self):
        """A 'Remaining Titles' placeholder entry must be stripped."""
        payload = [
            {
                "deal_rating": "Okay",
                "confidence_score": 60,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€20",
                "itemized_resale_estimates": [
                    {
                        "game": "Batman Arkham Knight",
                        "price_eur": 8.0,
                        "price_source": "ebay_active",
                        "is_exceptional": False,
                    },
                    {
                        "game": "Remaining Titles",
                        "price_eur": 5.0,
                        "price_source": "ai_estimate",
                        "is_exceptional": False,
                    },
                ],
                "estimated_total_cost": 12.0,
                "estimated_gross_profit": 1.0,
                "verdict_summary": "Decent.",
            }
        ]
        result = self._parse(payload)
        games = [e["game"] for e in result[0]["ai_itemized_resale_estimates"]]
        assert "Remaining Titles" not in games
        assert "Batman Arkham Knight" in games

    def test_no_aggregate_entries_unchanged(self):
        """When no aggregate entries are present the list is returned unchanged."""
        payload = [
            {
                "deal_rating": "Must Have",
                "confidence_score": 95,
                "potential_scam": False,
                "scam_warning": "",
                "visual_findings": [],
                "red_flags": [],
                "fair_market_estimate": "~€60",
                "itemized_resale_estimates": [
                    {"game": "God of War", "price_eur": 20.0, "price_source": "ebay_sold", "is_exceptional": True},
                    {"game": "Spider-Man", "price_eur": 18.0, "price_source": "ebay_sold", "is_exceptional": False},
                    {
                        "game": "Horizon Zero Dawn",
                        "price_eur": 12.0,
                        "price_source": "ai_estimate",
                        "is_exceptional": False,
                    },
                ],
                "estimated_total_cost": 15.0,
                "estimated_gross_profit": 35.0,
                "verdict_summary": "Amazing deal.",
            }
        ]
        result = self._parse(payload)
        games = [e["game"] for e in result[0]["ai_itemized_resale_estimates"]]
        assert games == ["God of War", "Spider-Man", "Horizon Zero Dawn"]


# ---------------------------------------------------------------------------
# Tests for JSON control-character sanitisation
# ---------------------------------------------------------------------------


class TestSanitizeJsonText:
    """Verify _sanitize_json_text strips invalid JSON control characters."""

    def test_clean_text_unchanged(self):
        """Text without control characters is returned unchanged."""
        from ai_providers.base import _sanitize_json_text

        text = '[{"deal_rating": "Good", "verdict_summary": "Nice deal."}]'
        assert _sanitize_json_text(text) == text

    def test_strips_form_feed(self):
        """Form-feed (0x0C) is stripped."""
        from ai_providers.base import _sanitize_json_text

        text = '[{"verdict_summary": "Good\x0cdeal."}]'
        result = _sanitize_json_text(text)
        assert "\x0c" not in result
        assert "Gooddeal." in result

    def test_strips_backspace(self):
        """Backspace (0x08) is stripped."""
        from ai_providers.base import _sanitize_json_text

        text = "hello\x08world"
        assert _sanitize_json_text(text) == "helloworld"

    def test_preserves_tab_lf_cr(self):
        """TAB (0x09), LF (0x0A), CR (0x0D) are preserved (valid in JSON)."""
        from ai_providers.base import _sanitize_json_text

        text = "line1\nline2\r\n\ttabbed"
        assert _sanitize_json_text(text) == text

    def test_parse_batch_response_survives_control_char(self):
        """_parse_batch_response parses correctly when response contains
        invalid control characters that would otherwise cause json.loads to
        raise 'Invalid control character at …'."""
        item = {
            "deal_rating": "Good",
            "confidence_score": 75,
            "potential_scam": False,
            "scam_warning": "",
            "visual_findings": [],
            "red_flags": [],
            "fair_market_estimate": "~€25",
            "itemized_resale_estimates": [
                {"game": "Halo 3", "price_eur": 12.0, "price_source": "ebay_sold", "is_exceptional": False},
            ],
            "estimated_total_cost": 10.0,
            "estimated_gross_profit": 2.0,
            "verdict_summary": "Good deal.",
        }
        raw = json.dumps([item])
        # Inject a form-feed (0x0C) inside the verdict_summary string value.
        raw_with_ctrl = raw.replace("Good deal.", "Good\x0cdeal.")
        result = GeminiAssessor._parse_batch_response(raw_with_ctrl, 1)
        assert len(result) == 1
        assert result[0]["ai_assessed"] is True
        assert result[0]["ai_deal_rating"] == "Good"
        assert "ai_error_type" not in result[0]


# ---------------------------------------------------------------------------
# Tests for improved _extract_potential_game_titles (pipe separator, per-part
# cleanup)
# ---------------------------------------------------------------------------


class TestExtractPotentialGameTitlesPipeSeparator:
    """Tests for the pipe-separator and per-part noise-cleanup improvements."""

    def test_pipe_separator_splits_titles(self):
        """Pipe (|) is treated as a title separator."""
        title = "Assassins Creed Sammlung Xbox 360 | 1, 2, Brotherhood"
        result = _extract_potential_game_titles(title)
        # Should extract the series name from before the pipe
        assert any("Assassins Creed" in t for t in result)

    def test_quantity_prefix_stripped_from_part(self):
        """Leading '7x' quantity prefix is stripped from an extracted part."""
        title = "7x Assassins Creed Konvolut Sammlung komplett Xbox 360 | 1, 2, 3"
        result = _extract_potential_game_titles(title)
        # 'Assassins Creed' should be extracted; '7x' should NOT be in the result
        for t in result:
            assert not t.startswith("7x")
            assert not t.startswith("7X")

    def test_komplett_stripped_from_part(self):
        """Condition word 'komplett' is stripped from extracted parts."""
        title = "5x Halo komplett Sammlung Xbox 360 | Halo 3, Halo 4"
        result = _extract_potential_game_titles(title)
        for t in result:
            assert "komplett" not in t.lower()

    def test_platform_number_stripped_from_part(self):
        """Standalone platform version number '360' is removed (via full
        platform-pattern pass on the whole title before splitting)."""
        title = "7x Assassins Creed Konvolut Sammlung komplett Xbox 360 | 1, 2, 3"
        result = _extract_potential_game_titles(title)
        # '360' should not appear as a standalone suffix in any extracted part
        for t in result:
            assert not t.strip().endswith("360")

    def test_game_word_preserved_in_game_title(self):
        """'game' inside a real game title (e.g. 'Game Dev Tycoon') is kept."""
        title = "Bundle: Game Dev Tycoon, Game of Thrones, Dishonored"
        result = _extract_potential_game_titles(title)
        assert any("Game Dev Tycoon" in t for t in result)

    def test_assassins_creed_konvolut_realistic(self):
        """Realistic 'Assassins Creed Konvolut' log example extracts the
        series name without platform remnants."""
        title = "7x Assassins Creed Konvolut Sammlung komplett Xbox 360 | 1, 2, 3, 4, Brotherhoo…"
        result = _extract_potential_game_titles(title)
        assert len(result) >= 1
        # The first extracted game should be the clean series name
        assert any("Assassins Creed" in t and "360" not in t for t in result)


# ---------------------------------------------------------------------------
# Tests for normalised itemized_resale_estimates in _parse_batch_response
# ---------------------------------------------------------------------------


class TestParseBatchResponseNormaliseItemized:
    """Verify that itemized entries are always fully normalised (no None fields)."""

    def _make_item(self, itemized):
        return {
            "deal_rating": "Good",
            "confidence_score": 70,
            "potential_scam": False,
            "scam_warning": "",
            "visual_findings": [],
            "red_flags": [],
            "fair_market_estimate": "~€20",
            "itemized_resale_estimates": itemized,
            "estimated_total_cost": 10.0,
            "estimated_gross_profit": 5.0,
            "verdict_summary": "Decent.",
        }

    def _parse(self, payload):
        return GeminiAssessor._parse_batch_response(json.dumps(payload), len(payload))

    def test_null_price_eur_defaults_to_zero(self):
        """price_eur=null in AI response is normalised to 0.0."""
        payload = [
            self._make_item(
                [
                    {"game": "Halo 3", "price_eur": None, "price_source": "ai_estimate", "is_exceptional": False},
                ]
            )
        ]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["price_eur"] == 0.0
        assert isinstance(entry["price_eur"], float)

    def test_missing_price_source_defaults_to_ai_estimate(self):
        """Missing price_source defaults to 'ai_estimate'."""
        payload = [
            self._make_item(
                [
                    {"game": "God of War", "price_eur": 15.0},
                ]
            )
        ]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["price_source"] == "ai_estimate"

    def test_missing_game_name_entry_is_skipped(self):
        """Entries without a game name (empty string or None) are skipped."""
        payload = [
            self._make_item(
                [
                    {"game": "", "price_eur": 5.0, "price_source": "ai_estimate", "is_exceptional": False},
                    {"game": None, "price_eur": 5.0, "price_source": "ai_estimate", "is_exceptional": False},
                    {"game": "Dishonored", "price_eur": 10.0, "price_source": "ebay_sold", "is_exceptional": False},
                ]
            )
        ]
        result = self._parse(payload)
        entries = result[0]["ai_itemized_resale_estimates"]
        assert len(entries) == 1
        assert entries[0]["game"] == "Dishonored"

    def test_price_eur_string_coerced_to_float(self):
        """price_eur given as a string is coerced to float."""
        payload = [
            self._make_item(
                [
                    {"game": "Mass Effect", "price_eur": "8.50", "price_source": "ebay_sold", "is_exceptional": False},
                ]
            )
        ]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["price_eur"] == 8.50
        assert isinstance(entry["price_eur"], float)

    def test_is_exceptional_defaults_to_false(self):
        """is_exceptional defaults to False when not present."""
        payload = [
            self._make_item(
                [
                    {"game": "Zelda", "price_eur": 30.0, "price_source": "ebay_sold"},
                ]
            )
        ]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["is_exceptional"] is False
