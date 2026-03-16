#!/usr/bin/env python3
"""
Unit tests for gemini_assessor.py — focusing on the deterministic
bundle bait-and-switch scam detector introduced to catch the canonical
'Spielesammlung + Stückzahl + verfügbar/verkauft > 1' pattern, and
the sports/Kinect deal detector that filters out low-resale-value listings.
"""

import json
import unittest.mock as mock

import pytest

from gemini_assessor import (
    GeminiAssessor,
    _ASSESS_TOTAL_BUDGET_S,
    _BATCH_SIZE,
    _EBAY_CACHE_TTL,
    _EBAY_PREFETCH_BUDGET_S,
    _GEMINI_REQUEST_TIMEOUT,
    _apply_scam_override,
    _apply_sports_kinect_override,
    _build_single_game_search_query,
    _detect_bundle_individual_sale_scam,
    _detect_sports_kinect_deal,
    _extract_platform_name,
    _extract_potential_game_titles,
    _is_aggregate_placeholder,
)

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
        from gemini_assessor import _MAX_GAMES_PER_BUNDLE
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
                    {"game": "Batman Arkham Knight", "price_eur": 18.0, "price_source": "ebay_active"},
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
            {"deal_rating": "Must Have", "confidence_score": 90, "potential_scam": False,
             "scam_warning": "", "visual_findings": [], "red_flags": [],
             "fair_market_estimate": "~€50",
             "itemized_resale_estimates": [
                 {"game": "Halo 3 Xbox 360", "price_eur": 50.0, "price_source": "ebay_sold"}
             ],
             "estimated_total_cost": 10.0, "estimated_gross_profit": 40.0,
             "verdict_summary": "Amazing."},
            {"deal_rating": "Good", "confidence_score": 75, "potential_scam": False,
             "scam_warning": "", "visual_findings": [], "red_flags": [],
             "fair_market_estimate": "~€25",
             "itemized_resale_estimates": [
                 {"game": "Batman Arkham Knight PS4", "price_eur": 25.0, "price_source": "ebay_active"}
             ],
             "estimated_total_cost": 15.0, "estimated_gross_profit": 10.0,
             "verdict_summary": "Good."},
            {"deal_rating": "Okay", "confidence_score": 60, "potential_scam": False,
             "scam_warning": "", "visual_findings": [], "red_flags": [],
             "fair_market_estimate": "~€12",
             "itemized_resale_estimates": [
                 {"game": "Minecraft Xbox 360", "price_eur": 12.0, "price_source": "ai_estimate"}
             ],
             "estimated_total_cost": 10.0, "estimated_gross_profit": 2.0,
             "verdict_summary": "Decent."},
            {"deal_rating": "Avoid", "confidence_score": 85, "potential_scam": False,
             "scam_warning": "", "visual_findings": [], "red_flags": [],
             "fair_market_estimate": "~€8",
             "itemized_resale_estimates": [
                 {"game": "FIFA 22 PS4", "price_eur": 3.0, "price_source": "ebay_sold"}
             ],
             "estimated_total_cost": 10.0, "estimated_gross_profit": -7.0,
             "verdict_summary": "Loss."},
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
    """Tests for the in-memory eBay price cache helpers."""

    def _make_assessor(self):
        """Return a GeminiAssessor instance (no API key needed for cache tests)."""
        return GeminiAssessor()

    def test_cache_miss_returns_none(self):
        """A fresh assessor returns None for any query."""
        a = self._make_assessor()
        assert a._cached_ebay_price("Halo 3 (Microsoft Xbox 360)") is None

    def test_store_and_retrieve(self):
        """Stored price is returned on the next lookup."""
        a = self._make_assessor()
        a._store_ebay_price_in_cache("Halo 3 (Microsoft Xbox 360)", 12.50, "sold_listings")
        result = a._cached_ebay_price("Halo 3 (Microsoft Xbox 360)")
        assert result is not None
        price, source = result
        assert price == pytest.approx(12.50)
        assert source == "sold_listings"

    def test_store_none_price_is_cached(self):
        """A None price (no eBay result) is also cached to avoid retrying."""
        a = self._make_assessor()
        a._store_ebay_price_in_cache("Unknown Game (Nintendo Switch)", None, "no_result")
        result = a._cached_ebay_price("Unknown Game (Nintendo Switch)")
        assert result is not None
        price, source = result
        assert price is None
        assert source == "no_result"

    def test_cache_entry_expires_after_ttl(self):
        """Cache entries are evicted when their TTL has elapsed."""
        import time as _time
        a = self._make_assessor()
        a._store_ebay_price_in_cache("God of War (Sony PlayStation 4)", 20.0, "active_listings")
        # Manually expire the entry by backdating its timestamp.
        query = "God of War (Sony PlayStation 4)"
        price, source, expire_at = a._ebay_price_cache[query]
        a._ebay_price_cache[query] = (price, source, _time.monotonic() - 1.0)
        assert a._cached_ebay_price(query) is None
        # Evicted entry should be removed from the dict.
        assert query not in a._ebay_price_cache

    def test_separate_queries_do_not_collide(self):
        """Different queries are stored and retrieved independently."""
        a = self._make_assessor()
        a._store_ebay_price_in_cache("Halo 3 (Microsoft Xbox 360)", 12.0, "sold_listings")
        a._store_ebay_price_in_cache("Zelda (Nintendo Switch)", 35.0, "active_listings")
        r1 = a._cached_ebay_price("Halo 3 (Microsoft Xbox 360)")
        r2 = a._cached_ebay_price("Zelda (Nintendo Switch)")
        assert r1 is not None and r1[0] == pytest.approx(12.0)
        assert r2 is not None and r2[0] == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# _collect_ebay_queries_for_deal
# ---------------------------------------------------------------------------


class TestCollectEbayQueriesForDeal:
    """Tests for the per-deal query-collection helper."""

    def _make_assessor_with_client(self):
        a = GeminiAssessor()
        # Provide a mock eBay client so the method knows to collect queries.
        a._ebay_client = object()
        return a

    def test_single_game_xbox360(self):
        """Single-game Xbox 360 listing produces a 'GAME (Microsoft Xbox 360)' query."""
        a = self._make_assessor_with_client()
        deal = {"title": "Halo 3 Xbox 360 gebraucht"}
        queries = a._collect_ebay_queries_for_deal(deal)
        assert len(queries) == 1
        assert queries[0].endswith("(Microsoft Xbox 360)")
        assert "Halo" in queries[0]

    def test_single_game_ps4(self):
        """Single-game PS4 listing produces a 'GAME (Sony PlayStation 4)' query."""
        a = self._make_assessor_with_client()
        deal = {"title": "God of War PS4"}
        queries = a._collect_ebay_queries_for_deal(deal)
        assert len(queries) == 1
        assert "(Sony PlayStation 4)" in queries[0]

    def test_bundle_produces_multiple_queries(self):
        """Bundle listing produces one query per extracted game title."""
        a = self._make_assessor_with_client()
        deal = {"title": "Xbox 360 Bundle: Halo 3, Gears of War, Mass Effect"}
        queries = a._collect_ebay_queries_for_deal(deal)
        # Should have at least 2 game queries
        assert len(queries) >= 2
        # Every query should include the platform
        for q in queries:
            assert "(Microsoft Xbox 360)" in q

    def test_bundle_no_titles_returns_empty(self):
        """Bundle keyword present but no extractable titles → empty list."""
        a = self._make_assessor_with_client()
        deal = {"title": "10 Xbox 360 Spiele Sammlung Lot"}
        queries = a._collect_ebay_queries_for_deal(deal)
        # Either empty or only platform-free generic words
        for q in queries:
            # Should not be a bare platform keyword alone
            assert len(q.strip()) >= 3

    def test_no_ebay_client_returns_empty(self):
        """Returns empty list when no eBay client is registered."""
        a = GeminiAssessor()
        a._ebay_client = None
        deal = {"title": "Halo 3 Xbox 360"}
        assert a._collect_ebay_queries_for_deal(deal) == []

    def test_empty_title_returns_empty(self):
        """Empty title returns empty list."""
        a = self._make_assessor_with_client()
        assert a._collect_ebay_queries_for_deal({"title": ""}) == []

    def test_missing_title_returns_empty(self):
        """Missing title key returns empty list."""
        a = self._make_assessor_with_client()
        assert a._collect_ebay_queries_for_deal({}) == []


# ---------------------------------------------------------------------------
# _prefetch_ebay_prices_parallel
# ---------------------------------------------------------------------------


class TestPrefetchEbayPricesParallel:
    """Tests for the parallel eBay price pre-fetcher."""

    def _make_assessor_with_mock_client(self, price_map=None):
        """Return a GeminiAssessor with a mock eBay client.

        *price_map* maps query substrings to (price, source) tuples so tests
        can simulate different eBay API outcomes.
        """
        a = GeminiAssessor()
        price_map = price_map or {}

        def _mock_get_median(query, max_results=10):
            for key, (price, source) in price_map.items():
                if key in query:
                    return price, source, []
            return None, "no_result", []

        mock_client = mock.MagicMock()
        mock_client.get_median_sold_price.side_effect = _mock_get_median
        a._ebay_client = mock_client
        return a

    def test_populates_cache_for_single_game_deal(self):
        """Pre-fetch populates the cache for a single-game deal."""
        a = self._make_assessor_with_mock_client({"Halo": (12.0, "sold_listings")})
        deals = [{"title": "Halo 3 Xbox 360"}]
        a._prefetch_ebay_prices_parallel(deals)
        # Cache should now have a Halo 3 entry.
        found = any(
            a._cached_ebay_price(q) is not None
            for q in a._collect_ebay_queries_for_deal(deals[0])
        )
        assert found, "Expected Halo 3 price to be in cache after prefetch"

    def test_populates_cache_for_bundle_deal(self):
        """Pre-fetch populates cache entries for games in a bundle listing."""
        a = self._make_assessor_with_mock_client({
            "Halo": (12.0, "sold_listings"),
            "Gears": (9.0, "sold_listings"),
        })
        deals = [{"title": "Xbox 360 Bundle: Halo 3, Gears of War"}]
        a._prefetch_ebay_prices_parallel(deals)
        queries = a._collect_ebay_queries_for_deal(deals[0])
        assert len(queries) >= 1
        # At least one game should be cached.
        assert any(a._cached_ebay_price(q) is not None for q in queries)

    def test_deduplicates_queries_across_deals(self):
        """Same game appearing in multiple deals triggers only one eBay call."""
        call_count = {"n": 0}

        def _mock_get_median(query, max_results=10):
            call_count["n"] += 1
            return 10.0, "sold_listings", []

        a = GeminiAssessor()
        mock_client = mock.MagicMock()
        mock_client.get_median_sold_price.side_effect = _mock_get_median
        a._ebay_client = mock_client

        # Two deals with the same title → same query → should deduplicate.
        deals = [
            {"title": "Halo 3 Xbox 360"},
            {"title": "Halo 3 Xbox 360"},
        ]
        a._prefetch_ebay_prices_parallel(deals)
        assert call_count["n"] == 1, "Duplicate queries should only be fetched once"

    def test_uses_cache_on_second_call(self):
        """A second prefetch for the same deals does not make any eBay API calls."""
        call_count = {"n": 0}

        def _mock_get_median(query, max_results=10):
            call_count["n"] += 1
            return 10.0, "sold_listings", []

        a = GeminiAssessor()
        mock_client = mock.MagicMock()
        mock_client.get_median_sold_price.side_effect = _mock_get_median
        a._ebay_client = mock_client

        deals = [{"title": "Halo 3 Xbox 360"}]
        a._prefetch_ebay_prices_parallel(deals)
        first_count = call_count["n"]
        a._prefetch_ebay_prices_parallel(deals)
        assert call_count["n"] == first_count, "Second prefetch should hit cache, not call eBay again"

    def test_no_ebay_client_is_noop(self):
        """When no eBay client is registered the method returns without error."""
        a = GeminiAssessor()
        a._ebay_client = None
        deals = [{"title": "Halo 3 Xbox 360"}]
        a._prefetch_ebay_prices_parallel(deals)  # Should not raise

    def test_failed_ebay_call_does_not_raise(self):
        """A failing eBay API call is silently absorbed; the cache is not poisoned."""
        a = GeminiAssessor()
        mock_client = mock.MagicMock()
        mock_client.get_median_sold_price.side_effect = RuntimeError("connection refused")
        a._ebay_client = mock_client
        deals = [{"title": "Halo 3 Xbox 360"}]
        a._prefetch_ebay_prices_parallel(deals)  # Should not raise


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
        assert _BATCH_SIZE <= 5, (
            f"_BATCH_SIZE={_BATCH_SIZE} is too large; keep ≤ 5 for low per-call latency"
        )

    def test_batch_size_positive(self):
        assert _BATCH_SIZE >= 1

    def test_per_call_timeout_fits_in_budget(self):
        """Each individual call timeout must be < total budget."""
        assert _GEMINI_REQUEST_TIMEOUT < _ASSESS_TOTAL_BUDGET_S

    def test_total_budget_leaves_gunicorn_headroom(self):
        """eBay pre-fetch + total Gemini budget must stay below Gunicorn timeout."""
        assert _EBAY_PREFETCH_BUDGET_S + _ASSESS_TOTAL_BUDGET_S < self._GUNICORN_TIMEOUT, (
            "Combined eBay prefetch + Gemini budget exceeds Gunicorn worker timeout"
        )


# ---------------------------------------------------------------------------
# Top-3 value games logic (frontend helper parity test)
# ---------------------------------------------------------------------------


class TestTopValueGamesSelection:
    """Verify the top-3 selection logic that the frontend uses is correct.

    The frontend selects items by filtering to price_eur > 0, sorting by
    price_eur descending, then taking the first 3.  These tests mirror that
    logic in Python to ensure the algorithm stays correct.
    """

    @staticmethod
    def _top_value_games(itemized, max_top=3):
        """Python mirror of the JS top-value-games selection."""
        with_prices = [
            i for i in itemized
            if i.get("price_eur") is not None and i["price_eur"] > 0
        ]
        with_prices.sort(key=lambda i: i["price_eur"], reverse=True)
        return with_prices[:max_top]

    def test_top_3_selected_correctly(self):
        itemized = [
            {"game": "Halo 3",          "price_eur": 12.0},
            {"game": "Mass Effect 2",    "price_eur": 25.0},
            {"game": "Gears of War",     "price_eur": 9.0},
            {"game": "Dead Space",       "price_eur": 18.0},
            {"game": "Bioshock",         "price_eur": 15.0},
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
            {"game": "No Data",   "price_eur": None},
            {"game": "Game A",    "price_eur": 8.0},
            {"game": "Zero Price","price_eur": 0},
            {"game": "Game B",    "price_eur": 15.0},
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
                    {"game": "Halo 3", "price_eur": 12.0, "price_source": "ebay_sold",
                     "is_exceptional": False},
                    {"game": "Gears of War", "price_eur": 10.0, "price_source": "ebay_sold",
                     "is_exceptional": False},
                    {"game": "Additional Titles", "price_eur": 8.0,
                     "price_source": "ai_estimate", "is_exceptional": False},
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
                    {"game": "Batman Arkham Knight", "price_eur": 8.0,
                     "price_source": "ebay_active", "is_exceptional": False},
                    {"game": "Remaining Titles", "price_eur": 5.0,
                     "price_source": "ai_estimate", "is_exceptional": False},
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
                    {"game": "God of War", "price_eur": 20.0, "price_source": "ebay_sold",
                     "is_exceptional": True},
                    {"game": "Spider-Man", "price_eur": 18.0, "price_source": "ebay_sold",
                     "is_exceptional": False},
                    {"game": "Horizon Zero Dawn", "price_eur": 12.0,
                     "price_source": "ai_estimate", "is_exceptional": False},
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
        from gemini_assessor import _sanitize_json_text
        text = '[{"deal_rating": "Good", "verdict_summary": "Nice deal."}]'
        assert _sanitize_json_text(text) == text

    def test_strips_form_feed(self):
        """Form-feed (0x0C) is stripped."""
        from gemini_assessor import _sanitize_json_text
        text = '[{"verdict_summary": "Good\x0cdeal."}]'
        result = _sanitize_json_text(text)
        assert "\x0c" not in result
        assert "Gooddeal." in result

    def test_strips_backspace(self):
        """Backspace (0x08) is stripped."""
        from gemini_assessor import _sanitize_json_text
        text = 'hello\x08world'
        assert _sanitize_json_text(text) == "helloworld"

    def test_preserves_tab_lf_cr(self):
        """TAB (0x09), LF (0x0A), CR (0x0D) are preserved (valid in JSON)."""
        from gemini_assessor import _sanitize_json_text
        text = 'line1\nline2\r\n\ttabbed'
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
                {"game": "Halo 3", "price_eur": 12.0, "price_source": "ebay_sold",
                 "is_exceptional": False},
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
        title = (
            "7x Assassins Creed Konvolut Sammlung komplett Xbox 360 "
            "| 1, 2, 3, 4, Brotherhoo…"
        )
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
        payload = [self._make_item([
            {"game": "Halo 3", "price_eur": None, "price_source": "ai_estimate",
             "is_exceptional": False},
        ])]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["price_eur"] == 0.0
        assert isinstance(entry["price_eur"], float)

    def test_missing_price_source_defaults_to_ai_estimate(self):
        """Missing price_source defaults to 'ai_estimate'."""
        payload = [self._make_item([
            {"game": "God of War", "price_eur": 15.0},
        ])]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["price_source"] == "ai_estimate"

    def test_missing_game_name_entry_is_skipped(self):
        """Entries without a game name (empty string or None) are skipped."""
        payload = [self._make_item([
            {"game": "", "price_eur": 5.0, "price_source": "ai_estimate",
             "is_exceptional": False},
            {"game": None, "price_eur": 5.0, "price_source": "ai_estimate",
             "is_exceptional": False},
            {"game": "Dishonored", "price_eur": 10.0, "price_source": "ebay_sold",
             "is_exceptional": False},
        ])]
        result = self._parse(payload)
        entries = result[0]["ai_itemized_resale_estimates"]
        assert len(entries) == 1
        assert entries[0]["game"] == "Dishonored"

    def test_price_eur_string_coerced_to_float(self):
        """price_eur given as a string is coerced to float."""
        payload = [self._make_item([
            {"game": "Mass Effect", "price_eur": "8.50", "price_source": "ebay_sold",
             "is_exceptional": False},
        ])]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["price_eur"] == 8.50
        assert isinstance(entry["price_eur"], float)

    def test_is_exceptional_defaults_to_false(self):
        """is_exceptional defaults to False when not present."""
        payload = [self._make_item([
            {"game": "Zelda", "price_eur": 30.0, "price_source": "ebay_sold"},
        ])]
        result = self._parse(payload)
        entry = result[0]["ai_itemized_resale_estimates"][0]
        assert entry["is_exceptional"] is False
