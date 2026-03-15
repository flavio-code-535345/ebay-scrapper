#!/usr/bin/env python3
"""
Unit tests for gemini_assessor.py — focusing on the deterministic
bundle bait-and-switch scam detector introduced to catch the canonical
'Spielesammlung + Stückzahl + verfügbar/verkauft > 1' pattern, and
the sports/Kinect deal detector that filters out low-resale-value listings.
"""

import pytest

from gemini_assessor import (
    _apply_scam_override,
    _apply_sports_kinect_override,
    _detect_bundle_individual_sale_scam,
    _detect_sports_kinect_deal,
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


from gemini_assessor import _extract_potential_game_titles


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


from gemini_assessor import _extract_platform_name


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


from gemini_assessor import _build_single_game_search_query


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


import json
from gemini_assessor import GeminiAssessor


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
