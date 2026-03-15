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
        many_games = ", ".join([f"Game {i}" for i in range(20)])
        title = f"Bundle: {many_games}"
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
