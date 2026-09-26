"""Tests for the "does the price buy the bundle?" checks in ai_providers/base.py:
per-game prices, pick-one listings, stated whole-lot prices, make-an-offer
placeholders and single games dressed up as bundles."""

from pathlib import Path

import pytest

from ai_providers.base import (
    _apply_scam_override,
    _detect_bundle_individual_sale_scam,
    analyze_price_scope,
    bundle_game_count,
    is_offer_placeholder,
    is_single_game_listing,
    looks_like_multi_item,
)
from ai_providers.enrichment import EnrichedDeal
from ai_providers.gemini import GeminiAssessor
from kleinanzeigen_scraper import parse_ad_description

_FIXTURES = Path(__file__).parent / "fixtures"
_EXAMPLE_TITLE = "10 PlayStation 3 Spiele Sammlung / 6 PS 4 spiele"
# The full description of that real ad (search results showed only its first ~100 characters).
_EXAMPLE_DESCRIPTION = parse_ad_description((_FIXTURES / "kleinanzeigen_ad.html").read_text(encoding="utf-8"))
_BUNDLE = "PS4 Spiele Sammlung"


def _scope(description, price, title=_BUNDLE):
    return analyze_price_scope({"title": title, "description": description, "price": price})


class TestRealListing:
    """ "Stück preis 7euro VB oder komplett paket 120 euro inkl versand" on a
    16-game lot listed at "7 € VB" — rated "Must Have" before this existed."""

    def test_listed_price_is_per_game_and_the_lot_price_is_found(self):
        scope = _scope(_EXAMPLE_DESCRIPTION, 7.0, title=_EXAMPLE_TITLE)
        assert scope.kind == "per_item"
        assert scope.evidence == "Stück preis"
        assert scope.item_price == 7.0
        assert scope.lot_price == 120.0
        assert scope.lot_includes_shipping is True

    def test_explanation(self):
        deal = {"title": _EXAMPLE_TITLE, "description": _EXAMPLE_DESCRIPTION, "price": 7.0}
        assert analyze_price_scope(deal).explain(deal) == (
            "Listed €7.00 is the price per game ('Stück preis'); "
            "the whole lot costs €120.00 incl. shipping according to the description."
        )

    def test_priced_at_the_lot_price_it_is_a_normal_bundle(self):
        """Once re-priced to 120 € the listing is judged on its merits."""
        assert _scope(_EXAMPLE_DESCRIPTION, 120.0, title=_EXAMPLE_TITLE) is None

    def test_game_count_adds_up_both_platforms(self):
        assert bundle_game_count(_EXAMPLE_TITLE) == 16


class TestPriceIsNotForTheLot:
    @pytest.mark.parametrize(
        ("description", "kind", "item_price"),
        [
            ("Stückpreis 5€", "per_item", 5.0),
            ("Stk.-Preis: 4 €", "per_item", 4.0),
            ("Einzelpreis 3 Euro", "per_item", 3.0),
            ("Der Preis gilt pro Spiel!", "per_item", None),
            ("Preis gilt für ein Spiel", "per_item", None),
            ("5 € das Stück", "per_item", 5.0),
            ("5€/Stk", "per_item", 5.0),
            ("Spiele je 5 €", "per_item", 5.0),
            ("alle Spiele à 3€", "per_item", 3.0),
            ("Alle 20 Spiele 5 € pro Stück", "per_item", 5.0),  # "alle 20 Spiele 5 €" is not a lot price
            ("Preise je nach Spiel", "price_varies", None),
            ("Such dir ein Spiel aus", "buyer_picks", None),
            ("Ein Spiel Ihrer Wahl", "buyer_picks", None),
            ("Welches Spiel möchtest du haben?", "buyer_picks", None),
            ("Die Spiele werden einzeln verkauft.", "sold_singly", None),
            ("Nur Einzelverkauf.", "sold_singly", None),
        ],
    )
    def test_description(self, description, kind, item_price):
        scope = _scope(description, item_price or 5.0)
        assert (scope.kind, scope.item_price) == (kind, item_price)

    @pytest.mark.parametrize(
        ("title", "kind"),
        [
            ("PS4 Spiele ab 3€", "price_varies"),
            ("PS4 Spiele Auswahl", "buyer_picks"),
            ("Xbox Spiele je 2 €", "per_item"),
        ],
    )
    def test_title(self, title, kind):
        assert _scope("", 3.0, title=title).kind == kind

    def test_game_count_from_the_description_when_the_title_has_none(self):
        """Seen live: "Sammlung von 12 verschiedenen PlayStation 3 Spielen … Pro Spiel 10€", listed at 10 €."""
        deal = {
            "title": "PlayStation 3 Spiele PS3 Games Konvolut",
            "description": "Meine Sammlung von 12 verschiedenen PlayStation 3 Spielen.\n\nPro Spiel 10€",
            "price": 10.0,
        }
        assert analyze_price_scope(deal).explain(deal) == (
            "Price is per game, not for the bundle ('Pro Spiel') — all 12 would cost about €120."
        )

    def test_whole_lot_price_alongside_a_piece_price(self):
        scope = _scope("Gesamtpreis 80€ oder Stückpreis 5€", 5.0)
        assert (scope.item_price, scope.lot_price, scope.lot_includes_shipping) == (5.0, 80.0, False)


class TestPriceIsForTheLot:
    @pytest.mark.parametrize(
        ("description", "price"),
        [
            ("Preis gilt für alle Spiele, nicht pro Stück.", 40.0),
            ("Kein Stückpreis!", 40.0),
            ("Werden nicht einzeln verkauft.", 40.0),
            ("Keine Einzelverkäufe", 40.0),
            ("Einzelverkauf oder komplett Abnahme möglich", 40.0),  # both offered: price may be the lot's
            ("Versand pro Spiel 1,50€", 20.0),
            ("Versandkosten betragen 1,50 € pro Spiel", 20.0),
            ("Versand ab 4,99 €", 20.0),
            ("Ich biete hier eine Auswahl an verschiedenen Videospielen", 20.0),
            ("Nur ein Spiel hat leichte Kratzer", 20.0),
            ("Bei Fragen bitte per Nachricht melden", 20.0),
            ("Einzelpreis 5€, alle zusammen 60€", 60.0),  # listed at the lot price
            ("Stückpreis 5 €", 60.0),  # 60 € is clearly not the 5 € piece price
        ],
    )
    def test_not_flagged(self, description, price):
        assert _scope(description, price) is None

    def test_single_item_may_be_priced_per_piece(self):
        """ "Preis pro Stück, 3 vorhanden" on one controller is honest."""
        assert _scope("Preis pro Stück, 3 vorhanden", 15.0, title="Xbox 360 Controller schwarz") is None
        assert not looks_like_multi_item("Xbox 360 Controller schwarz")


class TestSingleGameListings:
    """Seen live on Kleinanzeigen: one game padded with bundle words for search."""

    @pytest.mark.parametrize(
        "title",
        [
            "Battlefield 1 PS4 Spiel Sammlung PS2 PS3 PS5 Konvolut Bundle Top",
            "Dragon Ball Xenoverse PS4 Spiel aus Sammlung PS2 PS3 PS5 Konvolut",
            "LEGO Marvel Super Heroes PS3 Spiel Sammlung Konvolut PS2 PS4 PS5",
        ],
    )
    def test_padded_titles(self, title):
        assert is_single_game_listing({"title": title})

    def test_description_says_one_game(self):
        text = parse_ad_description((_FIXTURES / "kleinanzeigen_ad_new_layout.html").read_text(encoding="utf-8"))
        assert text.startswith("Ich verkaufe das Spiel Battlefield 1")
        assert is_single_game_listing({"title": "PS4 Konvolut", "description": text})
        # From the captured search page: one sealed game "aus Sammlungsauflösung".
        assert is_single_game_listing(
            {
                "title": "Resident Evil Revelations Xbox 360 Sealed Sammlung in OVP FSK16",
                "description": "Biete hier diesen Top Titel an. Resident Evil Revelations! Aus Sammlungsauflösung!",
            }
        )

    @pytest.mark.parametrize(
        ("title", "description"),
        [
            ("LEGO Marvel Collection PS4 – 3 Spiele in einer Sammlung", ""),
            ("Assassin's Creed PS3-Sammlung (4 Spiele)", ""),
            ("Xbox 360 Spiel Sammlung", "Sammlung von 12 Spielen"),
            (_EXAMPLE_TITLE, _EXAMPLE_DESCRIPTION),
        ],
    )
    def test_real_bundles(self, title, description):
        assert not is_single_game_listing({"title": title, "description": description})


def test_offer_placeholder():
    assert is_offer_placeholder({"price": 1.0, "shipping_note": "VB"})
    assert is_offer_placeholder({"price": 0.0, "shipping_note": "VB"})
    assert not is_offer_placeholder({"price": 45.0, "shipping_note": "VB"})
    assert not is_offer_placeholder({"price": 1.0, "shipping_note": ""})


class TestAssessorOverride:
    def test_per_game_price_is_avoided_even_if_the_ai_said_must_have(self):
        deal = {"title": _EXAMPLE_TITLE, "description": "Stückpreis 7€", "price": 7.0}
        result = _apply_scam_override(deal, {"ai_deal_rating": "Must Have", "ai_verdict_summary": "Grab it."})
        assert result["ai_deal_rating"] == "Avoid"
        assert result["ai_verdict_summary"].startswith("⚠️ **NOT A BUNDLE PRICE — AVOID**: Price is per game")
        assert "all 16 would cost about €112" in result["ai_scam_warning"]

    def test_listing_repriced_to_its_lot_price_keeps_the_ai_rating(self):
        deal = {"title": _EXAMPLE_TITLE, "description": _EXAMPLE_DESCRIPTION, "price": 120.0, "listed_price": 7.0}
        assert _detect_bundle_individual_sale_scam(deal) is None

    def test_prompt_carries_the_price_note(self):
        assessor = GeminiAssessor()
        deal = {
            "title": _EXAMPLE_TITLE,
            "price": 120.0,
            "listed_price": 7.0,
            "price_note": "Listed €7.00 is the price per game ('Stück preis'); the whole lot costs €120.00.",
        }
        text = assessor._format_deal_text(deal, EnrichedDeal(), header="--- ITEM 1 ---", description_limit=800)
        assert "Price: €120.0\nListed price (per game): €7.0\nPrice note: Listed €7.00 is the price per game" in text
