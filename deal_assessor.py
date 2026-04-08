#!/usr/bin/env python3
"""
Deal Assessment Engine
Evaluates and scores eBay deals based on multiple criteria
"""

import logging
from typing import Dict
import statistics

logger = logging.getLogger(__name__)

class DealAssessor:
    """Intelligent deal assessment system"""
    
    # Scoring weights
    PRICE_WEIGHT = 0.40
    SELLER_WEIGHT = 0.25
    CONDITION_WEIGHT = 0.20
    TREND_WEIGHT = 0.15
    
    # Price threshold (deals below this are "excellent")
    PRICE_THRESHOLD_PERCENTILE = 0.30

    # Category-specific minimum profit-margin thresholds (as a fraction of price).
    # Used by _score_price when ai_category is known.  A higher threshold means
    # only better deals score well in that category.
    CATEGORY_MARGIN_THRESHOLDS: Dict[str, float] = {
        # Electronics: slim margins due to high competition and quick depreciation.
        "electronics": 0.15,
        "smartphones": 0.12,
        "tablets": 0.12,
        "laptops": 0.15,
        "cameras": 0.18,
        # Gaming: good margins on consoles, moderate on accessories.
        "gaming": 0.20,
        "consoles": 0.20,
        "video games": 0.25,
        # Watches & jewellery: high value, good margins possible.
        "watches": 0.30,
        "jewellery": 0.30,
        "jewelry": 0.30,
        # Clothing & footwear: sneaker flipping especially profitable.
        "sneakers": 0.35,
        "shoes": 0.25,
        "clothing": 0.20,
        # Collectibles & toys.
        "collectibles": 0.30,
        "toys": 0.25,
        # Books & media: lower margins.
        "books": 0.40,
        "music": 0.30,
        "dvd": 0.30,
        "blu-ray": 0.30,
    }

    # Default margin threshold when category is unknown.
    DEFAULT_MARGIN_THRESHOLD: float = 0.20
    
    def __init__(self):
        self.market_data = {}  # Store historical prices for comparison
    
    def assess_deal(self, deal: Dict) -> Dict:
        """
        Assess a deal and return scores

        Returns dict with:
        - price_score: 0-100
        - seller_score: 0-100
        - condition_score: 0-100
        - trend_score: 0-100
        - overall_score: 0-100
        - recommendation: Text recommendation
        """
        try:
            price_score = self._score_price(deal)
            seller_score = self._score_seller(deal)
            condition_score = self._score_condition(deal)
            trend_score = self._score_trends(deal)

            # Calculate weighted overall score
            overall_score = (
                price_score * self.PRICE_WEIGHT +
                seller_score * self.SELLER_WEIGHT +
                condition_score * self.CONDITION_WEIGHT +
                trend_score * self.TREND_WEIGHT
            )

            recommendation = self._get_recommendation(overall_score)

            return {
                'price_score': price_score,
                'seller_score': seller_score,
                'condition_score': condition_score,
                'trend_score': trend_score,
                'overall_score': overall_score,
                'recommendation': recommendation
            }
        except Exception as exc:
            logger.error("Error assessing deal %r: %s", deal.get('title', '?'), exc, exc_info=True)
            return {
                'price_score': 0,
                'seller_score': 0,
                'condition_score': 0,
                'trend_score': 0,
                'overall_score': 0,
                'recommendation': '⚠️ Assessment failed'
            }
    
    def _score_price(self, deal: Dict) -> float:
        """Score deal based on price (0-100).

        When ``ai_category`` is present the score is calibrated against the
        category-specific margin threshold from
        :attr:`CATEGORY_MARGIN_THRESHOLDS`.  For unknown categories the flat
        price-bracket scoring is used as a fallback.
        """
        price = deal.get('price', 0)
        
        if price <= 0:
            return 50

        # Category-aware scoring when ai_category is available.
        ai_category = (deal.get('ai_category') or '').strip().lower()
        if ai_category:
            threshold = self.CATEGORY_MARGIN_THRESHOLDS.get(
                ai_category, self.DEFAULT_MARGIN_THRESHOLD
            )
            # Use sold_price_ref (from Feature 2) when available; otherwise fall
            # back to the flat bracket logic for this category.
            sold_price = deal.get('sold_price_ref')
            if sold_price and sold_price > 0:
                margin = (sold_price - price) / sold_price
                if margin >= threshold * 2:
                    return 95
                elif margin >= threshold:
                    return 80
                elif margin >= 0:
                    return 60
                else:
                    return 35
            # No sold_price_ref but category is known — use tighter flat scoring.
            if price < 50:
                return 95
            elif price < 100:
                return 85
            elif price < 250:
                return 75
            elif price < 500:
                return 65
            elif price < 1000:
                return 50
            else:
                return 40

        # Flat price-bracket fallback (original logic) when category is unknown.
        if price < 50:
            return 95
        elif price < 100:
            return 85
        elif price < 250:
            return 75
        elif price < 500:
            return 65
        elif price < 1000:
            return 50
        else:
            return 40
    
    def _score_seller(self, deal: Dict) -> float:
        """Score based on seller rating (0-100)"""
        seller_rating = deal.get('seller_rating', 0)
        
        # Seller rating is already a percentage
        if seller_rating >= 99:
            return 100
        elif seller_rating >= 98:
            return 90
        elif seller_rating >= 95:
            return 80
        elif seller_rating >= 90:
            return 70
        elif seller_rating >= 85:
            return 50
        else:
            return 30
    
    def _score_condition(self, deal: Dict) -> float:
        """Score based on item condition (0-100)"""
        condition = deal.get('condition', '').lower()
        
        if 'new' in condition:
            return 100
        elif 'like new' in condition or 'excellent' in condition:
            return 90
        elif 'good' in condition:
            return 75
        elif 'fair' in condition:
            return 50
        elif 'used' in condition:
            return 60
        elif 'refurbished' in condition:
            return 75
        else:
            return 50
    
    def _score_trends(self, deal: Dict) -> float:
        """Score based on market trends (0-100)"""
        views = deal.get('views', 0)
        watchers = deal.get('watchers', 0)
        sold_count = deal.get('sold_count', 0)
        is_trending = deal.get('is_trending', False)
        
        score = 50  # Base score
        
        # Add points for activity
        if views > 100:
            score += 10
        if watchers > 10:
            score += 15
        if sold_count > 5:
            score += 15
        if is_trending:
            score += 10
        
        return min(score, 100)
    
    def _get_recommendation(self, score: float) -> str:
        """Get text recommendation based on score"""
        if score >= 85:
            return '🔥 Excellent Deal'
        elif score >= 70:
            return '✅ Good Deal'
        elif score >= 50:
            return '👍 Fair Deal'
        elif score >= 30:
            return '⚠️ Below Average'
        else:
            return '❌ Poor Deal'
    
    def update_market_data(self, title: str, price: float):
        """Update market data for price comparison"""
        if title not in self.market_data:
            self.market_data[title] = []
        
        self.market_data[title].append(price)
        
        # Keep only last 100 prices to save memory
        if len(self.market_data[title]) > 100:
            self.market_data[title] = self.market_data[title][-100:]
    
    def get_market_average(self, title: str) -> float:
        """Get average price for similar items"""
        if title in self.market_data and self.market_data[title]:
            return statistics.mean(self.market_data[title])
        return 0
