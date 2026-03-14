#!/usr/bin/env python3
"""
eBay Web Scraper Engine
Handles fetching and parsing eBay listings
"""

import logging
import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, List, Tuple
import random

logger = logging.getLogger(__name__)

class EbayScraper:
    def __init__(self):
        self.base_url = "https://www.ebay.com/sch/i.html"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()

    def search(self, query: str, max_results: int = 50) -> Tuple[List[Dict], List[str]]:
        """Search eBay for items matching query.

        Returns a tuple of (deals, errors) where errors is a list of
        human-readable strings describing any problems encountered.
        """
        errors: List[str] = []

        try:
            params = {
                '_nkw': query,
                '_sop': '12',  # Sort by newly listed
                'LH_ItemCondition': '3000|3000|1000',  # All conditions
                'rt': 'nc'
            }

            logger.info("Searching eBay for %r (max_results=%d)", query, max_results)

            try:
                response = self.session.get(
                    self.base_url, params=params, headers=self.headers, timeout=10
                )
            except requests.exceptions.Timeout:
                msg = "HTTP request timed out after 10 seconds"
                logger.error(msg)
                errors.append(msg)
                return [], errors
            except requests.exceptions.ConnectionError as exc:
                msg = f"Connection error: {exc}"
                logger.error(msg)
                errors.append(msg)
                return [], errors

            logger.info("HTTP %d %s", response.status_code, response.reason)

            if not response.ok:
                msg = f"HTTP {response.status_code}: {response.reason}"
                logger.error("eBay returned an error – %s", msg)
                errors.append(msg)
                if response.status_code == 403:
                    errors.append(
                        "Access denied – eBay may be blocking automated requests. "
                        "Try again later or use a different network."
                    )
                elif response.status_code == 429:
                    errors.append(
                        "Rate limited – too many requests sent in a short time. "
                        "Wait a few minutes before searching again."
                    )
                return [], errors

            soup = BeautifulSoup(response.content, 'html.parser')
            deals: List[Dict] = []

            # Find all item listings
            items = soup.find_all('div', {'class': 's-item'})
            logger.info("BeautifulSoup found %d raw item elements", len(items))

            if not items:
                msg = (
                    "BeautifulSoup found 0 item elements with class 's-item'. "
                    "eBay may have changed their page structure."
                )
                logger.warning(msg)
                errors.append(msg)

            parse_errors = 0
            for item in items[:max_results]:
                try:
                    deal = self._parse_item(item)
                    if deal:
                        deals.append(deal)
                except Exception as exc:
                    parse_errors += 1
                    logger.warning("Error parsing item element: %s", exc, exc_info=True)
                    continue

            if parse_errors:
                errors.append(f"{parse_errors} item(s) could not be parsed and were skipped.")

            logger.info("Returning %d deals (%d errors)", len(deals), len(errors))
            time.sleep(random.uniform(1, 3))  # Rate limiting
            return deals, errors

        except Exception as exc:
            msg = f"Unexpected error during search: {exc}"
            logger.error(msg, exc_info=True)
            errors.append(msg)
            return [], errors

    def _parse_item(self, item_element) -> Dict:
        """Parse individual item element into deal dictionary"""
        try:
            # Extract title
            title_elem = item_element.find('h2', {'class': 's-item__title'})
            title = title_elem.text.strip() if title_elem else "Unknown"

            # Extract price
            price_elem = item_element.find('span', {'class': 's-item__price'})
            price_text = price_elem.text.strip() if price_elem else "$0.00"
            price = self._parse_price(price_text)

            # Extract condition
            condition_elem = item_element.find('span', {'class': 'SECONDARY_INFO'})
            condition = condition_elem.text.strip() if condition_elem else "Unknown"

            # Extract seller rating
            seller_elem = item_element.find('span', {'class': 's-item__seller-info-text'})
            seller_rating = self._parse_seller_rating(seller_elem.text) if seller_elem else 0

            # Extract item URL
            link_elem = item_element.find('a', {'class': 's-item__link'})
            item_url = link_elem.get('href', '') if link_elem else ""

            # Extract shipping info
            shipping_elem = item_element.find('span', {'class': 's-item__shipping'})
            shipping = shipping_elem.text.strip() if shipping_elem else "Calculate"

            # Check if new/trending
            is_trending = bool(item_element.find('span', {'class': 'SHOP_NEW_TAG'}))

            return {
                'title': title,
                'price': price,
                'condition': condition,
                'seller_rating': seller_rating,
                'url': item_url,
                'shipping': shipping,
                'is_trending': is_trending,
                'timestamp': time.time()
            }

        except Exception as exc:
            logger.warning("Error in _parse_item: %s", exc, exc_info=True)
            return None

    def _parse_price(self, price_str: str) -> float:
        """Extract numeric price from string"""
        try:
            # Remove currency symbols and text
            clean = price_str.replace('$', '').replace(',', '').split()[0]
            return float(clean)
        except Exception:
            return 0.0

    def _parse_seller_rating(self, seller_str: str) -> float:
        """Extract seller rating percentage"""
        try:
            # Extract percentage from seller info
            if '%' in seller_str:
                rating_str = seller_str.split()[0].replace('(', '').replace(')', '')
                return float(rating_str.replace('%', ''))
            return 0.0
        except Exception:
            return 0.0

    def get_item_details(self, item_url: str) -> Dict:
        """Fetch detailed information about specific item"""
        try:
            response = self.session.get(item_url, headers=self.headers, timeout=10)
            logger.info("get_item_details HTTP %d %s", response.status_code, response.reason)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            details = {
                'views': self._extract_views(soup),
                'watchers': self._extract_watchers(soup),
                'sold_count': self._extract_sold_count(soup),
                'time_listed': self._extract_time_listed(soup)
            }

            time.sleep(random.uniform(1, 2))
            return details

        except Exception as exc:
            logger.error("Error getting item details: %s", exc, exc_info=True)
            return {}
    
    def _extract_views(self, soup) -> int:
        """Extract view count from item page"""
        try:
            views_elem = soup.find('span', string=lambda s: s and 'views' in s.lower())
            if views_elem:
                count = views_elem.text.split()[0].replace(',', '')
                return int(count)
        except:
            pass
        return 0
    
    def _extract_watchers(self, soup) -> int:
        """Extract watcher count from item page"""
        try:
            watchers_elem = soup.find('span', string=lambda s: s and 'watchers' in s.lower())
            if watchers_elem:
                count = watchers_elem.text.split()[0].replace(',', '')
                return int(count)
        except:
            pass
        return 0
    
    def _extract_sold_count(self, soup) -> int:
        """Extract sold count from item page"""
        try:
            sold_elem = soup.find('span', string=lambda s: s and 'sold' in s.lower())
            if sold_elem:
                count = sold_elem.text.split()[0].replace(',', '')
                return int(count)
        except:
            pass
        return 0
    
    def _extract_time_listed(self, soup) -> str:
        """Extract when item was listed"""
        try:
            time_elem = soup.find('span', string=lambda s: s and 'listed' in s.lower())
            if time_elem:
                return time_elem.text.strip()
        except:
            pass
        return "Unknown"