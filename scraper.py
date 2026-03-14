#!/usr/bin/env python3
"""
eBay Web Scraper Engine
Handles fetching and parsing eBay listings
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict
import random

class EbayScraper:
    def __init__(self):
        self.base_url = "https://www.ebay.com/sch/i.html"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
        
    def search(self, query: str, max_results: int = 50) -> List[Dict]:
        """Search eBay for items matching query"""
        try:
            params = {
                '_nkw': query,
                '_sop': '12',  # Sort by newly listed
                'LH_ItemCondition': '3000|3000|1000',  # All conditions
                'rt': 'nc'
            }
            
            response = self.session.get(self.base_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            deals = []
            
            # Find all item listings
            items = soup.find_all('div', {'class': 's-item'})
            
            for item in items[:max_results]:
                try:
                    deal = self._parse_item(item)
                    if deal:
                        deals.append(deal)
                except Exception as e:
                    print(f"Error parsing item: {e}")
                    continue
            
            time.sleep(random.uniform(1, 3))  # Rate limiting
            return deals
            
        except Exception as e:
            print(f"Error searching eBay: {e}")
            return []
    
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
            
        except Exception as e:
            print(f"Error in _parse_item: {e}")
            return None
    
    def _parse_price(self, price_str: str) -> float:
        """Extract numeric price from string"""
        try:
            # Remove currency symbols and text
            clean = price_str.replace('$', '').replace(',', '').split()[0]
            return float(clean)
        except:
            return 0.0
    
    def _parse_seller_rating(self, seller_str: str) -> float:
        """Extract seller rating percentage"""
        try:
            # Extract percentage from seller info
            if '%' in seller_str:
                rating_str = seller_str.split()[0].replace('(', '').replace(')', '')
                return float(rating_str.replace('%', ''))
            return 0.0
        except:
            return 0.0
    
    def get_item_details(self, item_url: str) -> Dict:
        """Fetch detailed information about specific item"""
        try:
            response = self.session.get(item_url, headers=self.headers, timeout=10)
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
            
        except Exception as e:
            print(f"Error getting item details: {e}")
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