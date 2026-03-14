#!/usr/bin/env python3
"""
Flask REST API for eBay Deal Scraper
Provides endpoints for searching, history, export, stats and health checks
"""

import logging
import os
from flask import Flask, request, jsonify, render_template, Response

from scraper import EbayScraper
from deal_assessor import DealAssessor
import database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

scraper = EbayScraper()
assessor = DealAssessor()

database.init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Request body must be valid JSON with Content-Type: application/json'}), 400
    query = data.get('query', '').strip()
    try:
        max_results = max(1, min(int(data.get('max_results', 50)), 200))
    except (TypeError, ValueError):
        return jsonify({'error': 'max_results must be a positive integer'}), 400

    if not query:
        return jsonify({'error': 'query is required'}), 400

    deals, search_errors = scraper.search(query, max_results=max_results)
    logger.info("Search for %r returned %d deals, %d error(s)", query, len(deals), len(search_errors))

    assessed = []
    for deal in deals:
        assessment = assessor.assess_deal(deal)
        assessed.append({**deal, **assessment})

    database.save_search(query, assessed)

    return jsonify({
        'query': query,
        'deal_count': len(assessed),
        'deals': assessed,
        'errors': search_errors,
    })


@app.route('/api/history')
def history():
    limit = int(request.args.get('limit', 20))
    return jsonify(database.get_history(limit))


@app.route('/api/deals/<int:search_id>')
def deals(search_id):
    return jsonify(database.get_deals_by_search(search_id))


@app.route('/api/export')
def export():
    csv_data = database.export_csv()
    return Response(
        csv_data,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=ebay_deals.csv'},
    )


@app.route('/api/stats')
def stats():
    return jsonify(database.get_stats())


@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(host=host, port=port, debug=debug)
