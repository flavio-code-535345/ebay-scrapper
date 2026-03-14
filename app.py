#!/usr/bin/env python3
"""
Flask REST API for eBay Deal Scraper
Provides endpoints for searching, history, export, stats and health checks
"""

import logging
import os
import time
from flask import Flask, request, jsonify, render_template, Response

from scraper import EbayScraper
from deal_assessor import DealAssessor
from gemini_assessor import GeminiAssessor
import database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

scraper = EbayScraper()
assessor = DealAssessor()
gemini = GeminiAssessor()

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

    # Rules-based assessment (always available as baseline/fallback).
    rules_assessments = [assessor.assess_deal(deal) for deal in deals]

    # AI assessment via Gemini: send all deals in a single (or few) request(s)
    # to minimise quota consumption rather than calling once per deal.
    ai_assessments = gemini.assess_deals_batch(deals) if deals else []

    if gemini.enabled and ai_assessments:
        failed = sum(1 for a in ai_assessments if a is None)
        rate_limited = sum(
            1 for a in ai_assessments if a and a.get("ai_error_type") == "rate_limit"
        )
        parse_errors = sum(
            1 for a in ai_assessments if a and a.get("ai_error_type") == "parse_error"
        )
        if failed:
            logger.warning(
                "Gemini batch: %d/%d items failed AI assessment; using rules engine.",
                failed,
                len(ai_assessments),
            )
        if rate_limited:
            logger.warning(
                "Gemini batch: %d/%d items rate-limited; skipping AI assessment.",
                rate_limited,
                len(ai_assessments),
            )
        if parse_errors:
            logger.warning(
                "Gemini batch: %d/%d items had parse errors; AI fields set to defaults.",
                parse_errors,
                len(ai_assessments),
            )

    assessed = []
    for i, deal in enumerate(deals):
        rules_assessment = rules_assessments[i]
        ai_assessment = ai_assessments[i] if i < len(ai_assessments) else None
        assessed.append({**deal, **rules_assessment, **(ai_assessment or {})})

    # Sort deals: best first. AI-rated "Must Buy" with high confidence leads,
    # followed by "Fair", then "Avoid", then rules-only results by overall_score.
    _rating_order = {"must buy": 0, "fair": 1, "avoid": 2}

    def _sort_key(d: dict):
        ai_order = _rating_order.get((d.get("ai_deal_rating") or "").lower(), 3)
        ai_conf = -(d.get("ai_confidence_score") or 0)
        score = -(d.get("overall_score") or 0)
        return (not d.get("ai_assessed", False), ai_order, ai_conf, score)

    assessed.sort(key=_sort_key)

    database.save_search(query, assessed)

    # Compute how many seconds remain in any rate-limit back-off window.
    paused_seconds = max(0.0, gemini.rate_limited_until - time.monotonic())

    return jsonify({
        'query': query,
        'deal_count': len(assessed),
        'deals': assessed,
        'errors': search_errors,
        'ai_enabled': gemini.enabled,
        'ai_rate_limited': gemini.is_rate_limited,
        'ai_paused_seconds': round(paused_seconds),
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
    paused_seconds = max(0.0, gemini.rate_limited_until - time.monotonic())
    return jsonify({
        'status': 'healthy',
        'ai_enabled': gemini.enabled,
        'ai_rate_limited': gemini.is_rate_limited,
        'ai_paused_seconds': round(paused_seconds),
    })


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(host=host, port=port, debug=debug)
