#!/usr/bin/env python3
"""
Flask REST API for eBay Deal Scraper
Provides endpoints for searching, history, export, stats and health checks
"""

import logging
import os
import re
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

# Load persisted Gemini model (if any) so it takes effect without a restart.
_saved_model = database.get_setting("gemini_model")
if _saved_model:
    gemini.model_name = _saved_model

# Load persisted AI-enabled toggle (default: True; stored as "true"/"false" string).
_saved_ai_enabled = database.get_setting("ai_enabled")
if _saved_ai_enabled is not None:
    gemini.user_enabled = str(_saved_ai_enabled).lower() == "true"


def _db_ai_user_enabled() -> bool:
    """Read the user's AI-enabled toggle from the database.

    Always reads from the shared SQLite database rather than the in-memory
    ``gemini.user_enabled`` attribute so that multi-worker (Gunicorn)
    deployments remain consistent: updating the setting in one worker is
    immediately visible to all other workers on the next request.

    Defaults to ``True`` when no setting has been persisted yet.
    """
    val = database.get_setting("ai_enabled")
    return str(val).lower() == "true" if val is not None else True


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

    # Filter to the top 20 decent bundle deals before sending to Gemini:
    # sort by rules score descending, drop low-quality entries, cap at 20.
    _DECENT_SCORE_MIN = 50
    _MAX_DISPLAY = 20
    _pairs = sorted(
        zip(deals, rules_assessments),
        key=lambda t: -(t[1].get('overall_score') or 0),
    )
    _decent = [(d, r) for d, r in _pairs if (r.get('overall_score') or 0) >= _DECENT_SCORE_MIN]
    if not _decent:
        _decent = list(_pairs[:_MAX_DISPLAY])
    _decent = _decent[:_MAX_DISPLAY]
    deals_filtered = [d for d, _ in _decent]
    rules_filtered = [r for _, r in _decent]

    # AI assessment via Gemini: send only the top filtered deals in a single
    # request to minimise quota consumption rather than calling once per deal.
    # Skip entirely when the user has disabled AI evaluation via the toggle.
    # Re-read ai_enabled from the database on every request so that the toggle
    # is respected in multi-worker (Gunicorn) deployments where in-memory state
    # is not shared across processes.
    _user_enabled = _db_ai_user_enabled()
    ai_active = gemini.enabled and _user_enabled
    ai_assessments = gemini.assess_deals_batch(deals_filtered) if (deals_filtered and ai_active) else []

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
    for i, deal in enumerate(deals_filtered):
        rules_assessment = rules_filtered[i]
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
        'ai_enabled': gemini.enabled and _user_enabled,
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
        'ai_enabled': gemini.enabled and _db_ai_user_enabled(),
        'ai_rate_limited': gemini.is_rate_limited,
        'ai_paused_seconds': round(paused_seconds),
        'ai_model': gemini.model_name,
    })


@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify({
        'gemini_model': gemini.model_name,
        'ai_enabled': _db_ai_user_enabled(),
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Request body must be valid JSON with Content-Type: application/json'}), 400

    # Gemini model names: alphanumeric, hyphens, underscores, and dots only.
    _MODEL_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_\-.]{0,99}$')

    errors = {}
    updated = {}

    if 'gemini_model' in data:
        model = str(data['gemini_model']).strip()
        if not model:
            errors['gemini_model'] = 'gemini_model must not be empty (e.g., gemini-2.0-flash-lite)'
        elif not _MODEL_NAME_RE.match(model):
            errors['gemini_model'] = (
                'gemini_model contains invalid characters; use only letters, '
                'digits, hyphens, underscores, and dots (e.g., gemini-2.0-flash-lite)'
            )
        else:
            try:
                gemini.model_name = model
                database.set_setting('gemini_model', model)
                updated['gemini_model'] = model
                logger.info("Settings: gemini_model updated to %r", model)
            except ValueError as exc:
                errors['gemini_model'] = str(exc)

    if 'ai_enabled' in data:
        ai_enabled = data['ai_enabled']
        if not isinstance(ai_enabled, bool):
            errors['ai_enabled'] = 'ai_enabled must be a boolean (true or false)'
        else:
            gemini.user_enabled = ai_enabled
            database.set_setting('ai_enabled', str(ai_enabled).lower())
            updated['ai_enabled'] = ai_enabled
            logger.info("Settings: ai_enabled updated to %r", ai_enabled)

    if errors:
        return jsonify({'errors': errors}), 400

    return jsonify({'updated': updated, 'gemini_model': gemini.model_name, 'ai_enabled': gemini.user_enabled})


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(host=host, port=port, debug=debug)
