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
from ebay_api_client import EbayApiClient
from deal_assessor import DealAssessor
from gemini_assessor import GeminiAssessor, _detect_sports_kinect_deal
import database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

scraper = EbayScraper()
ebay_api = EbayApiClient()
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

# ── Data source helpers ────────────────────────────────────────────────────

_VALID_DATA_SOURCES = {"auto", "api", "scraper"}


def _db_data_source() -> str:
    """Read the active data source from the database.

    Falls back to the DATA_SOURCE environment variable, then to "auto".
    """
    val = database.get_setting("data_source")
    if val and val in _VALID_DATA_SOURCES:
        return val
    env_val = os.environ.get("DATA_SOURCE", "auto").strip().lower()
    return env_val if env_val in _VALID_DATA_SOURCES else "auto"


def _db_germany_only() -> bool:
    """Germany-only location filter is always enabled.

    All searches and results use Germany (EBAY_DE) exclusively.
    """
    return True


def _resolve_engine(source: str):
    """Return the search callable and a label for the given *source* setting.

    ``source`` is one of ``"auto"``, ``"api"``, or ``"scraper"``.
    Returns ``(callable, label)`` where *callable* matches the
    ``search(query, max_results)`` signature of both engines.
    """
    if source == "api":
        if not ebay_api.is_configured:
            logger.warning(
                "data_source='api' but eBay API credentials are not set; "
                "falling back to scraper."
            )
            return scraper.search, "scraper"
        return ebay_api.search, "api"

    if source == "scraper":
        return scraper.search, "scraper"

    # "auto": prefer API when credentials are present.
    if ebay_api.is_configured:
        return ebay_api.search, "api"
    return scraper.search, "scraper"


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


def _is_german_location(location: str) -> bool:
    """Return True when *location* is in Germany or is empty/unknown.

    Items with no location data are considered potentially German (to avoid
    silently dropping valid results when the ``item_location`` field is
    unavailable, e.g. from the legacy scraper on listings that don't expose
    location).  Items with an explicit non-German location are filtered out.

    Matching rules (case-insensitive):
    - Empty string / None → keep (unknown origin, benefit of the doubt)
    - Ends with ``, DE`` (e.g. ``"Berlin, DE"``) → Germany
    - Equals ``DE`` exactly → Germany
    - Contains the word ``Deutschland`` → Germany
    - Contains the word ``Germany`` → Germany
    """
    if not location:
        return True
    upper = location.strip().upper()
    # Exact country code
    if upper == "DE":
        return True
    # "City, DE" format from the eBay Browse API
    if upper.endswith(", DE"):
        return True
    # German or English country names as whole words
    if "DEUTSCHLAND" in upper or "GERMANY" in upper:
        return True
    return False


# Keywords (lower-case) indicating a multi-item lot / bundle listing.
# Checked against the deal title for bundle-first sort priority.
_BUNDLE_KEYWORDS = frozenset({
    # German
    "sammlung", "konvolut", "paket", "lot", "bundle",
    "spielesammlung", "spielepaket", "spielekonvolut",
    # Common numeric-quantity patterns are handled separately below.
})

_BUNDLE_NUMBER_RE = re.compile(
    r'\b\d+\s*[xX×]\s*\w+|\b\d+\s*(?:spiele|games|stück|pieces|items)\b',
    re.IGNORECASE,
)


def _is_bundle(deal: dict) -> bool:
    """Return True when the deal title suggests a multi-item lot or bundle."""
    title = (deal.get("title") or "").lower()
    for kw in _BUNDLE_KEYWORDS:
        if kw in title:
            return True
    if _BUNDLE_NUMBER_RE.search(title):
        return True
    return False


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

    try:
        return _run_search(query, max_results)
    except Exception as exc:
        logger.exception("Unhandled error in search route for query %r: %s", query, exc)
        return jsonify({'error': f'Internal server error: {exc}', 'deals': [], 'deal_count': 0}), 500


def _run_search(query: str, max_results: int):
    data_source_setting = _db_data_source()
    search_fn, active_source = _resolve_engine(data_source_setting)

    deals, search_errors = search_fn(query, max_results=max_results)
    logger.info(
        "Search for %r via %s returned %d deals, %d error(s)",
        query, active_source, len(deals), len(search_errors),
    )

    # Post-filter: exclude deals that the user has previously skipped.
    skipped_urls = set(database.get_skipped_deal_urls())
    if skipped_urls:
        before_skip = len(deals)
        deals = [d for d in deals if d.get("url") not in skipped_urls]
        filtered_skip = before_skip - len(deals)
        if filtered_skip:
            logger.info("Skip filter removed %d previously-skipped deal(s)", filtered_skip)

    # Post-filter: drop any deal whose item_location is not Germany (DE).
    # This is a safety net in addition to the API/scraper-level filters
    # (itemLocationCountry and LH_ItemLocation) and is controlled by the
    # germany_only setting.  Items with no location data are kept to avoid
    # silently dropping valid results when the location field is unavailable.
    germany_only = _db_germany_only()
    if germany_only:
        before = len(deals)
        deals = [d for d in deals if _is_german_location(d.get("item_location", ""))]
        filtered_out = before - len(deals)
        if filtered_out:
            logger.info(
                "Germany-only filter removed %d non-German deal(s)", filtered_out
            )

    # Post-filter: drop sports/Kinect-themed deals — these have very low
    # resale value (FIFA, Forza, Kinect, TopSpin, etc.) and should never
    # surface as desirable results.
    before_sports = len(deals)
    deals = [d for d in deals if not _detect_sports_kinect_deal(d)]
    filtered_sports = before_sports - len(deals)
    if filtered_sports:
        logger.info(
            "Sports/Kinect filter removed %d deal(s) with low resale value",
            filtered_sports,
        )

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
    try:
        ai_assessments = gemini.assess_deals_batch(deals_filtered) if (deals_filtered and ai_active) else []
    except Exception as exc:
        logger.error(
            "assess_deals_batch raised unexpectedly (model=%r): %s — falling back to rules engine.",
            gemini.model_name, exc,
        )
        ai_assessments = []

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

    # Sort deals: bundles first, then by AI rating and score.
    # Within each group (bundle / single), AI-rated "Must Buy" with high
    # confidence leads, followed by "Fair", then "Avoid", then rules-only
    # results ordered by overall_score.
    _rating_order = {"must buy": 0, "fair": 1, "avoid": 2}

    def _sort_key(d: dict):
        is_single = not _is_bundle(d)  # 0 = bundle, 1 = single game
        ai_order = _rating_order.get((d.get("ai_deal_rating") or "").lower(), 3)
        ai_conf = -(d.get("ai_confidence_score") or 0)
        score = -(d.get("overall_score") or 0)
        return (is_single, not d.get("ai_assessed", False), ai_order, ai_conf, score)

    assessed.sort(key=_sort_key)

    # Persist the search — non-fatal: a DB error must not prevent the results
    # from being returned to the user.
    try:
        database.save_search(query, assessed)
    except Exception as exc:
        logger.error("Failed to persist search results to DB: %s", exc)

    # Compute how many seconds remain in any rate-limit back-off window.
    paused_seconds = max(0.0, gemini.rate_limited_until - time.monotonic())

    # Annotate each deal with whether the user has already saved it.
    try:
        saved_urls = set(d['url'] for d in database.get_saved_deals())
    except Exception as exc:
        logger.error("Failed to load saved deals from DB: %s", exc)
        saved_urls = set()
    for deal in assessed:
        deal['is_saved'] = deal.get('url') in saved_urls

    return jsonify({
        'query': query,
        'deal_count': len(assessed),
        'deals': assessed,
        'errors': search_errors,
        'ai_enabled': gemini.enabled and _user_enabled,
        'ai_rate_limited': gemini.is_rate_limited,
        'ai_paused_seconds': round(paused_seconds),
        'data_source': active_source,
        'germany_only': germany_only,
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
    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    return jsonify({
        'status': 'healthy',
        'ai_enabled': gemini.enabled and _db_ai_user_enabled(),
        'ai_rate_limited': gemini.is_rate_limited,
        'ai_paused_seconds': round(paused_seconds),
        'ai_model': gemini.model_name,
        'data_source': active_source,
        'data_source_setting': data_source_setting,
        'ebay_api_configured': ebay_api.is_configured,
        'ebay_marketplace_id': ebay_api.marketplace_id,
        'ebay_language': ebay_api.accept_language,
        'ebay_locale': ebay_api.locale,
        'ebay_delivery_country': ebay_api.delivery_country,
        'germany_only': _db_germany_only(),
    })


@app.route('/api/settings', methods=['GET'])
def get_settings():
    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    return jsonify({
        'gemini_model': gemini.model_name,
        'ai_enabled': _db_ai_user_enabled(),
        'data_source': data_source_setting,
        'active_data_source': active_source,
        'ebay_api_configured': ebay_api.is_configured,
        'ebay_marketplace_id': ebay_api.marketplace_id,
        'ebay_language': ebay_api.accept_language,
        'ebay_locale': ebay_api.locale,
        'ebay_delivery_country': ebay_api.delivery_country,
        'germany_only': _db_germany_only(),
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

    if 'data_source' in data:
        ds = str(data['data_source']).strip().lower()
        if ds not in _VALID_DATA_SOURCES:
            errors['data_source'] = (
                f"data_source must be one of: {', '.join(sorted(_VALID_DATA_SOURCES))}"
            )
        else:
            database.set_setting('data_source', ds)
            updated['data_source'] = ds
            logger.info("Settings: data_source updated to %r", ds)

    if errors:
        return jsonify({'errors': errors}), 400

    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    return jsonify({
        'updated': updated,
        'gemini_model': gemini.model_name,
        'ai_enabled': gemini.user_enabled,
        'data_source': data_source_setting,
        'active_data_source': active_source,
        'ebay_api_configured': ebay_api.is_configured,
        'ebay_marketplace_id': ebay_api.marketplace_id,
        'ebay_language': ebay_api.accept_language,
        'ebay_locale': ebay_api.locale,
        'ebay_delivery_country': ebay_api.delivery_country,
        'germany_only': _db_germany_only(),
    })


# ── Save / Skip deal endpoints ────────────────────────────────────────────────

# Maximum character length accepted for deal title strings in API requests.
_MAX_TITLE_LENGTH = 500


@app.route('/api/deals/save', methods=['POST'])
def deal_save():
    """Save (favourite) a deal by URL."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'url is required'}), 400
    title = str(data.get('title') or '')[:_MAX_TITLE_LENGTH]
    try:
        price = float(data.get('price') or 0)
    except (TypeError, ValueError):
        price = 0.0
    database.save_deal(url, title, price)
    return jsonify({'saved': True, 'url': url})


@app.route('/api/deals/unsave', methods=['POST'])
def deal_unsave():
    """Remove a deal from the saved list."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'url is required'}), 400
    database.unsave_deal(url)
    return jsonify({'saved': False, 'url': url})


@app.route('/api/deals/saved', methods=['GET'])
def deal_saved_list():
    """Return all saved deals."""
    return jsonify(database.get_saved_deals())


@app.route('/api/deals/skip', methods=['POST'])
def deal_skip():
    """Skip (hide) a deal so it is excluded from future search results."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'url is required'}), 400
    title = str(data.get('title') or '')[:_MAX_TITLE_LENGTH]
    try:
        price = float(data.get('price') or 0)
    except (TypeError, ValueError):
        price = 0.0
    database.skip_deal(url, title, price)
    return jsonify({'skipped': True, 'url': url})


@app.route('/api/deals/unskip', methods=['POST'])
def deal_unskip():
    """Remove a deal from the skipped list."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be valid JSON'}), 400
    url = (data.get('url') or '').strip()
    if not url:
        return jsonify({'error': 'url is required'}), 400
    database.unskip_deal(url)
    return jsonify({'skipped': False, 'url': url})


@app.route('/api/deals/skipped', methods=['GET'])
def deal_skipped_list():
    """Return all skipped deals with full metadata."""
    return jsonify(database.get_skipped_deals())


if __name__ == '__main__':
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    app.run(host=host, port=port, debug=debug)
