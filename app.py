#!/usr/bin/env python3
"""
Flask REST API for eBay Deal Scraper
Provides endpoints for searching, history, export, stats and health checks
"""

import json
import logging
import os
import re
import time

from flask import Flask, Response, jsonify, render_template, request

import database
from ai_providers import create_assessor
from ebay_api_client import _MARKETPLACE_LOCALE_MAP, EbayApiClient
from models import canonical_listing_id, sort_key_for_deal
from scraper import EbayScraper
from search.pipeline import run_search
from search.query import plan_search

try:
    from kleinanzeigen_scraper import KleinanzeigenScraper
except ImportError:
    KleinanzeigenScraper = None  # type: ignore[assignment,misc]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


_LOG_FORMAT_ENV = os.environ.get("LOG_FORMAT", "plain").strip().lower()
if _LOG_FORMAT_ENV == "json":
    _handler = logging.StreamHandler()
    _handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[_handler])
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
logger = logging.getLogger(__name__)

app = Flask(__name__)

scraper = EbayScraper()
ebay_api = EbayApiClient()
kleinanzeigen = KleinanzeigenScraper() if KleinanzeigenScraper else None

database.init_db()

assessor = create_assessor()
assessor.set_ebay_client(ebay_api)

_saved_model = database.get_setting("gemini_model")
if _saved_model:
    assessor.model_name = _saved_model

_saved_ai_enabled = database.get_setting("ai_enabled")
if _saved_ai_enabled is not None:
    assessor.user_enabled = str(_saved_ai_enabled).lower() == "true"

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
            logger.warning("data_source='api' but eBay API credentials are not set; falling back to scraper.")
            return scraper.search, "scraper"
        return ebay_api.search, "api"

    if source == "scraper":
        return scraper.search, "scraper"

    # "auto": prefer API when credentials are present.
    if ebay_api.is_configured:
        return ebay_api.search, "api"
    return scraper.search, "scraper"


def _db_ai_user_enabled() -> bool:
    val = database.get_setting("ai_enabled")
    return str(val).lower() == "true" if val is not None else True


@app.route("/")
def index():
    return render_template("index.html")


# Total wall-clock budget for the WHOLE /api/search request (search phase +
# Gemini AI assessment), measured from the moment the request starts. Many
# reverse proxies / edge networks kill a still-in-progress request after a
# fixed idle timeout regardless of what gunicorn's own (much longer) worker
# timeout allows — Cloudflare's proxied HTTP default is 100s, for example,
# and returns its own HTML error page (which the frontend then can't parse
# as JSON) rather than anything from this app. 75s leaves real margin under
# that for response serialization/transmission. `assess_deals_batch` is
# handed the remaining time as a deadline, so a slow search phase leaves
# correspondingly less time for AI scoring instead of the two stacking on
# top of each other — degrading gracefully (fewer/no AI ratings, but a real
# response) instead of the proxy cutting the connection with nothing.
# Override via SEARCH_DEADLINE_SECONDS for deployments without such a
# proxy in front (or with a longer one) that want the fuller AI budget.
_SEARCH_DEADLINE_S = int(os.environ.get("SEARCH_DEADLINE_SECONDS", "75"))

# The search phase gets at most this long (and never more than 45% of the
# whole deadline), so a slow or stalled source can't eat the AI's time — its
# results are dropped and reported instead.
_SEARCH_PHASE_MAX_S = 35.0

# Cap on how many deals get sent to Gemini for AI assessment per search.
_MAX_DISPLAY = 30

# Gemini model names: alphanumeric, hyphens, underscores, and dots only.
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,99}$")


def _skipped_keys() -> set[str]:
    """Skipped URLs plus their stable listing IDs, so a skipped listing stays
    hidden even when a source reports it under a differently-shaped URL."""
    urls = database.get_skipped_deal_urls()
    return set(urls) | {key for key in map(canonical_listing_id, urls) if key}


@app.route("/api/search", methods=["POST"])
def search():
    _request_start = time.monotonic()
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    # Accept a single "query" or a "queries" array (the quick-search chips
    # send several phrasings; the planner folds them into one plan).
    raw_query = str(data.get("query") or "").strip()
    raw_queries = data.get("queries")
    if raw_queries and isinstance(raw_queries, list):
        phrases = [str(q).strip() for q in raw_queries if str(q).strip()]
    else:
        phrases = [raw_query] if raw_query else []
    if not phrases:
        return jsonify({"error": "query or queries is required"}), 400

    plan = plan_search(phrases)
    query = plan.label

    data_source_setting = _db_data_source()
    search_fn, active_source = _resolve_engine(data_source_setting)
    budget_s = min(_SEARCH_PHASE_MAX_S, 0.45 * _SEARCH_DEADLINE_S)
    outcome = run_search(
        plan,
        ebay_search=search_fn,
        ebay_is_api=active_source == "api",
        auction_search=ebay_api.search_auctions if active_source == "api" else scraper.search_auctions,
        kleinanzeigen_search=kleinanzeigen.search if kleinanzeigen else None,
        skipped=_skipped_keys(),
        budget_s=budget_s,
        limit=_MAX_DISPLAY,
    )
    search_errors = outcome.errors
    germany_only = _db_germany_only()
    deals_filtered = outcome.selected
    ebay_n = sum(1 for d in deals_filtered if d.get("source") == "ebay")
    kdx_n = sum(1 for d in deals_filtered if d.get("source") == "kleinanzeigen")

    # AI assessment via Gemini: send only the top filtered deals in a single
    # request to minimise quota consumption rather than calling once per deal.
    # Skip entirely when the user has disabled AI evaluation via the toggle.
    # Re-read ai_enabled from the database on every request so that the toggle
    # is respected in multi-worker (Gunicorn) deployments where in-memory state
    # is not shared across processes.
    _user_enabled = _db_ai_user_enabled()
    ai_active = assessor.enabled and _user_enabled
    _deadline = _request_start + _SEARCH_DEADLINE_S
    ai_assessments = (
        assessor.assess_deals_batch(deals_filtered, deadline=_deadline) if (deals_filtered and ai_active) else []
    )

    timed_out = 0
    if assessor.enabled and ai_assessments:
        failed = sum(1 for a in ai_assessments if a is None)
        rate_limited = sum(1 for a in ai_assessments if a and a.get("ai_error_type") == "rate_limit")
        parse_errors = sum(1 for a in ai_assessments if a and a.get("ai_error_type") == "parse_error")
        timed_out = sum(1 for a in ai_assessments if a and a.get("ai_error_type") == "timeout")
        if failed:
            logger.warning(
                "Gemini batch: %d/%d items failed AI assessment.",
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
            for i, (deal, a) in enumerate(zip(deals_filtered, ai_assessments, strict=False)):
                if a and a.get("ai_error_type") == "parse_error":
                    logger.warning(
                        "Gemini parse error – item[%d]: %r",
                        i,
                        (deal.get("title") or "")[:80],
                    )
        if timed_out:
            logger.warning(
                "Gemini batch: %d/%d items timed out; AI assessment skipped.",
                timed_out,
                len(ai_assessments),
            )
            for i, (deal, a) in enumerate(zip(deals_filtered, ai_assessments, strict=False)):
                if a and a.get("ai_error_type") == "timeout":
                    logger.info(
                        "Gemini timeout – item[%d]: %r",
                        i,
                        (deal.get("title") or "")[:80],
                    )

    assessed = []
    for i, deal in enumerate(deals_filtered):
        ai_assessment = ai_assessments[i] if i < len(ai_assessments) else None
        assessed.append({**deal, **(ai_assessment or {})})

    # Sort deals: "Must Have"/"Must Buy" first, then all others — within each
    # group, deals with a known listing date sort newest → oldest, ahead of
    # deals with no known date at all (see models.sort_key_for_deal).
    assessed.sort(key=sort_key_for_deal)

    database.save_search(query, assessed)

    # Compute how many seconds remain in any rate-limit back-off window.
    paused_seconds = max(0.0, assessor.rate_limited_until - time.monotonic())

    saved_urls = set(d["url"] for d in database.get_saved_deals())
    for deal in assessed:
        deal["is_saved"] = deal.get("url") in saved_urls

    return jsonify(
        {
            "query": query,
            "deal_count": len(assessed),
            "deals": assessed,
            "errors": search_errors,
            "ebay_count": ebay_n,
            "kleinanzeigen_count": kdx_n,
            "ai_enabled": assessor.enabled and _user_enabled,
            "ai_rate_limited": assessor.is_rate_limited,
            "ai_paused_seconds": round(paused_seconds),
            "ai_timeout_count": timed_out,
            "data_source": active_source,
            "germany_only": germany_only,
            "candidate_count": outcome.candidates,
            "sources": [r.as_dict() for r in outcome.sources],
        }
    )


@app.route("/api/history")
def history():
    try:
        limit = max(1, min(int(request.args.get("limit", 20)), 200))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be a positive integer"}), 400
    return jsonify(database.get_history(limit))


@app.route("/api/deals/<int:search_id>")
def deals(search_id):
    return jsonify(database.get_deals_by_search(search_id))


@app.route("/api/export")
def export():
    csv_data = database.export_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=ebay_deals.csv"},
    )


@app.route("/api/stats")
def stats():
    return jsonify(database.get_stats())


@app.route("/api/health")
def health():
    paused_seconds = max(0.0, assessor.rate_limited_until - time.monotonic())
    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    return jsonify(
        {
            "status": "healthy",
            "ai_enabled": assessor.enabled and _db_ai_user_enabled(),
            "ai_rate_limited": assessor.is_rate_limited,
            "ai_paused_seconds": round(paused_seconds),
            "ai_model": assessor.model_name,
            "data_source": active_source,
            "data_source_setting": data_source_setting,
            "ebay_api_configured": ebay_api.is_configured,
            "ebay_marketplace_id": ebay_api.marketplace_id,
            "ebay_language": ebay_api.accept_language,
            "ebay_locale": ebay_api.locale,
            "ebay_delivery_country": ebay_api.delivery_country,
            "germany_only": _db_germany_only(),
            "available_marketplaces": sorted(_MARKETPLACE_LOCALE_MAP.keys()),
        }
    )


@app.route("/api/settings", methods=["GET"])
def get_settings():
    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    return jsonify(
        {
            "gemini_model": assessor.model_name,
            "ai_enabled": _db_ai_user_enabled(),
            "data_source": data_source_setting,
            "active_data_source": active_source,
            "ebay_api_configured": ebay_api.is_configured,
            "ebay_marketplace_id": ebay_api.marketplace_id,
            "ebay_language": ebay_api.accept_language,
            "ebay_locale": ebay_api.locale,
            "ebay_delivery_country": ebay_api.delivery_country,
            "germany_only": _db_germany_only(),
        }
    )


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    errors = {}
    updated = {}

    if "gemini_model" in data:
        model = str(data["gemini_model"]).strip()
        if not model:
            errors["gemini_model"] = "gemini_model must not be empty (e.g., gemini-2.0-flash-lite)"
        elif not _MODEL_NAME_RE.match(model):
            errors["gemini_model"] = (
                "gemini_model contains invalid characters; use only letters, "
                "digits, hyphens, underscores, and dots (e.g., gemini-2.0-flash-lite)"
            )
        else:
            try:
                assessor.model_name = model
                database.set_setting("gemini_model", model)
                updated["gemini_model"] = model
                logger.info("Settings: gemini_model updated to %r", model)
            except ValueError as exc:
                errors["gemini_model"] = str(exc)

    if "ai_enabled" in data:
        ai_enabled = data["ai_enabled"]
        if not isinstance(ai_enabled, bool):
            errors["ai_enabled"] = "ai_enabled must be a boolean (true or false)"
        else:
            assessor.user_enabled = ai_enabled
            database.set_setting("ai_enabled", str(ai_enabled).lower())
            updated["ai_enabled"] = ai_enabled
            logger.info("Settings: ai_enabled updated to %r", ai_enabled)

    if "data_source" in data:
        ds = str(data["data_source"]).strip().lower()
        if ds not in _VALID_DATA_SOURCES:
            errors["data_source"] = f"data_source must be one of: {', '.join(sorted(_VALID_DATA_SOURCES))}"
        else:
            database.set_setting("data_source", ds)
            updated["data_source"] = ds
            logger.info("Settings: data_source updated to %r", ds)

    if errors:
        return jsonify({"errors": errors}), 400

    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    return jsonify(
        {
            "updated": updated,
            "gemini_model": assessor.model_name,
            "ai_enabled": assessor.user_enabled,
            "data_source": data_source_setting,
            "active_data_source": active_source,
            "ebay_api_configured": ebay_api.is_configured,
            "ebay_marketplace_id": ebay_api.marketplace_id,
            "ebay_language": ebay_api.accept_language,
            "ebay_locale": ebay_api.locale,
            "ebay_delivery_country": ebay_api.delivery_country,
            "germany_only": _db_germany_only(),
        }
    )


# ── Save / Skip deal endpoints ────────────────────────────────────────────────

# Maximum character length accepted for deal title strings in API requests.
_MAX_TITLE_LENGTH = 500


@app.route("/api/deals/save", methods=["POST"])
def deal_save():
    """Save (favourite) a deal by URL."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    title = str(data.get("title") or "")[:_MAX_TITLE_LENGTH]
    try:
        price = float(data.get("price") or 0)
    except (TypeError, ValueError):
        price = 0.0
    database.save_deal(url, title, price)
    return jsonify({"saved": True, "url": url})


@app.route("/api/deals/unsave", methods=["POST"])
def deal_unsave():
    """Remove a deal from the saved list."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    database.unsave_deal(url)
    return jsonify({"saved": False, "url": url})


@app.route("/api/deals/saved", methods=["GET"])
def deal_saved_list():
    """Return all saved deals."""
    return jsonify(database.get_saved_deals())


@app.route("/api/deals/skip", methods=["POST"])
def deal_skip():
    """Skip (hide) a deal so it is excluded from future search results."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    title = str(data.get("title") or "")[:_MAX_TITLE_LENGTH]
    try:
        price = float(data.get("price") or 0)
    except (TypeError, ValueError):
        price = 0.0
    database.skip_deal(url, title, price)
    return jsonify({"skipped": True, "url": url})


@app.route("/api/deals/unskip", methods=["POST"])
def deal_unskip():
    """Remove a deal from the skipped list."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    database.unskip_deal(url)
    return jsonify({"skipped": False, "url": url})


@app.route("/api/deals/skipped", methods=["GET"])
def deal_skipped_list():
    """Return all skipped deals with full metadata."""
    return jsonify(database.get_skipped_deals())


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    if not debug:
        # Backward-compatible: FLASK_ENV=development still enables debug.
        debug = os.environ.get("FLASK_ENV", "production").lower() == "development"
    app.run(host=host, port=port, debug=debug)

# ── Startup validation ──────────────────────────────────────────────────────

_data_source = os.environ.get("DATA_SOURCE", "auto").strip().lower()
if _data_source == "api" and not ebay_api.is_configured:
    logger.warning(
        "DATA_SOURCE=api but EBAY_CLIENT_ID and EBAY_CLIENT_SECRET are not set — "
        "the eBay API engine will fall back to HTML scraping at search time. "
        "Set both credentials in your environment (or switch DATA_SOURCE to 'auto'/'scraper')."
    )
elif not ebay_api.is_configured:
    logger.info(
        "eBay API credentials not set — falling back to HTML scraper. "
        "Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET to use the official Browse API."
    )
else:
    logger.info("eBay API credentials found — Browse API will be used when data_source is 'api' or 'auto'.")

if not os.environ.get("GEMINI_API_KEY", "").strip():
    logger.info(
        "GEMINI_API_KEY not set — AI deal assessment is disabled. "
        "Deals will be returned without Gemini ratings. "
        "Set GEMINI_API_KEY in your environment to enable AI assessment."
    )
else:
    logger.info("GEMINI_API_KEY found — Gemini AI assessment is enabled.")
