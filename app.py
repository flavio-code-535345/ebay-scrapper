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
from datetime import UTC, datetime

from flask import Flask, Response, jsonify, render_template, request

import database
from ai_providers import (
    PROVIDER_META,
    VALID_PROVIDERS,
    apply_user_enabled,
    create_assessor,
    get_assessor,
    list_providers,
    normalize_provider,
    set_ebay_client_for_all,
)
from ai_providers.base import _detect_sports_kinect_deal
from ebay_api_client import EbayApiClient
from scraper import EbayScraper


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

database.init_db()

# Register eBay client with all AI providers (present and future).
set_ebay_client_for_all(ebay_api)

# Eager-load default assessor; additional providers load on first use.
assessor = create_assessor(os.environ.get("AI_PROVIDER", "gemini"))


def _load_provider_models_from_db() -> None:
    """Apply persisted model names onto each provider assessor."""
    for pid, meta in PROVIDER_META.items():
        saved = database.get_setting(meta["model_setting_key"])
        if saved:
            try:
                get_assessor(pid).model_name = saved
            except ValueError:
                logger.warning("Ignoring invalid saved model for %s: %r", pid, saved)


_load_provider_models_from_db()

# Load persisted AI-enabled toggle (default: True; stored as "true"/"false" string).
_saved_ai_enabled = database.get_setting("ai_enabled")
if _saved_ai_enabled is not None:
    apply_user_enabled(str(_saved_ai_enabled).lower() == "true")

# ── Data source helpers ────────────────────────────────────────────────────

_VALID_DATA_SOURCES = {"auto", "api", "scraper"}
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,99}$")


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
    """Read the user's AI-enabled toggle from the database.

    Always reads from the shared SQLite database rather than the in-memory
    ``assessor.user_enabled`` attribute so that multi-worker (Gunicorn)
    deployments remain consistent: updating the setting in one worker is
    immediately visible to all other workers on the next request.

    Defaults to ``True`` when no setting has been persisted yet.
    """
    val = database.get_setting("ai_enabled")
    return str(val).lower() == "true" if val is not None else True


def _db_ai_provider() -> str:
    """Active AI provider id from DB, then AI_PROVIDER env, else gemini."""
    val = database.get_setting("ai_provider")
    if val:
        return normalize_provider(val)
    return normalize_provider(os.environ.get("AI_PROVIDER", "gemini"))


def _current_assessor():
    """Resolve the active assessor for this request (multi-worker safe)."""
    global assessor
    provider = _db_ai_provider()
    a = get_assessor(provider)
    a.user_enabled = _db_ai_user_enabled()
    # Keep module-level alias in sync for tests / external imports.
    assessor = a
    return a


def _settings_payload() -> dict:
    """Shared settings/health fields for GET/POST /api/settings."""
    a = _current_assessor()
    provider = _db_ai_provider()
    meta = PROVIDER_META[provider]
    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    return {
        "ai_provider": provider,
        "ai_provider_label": meta["label"],
        "ai_model": a.model_name,
        "gemini_model": get_assessor("gemini").model_name,
        "opencode_go_model": get_assessor("opencode-go").model_name,
        "ai_enabled": _db_ai_user_enabled(),
        "ai_supports_images": bool(getattr(a, "supports_images", provider == "gemini")),
        "providers": list_providers(),
        "data_source": data_source_setting,
        "active_data_source": active_source,
        "ebay_api_configured": ebay_api.is_configured,
        "ebay_marketplace_id": ebay_api.marketplace_id,
        "ebay_language": ebay_api.accept_language,
        "ebay_locale": ebay_api.locale,
        "ebay_delivery_country": ebay_api.delivery_country,
        "germany_only": _db_germany_only(),
    }


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
    return "DEUTSCHLAND" in upper or "GERMANY" in upper


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400
    query = data.get("query", "").strip()
    try:
        max_results = max(1, min(int(data.get("max_results", 50)), 200))
    except (TypeError, ValueError):
        return jsonify({"error": "max_results must be a positive integer"}), 400

    if not query:
        return jsonify({"error": "query is required"}), 400

    # Select the appropriate search engine (API or scraper) based on settings.
    data_source_setting = _db_data_source()
    search_fn, active_source = _resolve_engine(data_source_setting)

    deals, search_errors = search_fn(query, max_results=max_results)
    logger.info(
        "Search for %r via %s returned %d deals, %d error(s)",
        query,
        active_source,
        len(deals),
        len(search_errors),
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
            logger.info("Germany-only filter removed %d non-German deal(s)", filtered_out)

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

    # Cap deals before AI — no score-based pre-filtering.
    _MAX_DISPLAY = 30
    deals_filtered = deals[:_MAX_DISPLAY]

    # Active AI provider (Gemini or OpenCode Go). Re-read from DB each request
    # so multi-worker Gunicorn processes stay consistent.
    active = _current_assessor()
    provider_id = _db_ai_provider()
    _user_enabled = _db_ai_user_enabled()
    ai_active = active.enabled and _user_enabled
    ai_assessments = active.assess_deals_batch(deals_filtered) if (deals_filtered and ai_active) else []

    timed_out = 0
    if active.enabled and ai_assessments:
        failed = sum(1 for a in ai_assessments if a is None)
        rate_limited = sum(1 for a in ai_assessments if a and a.get("ai_error_type") == "rate_limit")
        parse_errors = sum(1 for a in ai_assessments if a and a.get("ai_error_type") == "parse_error")
        timed_out = sum(1 for a in ai_assessments if a and a.get("ai_error_type") == "timeout")
        label = PROVIDER_META.get(provider_id, {}).get("label", provider_id)
        if failed:
            logger.warning(
                "%s batch: %d/%d items failed AI assessment.",
                label,
                failed,
                len(ai_assessments),
            )
        if rate_limited:
            logger.warning(
                "%s batch: %d/%d items rate-limited; skipping AI assessment.",
                label,
                rate_limited,
                len(ai_assessments),
            )
        if parse_errors:
            logger.warning(
                "%s batch: %d/%d items had parse errors; AI fields set to defaults.",
                label,
                parse_errors,
                len(ai_assessments),
            )
            for i, (deal, a) in enumerate(zip(deals_filtered, ai_assessments, strict=False)):
                if a and a.get("ai_error_type") == "parse_error":
                    logger.warning(
                        "%s parse error – item[%d]: %r",
                        label,
                        i,
                        (deal.get("title") or "")[:80],
                    )
        if timed_out:
            logger.warning(
                "%s batch: %d/%d items timed out; AI assessment skipped.",
                label,
                timed_out,
                len(ai_assessments),
            )
            for i, (deal, a) in enumerate(zip(deals_filtered, ai_assessments, strict=False)):
                if a and a.get("ai_error_type") == "timeout":
                    logger.info(
                        "%s timeout – item[%d]: %r",
                        label,
                        i,
                        (deal.get("title") or "")[:80],
                    )

    assessed = []
    for i, deal in enumerate(deals_filtered):
        ai_assessment = ai_assessments[i] if i < len(ai_assessments) else None
        assessed.append({**deal, **(ai_assessment or {})})

    # Sort deals: "Must Have"/"Must Buy" first, then all others — both groups
    # ordered newest → oldest by listing_date.
    def _parse_listing_date(d: dict) -> datetime:
        raw = d.get("listing_date") or ""
        if not raw:
            return datetime.min.replace(tzinfo=UTC)
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)

    def _sort_key(d: dict):
        rating = (d.get("ai_deal_rating") or "").lower()
        not_must_have = int(rating not in ("must have", "must buy"))  # 0 = must have first
        date = _parse_listing_date(d)
        return (not_must_have, -date.timestamp())

    assessed.sort(key=_sort_key)

    database.save_search(query, assessed)

    # Compute how many seconds remain in any rate-limit back-off window.
    paused_seconds = max(0.0, active.rate_limited_until - time.monotonic())

    saved_urls = set(d["url"] for d in database.get_saved_deals())
    for deal in assessed:
        deal["is_saved"] = deal.get("url") in saved_urls

    return jsonify(
        {
            "query": query,
            "deal_count": len(assessed),
            "deals": assessed,
            "errors": search_errors,
            "ai_enabled": active.enabled and _user_enabled,
            "ai_provider": provider_id,
            "ai_model": active.model_name,
            "ai_rate_limited": active.is_rate_limited,
            "ai_paused_seconds": round(paused_seconds),
            "ai_timeout_count": timed_out,
            "data_source": active_source,
            "germany_only": germany_only,
        }
    )


@app.route("/api/history")
def history():
    limit = int(request.args.get("limit", 20))
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
    active = _current_assessor()
    paused_seconds = max(0.0, active.rate_limited_until - time.monotonic())
    data_source_setting = _db_data_source()
    _, active_source = _resolve_engine(data_source_setting)
    provider = _db_ai_provider()
    return jsonify(
        {
            "status": "healthy",
            "ai_enabled": active.enabled and _db_ai_user_enabled(),
            "ai_provider": provider,
            "ai_rate_limited": active.is_rate_limited,
            "ai_paused_seconds": round(paused_seconds),
            "ai_model": active.model_name,
            "providers": list_providers(),
            "data_source": active_source,
            "data_source_setting": data_source_setting,
            "ebay_api_configured": ebay_api.is_configured,
            "ebay_marketplace_id": ebay_api.marketplace_id,
            "ebay_language": ebay_api.accept_language,
            "ebay_locale": ebay_api.locale,
            "ebay_delivery_country": ebay_api.delivery_country,
            "germany_only": _db_germany_only(),
        }
    )


@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify(_settings_payload())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    global assessor
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    errors = {}
    updated = {}

    # ── AI provider switch (persisted) ────────────────────────────────────
    if "ai_provider" in data:
        pid = normalize_provider(str(data["ai_provider"]))
        if pid not in VALID_PROVIDERS:
            errors["ai_provider"] = f"ai_provider must be one of: {', '.join(VALID_PROVIDERS)}"
        else:
            database.set_setting("ai_provider", pid)
            assessor = get_assessor(pid)
            assessor.user_enabled = _db_ai_user_enabled()
            updated["ai_provider"] = pid
            logger.info("Settings: ai_provider updated to %r", pid)

    def _save_model_for(provider_id: str, model: str, field_name: str) -> None:
        if not model:
            errors[field_name] = f"{field_name} must not be empty"
            return
        if not _MODEL_NAME_RE.match(model):
            errors[field_name] = (
                f"{field_name} contains invalid characters; use only letters, digits, hyphens, underscores, and dots"
            )
            return
        try:
            target = get_assessor(provider_id)
            target.model_name = model
            key = PROVIDER_META[provider_id]["model_setting_key"]
            database.set_setting(key, model)
            updated[field_name] = model
            logger.info("Settings: %s updated to %r", field_name, model)
        except ValueError as exc:
            errors[field_name] = str(exc)

    # Generic model for the *active* provider
    if "ai_model" in data:
        model = str(data["ai_model"]).strip()
        _save_model_for(_db_ai_provider(), model, "ai_model")

    if "gemini_model" in data:
        _save_model_for("gemini", str(data["gemini_model"]).strip(), "gemini_model")

    if "opencode_go_model" in data:
        _save_model_for("opencode-go", str(data["opencode_go_model"]).strip(), "opencode_go_model")

    if "ai_enabled" in data:
        ai_enabled = data["ai_enabled"]
        if not isinstance(ai_enabled, bool):
            errors["ai_enabled"] = "ai_enabled must be a boolean (true or false)"
        else:
            apply_user_enabled(ai_enabled)
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

    payload = _settings_payload()
    payload["updated"] = updated
    return jsonify(payload)


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

if os.environ.get("GEMINI_API_KEY", "").strip():
    logger.info("GEMINI_API_KEY found — Gemini AI assessment available.")
else:
    logger.info("GEMINI_API_KEY not set — Gemini provider unavailable.")

if os.environ.get("OPENCODE_GO_API_KEY", "").strip() or os.environ.get("OPENCODE_API_KEY", "").strip():
    logger.info("OpenCode Go API key found — OpenCode Go / Grok assessment available.")
else:
    logger.info(
        "OPENCODE_GO_API_KEY not set — OpenCode Go provider unavailable. "
        "Subscribe at https://opencode.ai/auth and set OPENCODE_GO_API_KEY."
    )

logger.info("Active AI provider at startup: %s", _db_ai_provider())
