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
from ai_providers import create_assessor
from ai_providers.base import _SPORTS_KINECT_KEYWORDS_RE, _detect_sports_kinect_deal
from ai_providers.gemini import _MODEL_NAME as _GEMINI_DEFAULT_MODEL
from ai_providers.gemini import _is_text_only_model
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

assessor = create_assessor()

# Register eBay client with the assessor so it can fetch real per-game prices.
assessor.set_ebay_client(ebay_api)

# Load persisted Gemini model (if any) so it takes effect without a restart.
_saved_model = database.get_setting("gemini_model")
if _saved_model:
    if _is_text_only_model(_saved_model):
        logger.warning(
            "Saved model %r is text-only — resetting to default %r.",
            _saved_model,
            _GEMINI_DEFAULT_MODEL,
        )
        database.set_setting("gemini_model", _GEMINI_DEFAULT_MODEL)
    else:
        assessor.model_name = _saved_model

# Load persisted AI-enabled toggle (default: True; stored as "true"/"false" string).
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
    """Read the user's AI-enabled toggle from the database.

    Always reads from the shared SQLite database rather than the in-memory
    ``assessor.user_enabled`` attribute so that multi-worker (Gunicorn)
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
    return "DEUTSCHLAND" in upper or "GERMANY" in upper


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    # Accept a single "query" (backward-compat) or "queries" (array) for
    # multi-phrase searches that catch more listing variations.
    raw_query = data.get("query", "").strip()
    raw_queries = data.get("queries")
    if raw_queries and isinstance(raw_queries, list):
        queries = [str(q).strip() for q in raw_queries if str(q).strip()]
    elif raw_query:
        queries = [raw_query]
    else:
        return jsonify({"error": "query or queries is required"}), 400
    query = queries[0]  # canonical query ref for logging / response payload

    try:
        max_results = max(1, min(int(data.get("max_results", 50)), 200))
    except (TypeError, ValueError):
        return jsonify({"error": "max_results must be a positive integer"}), 400

    data_source_setting = _db_data_source()
    search_fn, active_source = _resolve_engine(data_source_setting)

    # Run each query and merge results, deduplicating by URL.
    all_deals: list[dict] = []
    all_errors: list[str] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for q in queries:
        deals, errs = search_fn(q, max_results=max_results)
        all_errors.extend(errs)
        added = 0
        for d in deals:
            url = d.get("url", "")
            title = (d.get("title") or "").strip().lower()
            # Normalise: strip eBay tracking params so the same listing
            # across different queries deduplicates correctly.
            if "?" in url:
                url = url.split("?")[0]
            if not url:
                continue
            if url in seen_urls:
                continue
            # Also skip near-duplicates: same title + same price (likely same listing
            # with a slightly different eBay item-id url — common with multi-variant listings).
            price = d.get("price")
            title_price_key = f"{title}|{price}"
            if title_price_key in seen_titles and price is not None:
                continue
            seen_urls.add(url)
            if price is not None:
                seen_titles.add(title_price_key)
            all_deals.append(d)
            added += 1
        logger.info(
            "Search for %r via %s returned %d deals (%d new, %d errors)",
            q,
            active_source,
            len(deals),
            added,
            len(errs),
        )
    deals = all_deals
    search_errors = all_errors
    logger.info(
        "Multi-query search (%d phrases) via %s: %d unique deals, %d total errors",
        len(queries),
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

    # Post-filter: drop sports/Kinect-only deals — these have very low
    # resale value (FIFA, Forza, Kinect, TopSpin, etc.) and should never
    # surface as desirable results.  HOWEVER, if the title also contains
    # clearly non-sports games, keep it and let Gemini score it — the
    # non-sports titles may still make the bundle profitable.
    before_sports = len(deals)
    _filtered = []
    for d in deals:
        warning = _detect_sports_kinect_deal(d)
        if not warning:
            _filtered.append(d)
            continue
        # Check: does the title contain any bundle-indicating quantity of
        # non-trivial content (≥2 game-like tokens that are NOT sports)?
        title = (d.get("title") or "").lower()
        # Strip sports keywords, see what's left.
        cleaned = _SPORTS_KINECT_KEYWORDS_RE.sub(" ", title)
        cleaned = re.sub(r"\b(je\s+stk|stk|pro|jede|und|mit|für|oder|stück|wahl|aus)\b", " ", cleaned)
        cleaned = re.sub(r"\d+\s*[€x×]|\b\d+\b", " ", cleaned)
        tokens = [t for t in cleaned.split() if len(t) > 2 and t not in ("die", "der", "das", "ein", "sie", "von")]
        # A bundle likely has ≥3 remaining non-sports / non-noise tokens.
        if len(tokens) >= 3:
            _filtered.append(d)
        else:
            logger.debug("Sports-only deal removed: %r", (d.get("title") or "")[:80])
    deals = _filtered
    filtered_sports = before_sports - len(deals)
    if filtered_sports:
        logger.info(
            "Sports/Kinect filter removed %d deal(s) with low resale value",
            filtered_sports,
        )

    # Cap deals before sending to Gemini — no score-based pre-filtering.
    # All deals that pass the post-filters above are eligible for AI assessment.
    _MAX_DISPLAY = 30
    deals_filtered = deals[:_MAX_DISPLAY]

    # AI assessment via Gemini: send only the top filtered deals in a single
    # request to minimise quota consumption rather than calling once per deal.
    # Skip entirely when the user has disabled AI evaluation via the toggle.
    # Re-read ai_enabled from the database on every request so that the toggle
    # is respected in multi-worker (Gunicorn) deployments where in-memory state
    # is not shared across processes.
    _user_enabled = _db_ai_user_enabled()
    ai_active = assessor.enabled and _user_enabled
    ai_assessments = assessor.assess_deals_batch(deals_filtered) if (deals_filtered and ai_active) else []

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
            "ai_enabled": assessor.enabled and _user_enabled,
            "ai_rate_limited": assessor.is_rate_limited,
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
    global assessor
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON with Content-Type: application/json"}), 400

    # Gemini model names: alphanumeric, hyphens, underscores, and dots only.
    _MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,99}$")

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
