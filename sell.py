"""The Sell page: list a game on the seller's own eBay account in under a minute.

Scan the EAN → eBay's catalog supplies the product → the price comes from the
seller's rule (cheapest comparable Buy-It-Now total) → photos straight from the
phone camera → Publish. Shipping, returns, location and description are copied
from one of the seller's existing listings (see ebay_seller.py).

Everything here acts on the seller's eBay account, so every route except the
login itself requires the ``APP_PASSWORD`` session; without ``APP_PASSWORD`` the
page only explains the setup.
"""

from __future__ import annotations

import functools
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
import urllib.parse

from flask import Blueprint, jsonify, redirect, render_template, request, session, url_for

import database
from ebay_seller import FREE_LISTINGS_PER_MONTH, Draft, EbaySeller, SellerError, suggest_price

logger = logging.getLogger(__name__)

bp = Blueprint("sell", __name__)

seller = EbaySeller()
prices = None  # the app's EbayApiClient (app token, Browse API), set by app.py

_MAX_PHOTO_BYTES = 12 * 1024 * 1024
_FAILED_LOGIN_DELAY_S = 1.0


def _password() -> str:
    return os.environ.get("APP_PASSWORD", "")


def _password_tag() -> str:
    """Stored in the session: changing APP_PASSWORD signs every browser out."""
    return hashlib.sha256(("sell:" + _password()).encode()).hexdigest()[:32]


def logged_in() -> bool:
    return bool(_password()) and hmac.compare_digest(session.get("sell_auth", ""), _password_tag())


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not logged_in():
            if request.path.startswith("/api/"):
                return jsonify({"error": "Please sign in first."}), 401
            return redirect(url_for("sell.page"))
        return view(*args, **kwargs)

    return wrapped


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ── Page, login, eBay connection ─────────────────────────────────────────────


@bp.route("/sell")
def page():
    return render_template(
        "sell.html",
        password_set=bool(_password()),
        logged_in=logged_in(),
        ebay_configured=seller.is_configured,
        prices_configured=bool(prices and prices.is_configured),
        message=request.args.get("message", ""),
    )


@bp.route("/sell/login", methods=["POST"])
def login():
    supplied = request.form.get("password", "")
    if _password() and hmac.compare_digest(supplied.encode(), _password().encode()):
        session.permanent = True
        session["sell_auth"] = _password_tag()
        return redirect(url_for("sell.page"))
    time.sleep(_FAILED_LOGIN_DELAY_S)
    return redirect(url_for("sell.page", message="Wrong password."))


@bp.route("/sell/logout", methods=["POST"])
def logout():
    session.pop("sell_auth", None)
    return redirect(url_for("sell.page"))


@bp.route("/sell/ebay/connect")
@login_required
def ebay_connect():
    if not seller.is_configured:
        return redirect(url_for("sell.page", message="EBAY_RUNAME is not set — see the README."))
    state = secrets.token_urlsafe(24)
    session["ebay_oauth_state"] = state
    return redirect(seller.authorize_url(state))


def _finish_connect(code: str, state: str) -> str:
    expected = session.pop("ebay_oauth_state", None)
    if not expected or not hmac.compare_digest(state or "", expected):
        raise SellerError("That eBay sign-in does not belong to this session — please connect again.")
    if not code:
        raise SellerError("eBay returned no sign-in code.")
    return seller.connect(code)


@bp.route("/sell/ebay/callback")
@login_required
def ebay_callback():
    """eBay's "auth accepted URL" (configured on the RuName) points here."""
    if request.args.get("error"):
        return redirect(url_for("sell.page", message="The connection was declined on eBay."))
    try:
        username = _finish_connect(request.args.get("code", ""), request.args.get("state", ""))
    except SellerError as exc:
        return redirect(url_for("sell.page", message=str(exc)))
    return redirect(url_for("sell.page", message=f"Connected to {username}."))


@bp.route("/api/sell/ebay/code", methods=["POST"])
@login_required
def ebay_code():
    """Fallback when the auth accepted URL can't reach this app: the seller
    pastes the address eBay sent them to, which carries the code."""
    pasted = str((request.get_json(silent=True) or {}).get("url", "")).strip()
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(pasted).query)
    try:
        username = _finish_connect(query.get("code", [""])[0], query.get("state", [""])[0])
    except SellerError as exc:
        return _error(str(exc))
    return jsonify({"username": username})


@bp.route("/api/sell/ebay/disconnect", methods=["POST"])
@login_required
def ebay_disconnect():
    seller.disconnect()
    return jsonify({"connected": False})


# ── Status and template ──────────────────────────────────────────────────────


@bp.route("/api/sell/status")
@login_required
def status():
    template = seller.template()
    body = {
        "connected": seller.is_connected,
        "username": seller.username,
        "template": template.summary() if template else None,
        "recent": database.get_sell_listings(),
        "conditions": [],
        "free_listings": None,
    }
    if body["connected"] and template:
        try:
            body["conditions"] = seller.conditions(template.category_id)
        except SellerError as exc:
            logger.warning("Could not load item conditions: %s", exc)
    if body["connected"]:
        try:
            body["free_listings"] = {"used": seller.listings_this_month(), "allowance": FREE_LISTINGS_PER_MONTH}
        except SellerError as exc:
            logger.warning("Could not count this month's listings: %s", exc)
    return jsonify(body)


@bp.route("/api/sell/template", methods=["POST"])
@login_required
def import_template():
    item = str((request.get_json(silent=True) or {}).get("item", ""))
    try:
        template = seller.import_template(item)
    except SellerError as exc:
        return _error(str(exc))
    return jsonify({"template": template.summary()})


# ── One listing: product, price, photos, publish ─────────────────────────────


@bp.route("/api/sell/product")
@login_required
def product():
    """Catalog products for an EAN or title, plus — for the first match — the
    cheapest comparable offers and the price they suggest."""
    query = request.args.get("q", "").strip()
    if not query:
        return _error("Enter an EAN or a title.")
    try:
        products = seller.lookup_products(query)
    except SellerError as exc:
        return _error(str(exc), 502)
    ean = products[0]["ean"] if products else re.sub(r"\D", "", query)
    template = seller.template()
    offers, errors = [], []
    if prices is not None and prices.is_configured and re.fullmatch(r"\d{8,14}", ean or ""):
        offers, errors = prices.cheapest_offers(
            ean, exclude_seller=seller.username, postal_code=template.postal_code if template else ""
        )
    return jsonify(
        {
            "products": products,
            "ean": ean,
            "offers": offers,
            "suggested_price": suggest_price(offers),
            "shipping_cost": template.shipping_cost if template else None,
            "errors": errors,
        }
    )


@bp.route("/api/sell/photos", methods=["POST"])
@login_required
def upload_photo():
    photo = request.files.get("photo")
    if photo is None:
        return _error("No photo received.")
    data = photo.read(_MAX_PHOTO_BYTES + 1)
    if len(data) > _MAX_PHOTO_BYTES:
        return _error("The photo is larger than 12 MB.")
    try:
        url = seller.upload_photo(data, photo.filename or "foto.jpg", photo.mimetype or "image/jpeg")
    except SellerError as exc:
        return _error(str(exc), 502)
    return jsonify({"url": url})


def _draft_from(body: dict) -> Draft:
    try:
        price = float(str(body.get("price", "")).replace(",", "."))
    except ValueError:
        price = 0.0
    photos = [str(u) for u in body.get("photos") or [] if str(u).startswith("https://")]
    specifics = {str(k): str(v) for k, v in (body.get("item_specifics") or {}).items() if str(v).strip()}
    best_offer = body.get("best_offer")
    return Draft(
        title=str(body.get("title", "")).strip(),
        price=price,
        condition_id=str(body.get("condition_id", "")).strip(),
        photo_urls=photos,
        ean=re.sub(r"\D", "", str(body.get("ean", ""))),
        epid=str(body.get("epid", "")).strip(),
        best_offer=None if best_offer is None else bool(best_offer),
        item_specifics=specifics,
    )


@bp.route("/api/sell/publish", methods=["POST"])
@login_required
def publish():
    body = request.get_json(silent=True) or {}
    draft = _draft_from(body)
    verify_only = bool(body.get("verify_only"))
    try:
        result = seller.publish(draft, verify_only=verify_only)
    except SellerError as exc:
        return _error(str(exc), 422)
    if not verify_only:
        database.add_sell_listing(result.item_id, draft.title, draft.ean, draft.price, result.url)
        logger.info("Listed %s on eBay: %s (%.2f €)", result.item_id, draft.title, draft.price)
    return jsonify(
        {
            "item_id": result.item_id,
            "url": result.url,
            "fees": result.fees,
            "warnings": result.warnings,
            "verified_only": verify_only,
        }
    )
