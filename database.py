"""Database Management — SQLite operations for search history and deal storage."""

import csv
import io
import json
import os
import sqlite3
import sys
import time
from contextlib import contextmanager, suppress

DB_PATH = os.environ.get("DB_PATH", "ebay_deals.db")


@contextmanager
def get_db():
    """Yield a SQLite connection with row-factory set, closing on exit.

    Writes call ``conn.commit()`` before the context exits so that
    callers never leave dangling transactions.  Read-only callers
    must call ``conn.commit()`` themselves or rely on the default
    autocommit from the context manager (no-op for reads).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def _ensure_wal_mode():
    """Enable WAL journal mode for better concurrent-read performance.

    If WAL mode cannot be enabled (e.g., readonly filesystem), continue anyway.
    WAL is an optimization, not required for database functionality.
    """
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA synchronous=NORMAL")
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        print(f"[WARNING] Could not enable WAL mode: {e}", file=sys.stderr)


def init_db():
    _ensure_wal_mode()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                result_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS deals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER,
                title TEXT,
                price REAL,
                condition TEXT,
                seller_rating REAL,
                url TEXT,
                shipping TEXT,
                is_trending INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                FOREIGN KEY (search_id) REFERENCES searches(id)
            );

            CREATE INDEX IF NOT EXISTS idx_deals_search_id ON deals(search_id);
            CREATE INDEX IF NOT EXISTS idx_deals_created_at ON deals(created_at);
        """)

        _add_column_if_missing(cursor, "deals", "ai_deal_rating", "TEXT")
        _add_column_if_missing(cursor, "deals", "ai_confidence_score", "REAL")
        _add_column_if_missing(cursor, "deals", "ai_visual_findings", "TEXT")
        _add_column_if_missing(cursor, "deals", "ai_red_flags", "TEXT")
        _add_column_if_missing(cursor, "deals", "ai_fair_market_estimate", "TEXT")
        _add_column_if_missing(cursor, "deals", "ai_verdict_summary", "TEXT")
        _add_column_if_missing(cursor, "deals", "ai_assessed", "INTEGER DEFAULT 0")
        _add_column_if_missing(cursor, "deals", "ai_potential_scam", "INTEGER DEFAULT 0")
        _add_column_if_missing(cursor, "deals", "ai_scam_warning", "TEXT")
        _add_column_if_missing(cursor, "deals", "image_issues", "TEXT")
        _add_column_if_missing(cursor, "deals", "image_urls", "TEXT")
        _add_column_if_missing(cursor, "deals", "item_location", "TEXT")
        _add_column_if_missing(cursor, "deals", "description", "TEXT")
        _add_column_if_missing(cursor, "deals", "seller_count", "TEXT")
        _add_column_if_missing(cursor, "deals", "listing_date", "TEXT")
        _add_column_if_missing(cursor, "deals", "ai_itemized_resale_estimates", "TEXT")
        _add_column_if_missing(cursor, "deals", "ai_estimated_total_cost", "REAL")
        _add_column_if_missing(cursor, "deals", "ai_estimated_gross_profit", "REAL")

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS user_saved_deals (
                url TEXT PRIMARY KEY NOT NULL,
                title TEXT,
                price REAL,
                saved_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_skipped_deals (
                url TEXT PRIMARY KEY NOT NULL,
                title TEXT,
                price REAL,
                skipped_at REAL NOT NULL
            );
        """)

        _add_column_if_missing(cursor, "user_skipped_deals", "title", "TEXT")
        _add_column_if_missing(cursor, "user_skipped_deals", "price", "REAL")

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY NOT NULL,
                value TEXT NOT NULL
            );
        """)


def _add_column_if_missing(cursor, table: str, column: str, col_type: str) -> None:
    _ALLOWED_TABLES = {"deals", "searches", "user_skipped_deals"}
    _ALLOWED_COLUMNS = {
        "ai_deal_rating",
        "ai_confidence_score",
        "ai_visual_findings",
        "ai_red_flags",
        "ai_fair_market_estimate",
        "ai_verdict_summary",
        "ai_assessed",
        "ai_potential_scam",
        "ai_scam_warning",
        "image_issues",
        "image_urls",
        "item_location",
        "description",
        "seller_count",
        "listing_date",
        "title",
        "price",
        "ai_itemized_resale_estimates",
        "ai_estimated_total_cost",
        "ai_estimated_gross_profit",
    }
    _ALLOWED_TYPES = {"TEXT", "REAL", "INTEGER", "INTEGER DEFAULT 0"}
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"_add_column_if_missing: disallowed table name: {table!r}")
    if column not in _ALLOWED_COLUMNS:
        raise ValueError(f"_add_column_if_missing: disallowed column name: {column!r}")
    if col_type not in _ALLOWED_TYPES:
        raise ValueError(f"_add_column_if_missing: disallowed column type: {col_type!r}")
    with suppress(sqlite3.OperationalError):
        # Column already exists (or table is locked); safe to ignore.
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def _encode_list_field(value) -> str | None:
    """Encode a list-typed deal field as a JSON array string for storage.

    Always produces a JSON array (or None) rather than sometimes storing the
    raw value as-is — the old code only JSON-encoded when the value was
    already a Python list, so a stray non-list truthy value would be stored
    as raw text and round-trip inconsistently through get_deals_by_search's
    decoder (which only decodes bytes it recognizes as JSON).
    """
    if value is None:
        return None
    return json.dumps(value if isinstance(value, list) else [])


def save_search(query: str, deals: list[dict]) -> int:
    with get_db() as conn:
        cursor = conn.cursor()
        now = time.time()
        cursor.execute(
            "INSERT INTO searches (query, result_count, created_at) VALUES (?, ?, ?)",
            (query, len(deals), now),
        )
        search_id = cursor.lastrowid
        for deal in deals:
            cursor.execute(
                """INSERT INTO deals (search_id, title, price, condition, seller_rating,
                   url, shipping, is_trending, ai_deal_rating,
                   ai_confidence_score, ai_visual_findings, ai_red_flags,
                   ai_fair_market_estimate, ai_verdict_summary, ai_assessed,
                   ai_potential_scam, ai_scam_warning, image_issues, image_urls,
                   item_location, description, seller_count, listing_date,
                   ai_itemized_resale_estimates, ai_estimated_total_cost,
                   ai_estimated_gross_profit, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    search_id,
                    deal.get("title"),
                    deal.get("price"),
                    deal.get("condition"),
                    deal.get("seller_rating"),
                    deal.get("url"),
                    deal.get("shipping"),
                    int(bool(deal.get("is_trending"))),
                    deal.get("ai_deal_rating"),
                    deal.get("ai_confidence_score"),
                    _encode_list_field(deal.get("ai_visual_findings")),
                    _encode_list_field(deal.get("ai_red_flags")),
                    deal.get("ai_fair_market_estimate"),
                    deal.get("ai_verdict_summary"),
                    int(bool(deal.get("ai_assessed"))),
                    int(bool(deal.get("ai_potential_scam"))),
                    deal.get("ai_scam_warning"),
                    _encode_list_field(deal.get("image_issues")),
                    _encode_list_field(deal.get("image_urls")),
                    deal.get("item_location"),
                    deal.get("description"),
                    deal.get("seller_count"),
                    deal.get("listing_date"),
                    _encode_list_field(deal.get("ai_itemized_resale_estimates")),
                    deal.get("ai_estimated_total_cost"),
                    deal.get("ai_estimated_gross_profit"),
                    now,
                ),
            )
    return search_id


def get_history(limit: int = 20) -> list[dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM searches ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]


def get_deals_by_search(search_id: int) -> list[dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM deals WHERE search_id = ?", (search_id,))
        rows = [dict(row) for row in cursor.fetchall()]
    _JSON_LIST_FIELDS = (
        "ai_visual_findings",
        "ai_red_flags",
        "image_issues",
        "image_urls",
        "ai_itemized_resale_estimates",
    )
    for row in rows:
        for field in _JSON_LIST_FIELDS:
            raw = row.get(field)
            if isinstance(raw, str):
                with suppress(json.JSONDecodeError, ValueError):
                    row[field] = json.loads(raw)
    return rows


def export_csv() -> str:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT d.title, d.price, d.condition, d.seller_rating,
                   d.shipping, d.url, d.item_location, d.listing_date,
                   d.ai_deal_rating, d.ai_confidence_score, d.ai_verdict_summary,
                   d.ai_fair_market_estimate, d.ai_estimated_total_cost,
                   d.ai_estimated_gross_profit, d.ai_potential_scam, d.ai_scam_warning,
                   s.query, d.created_at
               FROM deals d
               JOIN searches s ON d.search_id = s.id
               ORDER BY d.created_at DESC"""
        )
        rows = cursor.fetchall()
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return output.getvalue()


def get_stats() -> dict:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM searches")
        searches = cursor.fetchone()["total"]
        cursor.execute("SELECT COUNT(*) as total FROM deals")
        deals = cursor.fetchone()["total"]
    return {"total_searches": searches, "total_deals": deals}


def get_setting(key: str, default: str | None = None) -> str | None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def save_deal(url: str, title: str = "", price: float = 0.0) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_saved_deals (url, title, price, saved_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(url) DO UPDATE SET title = excluded.title,"
            " price = excluded.price, saved_at = excluded.saved_at",
            (url, title, price, time.time()),
        )


def unsave_deal(url: str) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_saved_deals WHERE url = ?", (url,))


def get_saved_deals() -> list[dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, price, saved_at FROM user_saved_deals ORDER BY saved_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def skip_deal(url: str, title: str = "", price: float = 0.0) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO user_skipped_deals (url, title, price, skipped_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(url) DO UPDATE SET title = excluded.title,"
            " price = excluded.price, skipped_at = excluded.skipped_at",
            (url, title, price, time.time()),
        )


def unskip_deal(url: str) -> None:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_skipped_deals WHERE url = ?", (url,))


def get_skipped_deal_urls() -> list[str]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url FROM user_skipped_deals")
        return [row["url"] for row in cursor.fetchall()]


def get_skipped_deals() -> list[dict]:
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, price, skipped_at FROM user_skipped_deals ORDER BY skipped_at DESC")
        return [dict(row) for row in cursor.fetchall()]
