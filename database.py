#!/usr/bin/env python3
"""
Database Management
Handles SQLite database operations for search history and deal storage
"""

import sqlite3
import csv
import io
import json
import time
import os
from typing import List, Dict, Optional

DB_PATH = os.environ.get('DB_PATH', 'ebay_deals.db')


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables"""
    conn = get_connection()
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
            overall_score REAL,
            price_score REAL,
            seller_score REAL,
            condition_score REAL,
            trend_score REAL,
            recommendation TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (search_id) REFERENCES searches(id)
        );
    """)

    # Add AI-assessment columns to existing databases (migration-safe).
    _add_column_if_missing(cursor, "deals", "ai_deal_rating", "TEXT")
    _add_column_if_missing(cursor, "deals", "ai_confidence_score", "REAL")
    _add_column_if_missing(cursor, "deals", "ai_visual_findings", "TEXT")
    _add_column_if_missing(cursor, "deals", "ai_red_flags", "TEXT")
    _add_column_if_missing(cursor, "deals", "ai_fair_market_estimate", "TEXT")
    _add_column_if_missing(cursor, "deals", "ai_verdict_summary", "TEXT")
    _add_column_if_missing(cursor, "deals", "ai_assessed", "INTEGER DEFAULT 0")

    conn.commit()
    conn.close()


def _add_column_if_missing(cursor, table: str, column: str, col_type: str) -> None:
    """Add *column* to *table* when it does not already exist.

    Both *table* and *column* are validated against an explicit allowlist to
    prevent SQL injection through these DDL-level parameters (SQLite does not
    support parameterised DDL statements).
    """
    _ALLOWED_TABLES = {"deals", "searches"}
    _ALLOWED_COLUMNS = {
        "ai_deal_rating",
        "ai_confidence_score",
        "ai_visual_findings",
        "ai_red_flags",
        "ai_fair_market_estimate",
        "ai_verdict_summary",
        "ai_assessed",
    }
    _ALLOWED_TYPES = {
        "TEXT",
        "REAL",
        "INTEGER",
        "INTEGER DEFAULT 0",
    }

    if table not in _ALLOWED_TABLES:
        raise ValueError(f"_add_column_if_missing: disallowed table name: {table!r}")
    if column not in _ALLOWED_COLUMNS:
        raise ValueError(f"_add_column_if_missing: disallowed column name: {column!r}")
    if col_type not in _ALLOWED_TYPES:
        raise ValueError(f"_add_column_if_missing: disallowed column type: {col_type!r}")

    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception:
        pass  # column already exists – SQLite raises OperationalError


def save_search(query: str, deals: List[Dict]) -> int:
    """Save a search and its associated deals; returns the search id"""
    conn = get_connection()
    cursor = conn.cursor()
    now = time.time()

    cursor.execute(
        "INSERT INTO searches (query, result_count, created_at) VALUES (?, ?, ?)",
        (query, len(deals), now),
    )
    search_id = cursor.lastrowid

    for deal in deals:
        # Serialise list fields (visual_findings, red_flags) as JSON strings.
        visual_findings = deal.get('ai_visual_findings')
        red_flags = deal.get('ai_red_flags')

        cursor.execute(
            """INSERT INTO deals
               (search_id, title, price, condition, seller_rating, url, shipping,
                is_trending, overall_score, price_score, seller_score,
                condition_score, trend_score, recommendation,
                ai_deal_rating, ai_confidence_score, ai_visual_findings,
                ai_red_flags, ai_fair_market_estimate, ai_verdict_summary,
                ai_assessed, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                search_id,
                deal.get('title'),
                deal.get('price'),
                deal.get('condition'),
                deal.get('seller_rating'),
                deal.get('url'),
                deal.get('shipping'),
                int(bool(deal.get('is_trending'))),
                deal.get('overall_score'),
                deal.get('price_score'),
                deal.get('seller_score'),
                deal.get('condition_score'),
                deal.get('trend_score'),
                deal.get('recommendation'),
                deal.get('ai_deal_rating'),
                deal.get('ai_confidence_score'),
                json.dumps(visual_findings) if isinstance(visual_findings, list) else visual_findings,
                json.dumps(red_flags) if isinstance(red_flags, list) else red_flags,
                deal.get('ai_fair_market_estimate'),
                deal.get('ai_verdict_summary'),
                int(bool(deal.get('ai_assessed'))),
                now,
            ),
        )

    conn.commit()
    conn.close()
    return search_id


def get_history(limit: int = 20) -> List[Dict]:
    """Return recent search history"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM searches ORDER BY created_at DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_deals_by_search(search_id: int) -> List[Dict]:
    """Return deals for a specific search"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM deals WHERE search_id = ?", (search_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def export_csv() -> str:
    """Export all deals as CSV string"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT d.*, s.query FROM deals d
           JOIN searches s ON d.search_id = s.id
           ORDER BY d.created_at DESC"""
    )
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return output.getvalue()


def get_stats() -> Dict:
    """Return database statistics"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM searches")
    searches = cursor.fetchone()['total']
    cursor.execute("SELECT COUNT(*) as total FROM deals")
    deals = cursor.fetchone()['total']
    conn.close()
    return {'total_searches': searches, 'total_deals': deals}
