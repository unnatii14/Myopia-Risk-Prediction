"""
Tiny database layer that works with BOTH backends:

  * Production  -> PostgreSQL, when the DATABASE_URL env var is set
                   (Render Postgres provides this automatically).
  * Local / dev -> SQLite file (data.db), no setup required.

Only CREATE TABLE differs between the two dialects; all other queries use a
shared placeholder (PH) plus a few small helpers so the same code runs on
either backend.
"""
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
IS_POSTGRES = DATABASE_URL.startswith("postgres")

# Parameter placeholder: Postgres uses %s, SQLite uses ?
PH = "%s" if IS_POSTGRES else "?"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(BASE_DIR, "data.db")


def get_conn():
    """Return a DB-API connection to whichever backend is configured."""
    if IS_POSTGRES:
        import psycopg2
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url)
    import sqlite3
    return sqlite3.connect(SQLITE_PATH)


# -- Cross-backend helpers --------------------------------------------------

def is_integrity_error(e):
    """True if the exception is a unique/constraint violation on either backend."""
    name = type(e).__name__
    return "IntegrityError" in name or "UniqueViolation" in name


def row_as_dict(cur):
    """fetchone() as a dict keyed by column name (sqlite3 + psycopg2)."""
    if cur.description is None:
        return None
    cols = [d[0] for d in cur.description]
    r = cur.fetchone()
    return dict(zip(cols, r)) if r is not None else None


def rows_as_dicts(cur):
    """fetchall() as a list of dicts keyed by column name."""
    if cur.description is None:
        return []
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def insert_returning_id(cur, sql, params):
    """Run an INSERT and return the new row id on either backend.
    Pass sql WITHOUT a RETURNING clause."""
    if IS_POSTGRES:
        cur.execute(sql + " RETURNING id", params)
        return cur.fetchone()[0]
    cur.execute(sql, params)
    return cur.lastrowid


# -- Table creation (idempotent) --------------------------------------------

def init_users_table():
    conn = get_conn(); cur = conn.cursor()
    if IS_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                name          TEXT NOT NULL,
                child_name    TEXT,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                child_name    TEXT,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)
    conn.commit(); conn.close()


def init_screenings_table():
    conn = get_conn(); cur = conn.cursor()
    if IS_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS screenings (
                id          SERIAL PRIMARY KEY,
                email       TEXT NOT NULL,
                child_name  TEXT,
                screened_at TIMESTAMPTZ DEFAULT NOW(),
                input_data  TEXT NOT NULL,
                risk_score  INTEGER NOT NULL,
                risk_level  TEXT NOT NULL,
                has_re      INTEGER,
                diopters    REAL,
                severity    TEXT
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS screenings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email       TEXT NOT NULL,
                child_name  TEXT,
                screened_at TEXT DEFAULT (datetime('now')),
                input_data  TEXT NOT NULL,
                risk_score  INTEGER NOT NULL,
                risk_level  TEXT NOT NULL,
                has_re      INTEGER,
                diopters    REAL,
                severity    TEXT
            )
        """)
    conn.commit(); conn.close()


def init_contrib_table():
    conn = get_conn(); cur = conn.cursor()
    if IS_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contributions (
                id                 SERIAL PRIMARY KEY,
                email              TEXT    NOT NULL,
                image_b64          TEXT    NOT NULL,
                image_mime         TEXT,
                model_prediction   TEXT,
                model_confidence   REAL,
                reported_label     TEXT,
                consent_version    TEXT    NOT NULL,
                review_status      TEXT    NOT NULL DEFAULT 'pending',
                reviewer_label     TEXT,
                created_at         TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS contributions (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                email              TEXT    NOT NULL,
                image_b64          TEXT    NOT NULL,
                image_mime         TEXT,
                model_prediction   TEXT,
                model_confidence   REAL,
                reported_label     TEXT,
                consent_version    TEXT    NOT NULL,
                review_status      TEXT    NOT NULL DEFAULT 'pending',
                reviewer_label     TEXT,
                created_at         TEXT    DEFAULT (datetime('now'))
            )
        """)
    conn.commit(); conn.close()
