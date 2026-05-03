"""
Screening History Blueprint — save and retrieve past screenings per user.
Routes:
  POST /history/save   — save a new screening result
  GET  /history        — get all screenings for the logged-in user
  GET  /history/latest — get only the most recent screening
"""

from flask import Blueprint, request, jsonify
import sqlite3
import jwt
import json
import os
from datetime import datetime, timezone

history_bp = Blueprint("history", __name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "users.db")
JWT_SECRET = os.environ.get("JWT_SECRET", "myopia_dev_secret_key_2024")


def _init_history_table():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
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
    conn.commit()
    conn.close()

_init_history_table()


def _get_email_from_token():
    """Extract email from Bearer token. Returns email string or None."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("email")
    except Exception:
        return None


@history_bp.route("/save", methods=["POST"])
def save_screening():
    email = _get_email_from_token()
    if not email:
        return jsonify({"error": "Unauthorised"}), 401

    data = request.get_json() or {}
    input_data  = data.get("input_data", {})
    result      = data.get("result", {})
    child_name  = (input_data.get("childName") or "").strip() or None

    risk_score  = result.get("risk_score")
    risk_level  = result.get("risk_level")
    has_re      = 1 if result.get("has_re") else 0
    diopters    = result.get("diopters")
    severity    = result.get("severity")

    if risk_score is None or not risk_level:
        return jsonify({"error": "result.risk_score and result.risk_level are required"}), 400

    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        """INSERT INTO screenings
           (email, child_name, input_data, risk_score, risk_level, has_re, diopters, severity)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (email, child_name, json.dumps(input_data), risk_score, risk_level, has_re, diopters, severity)
    )
    new_id = cur.lastrowid
    conn.commit()
    conn.close()

    return jsonify({"ok": True, "id": new_id}), 201


@history_bp.route("", methods=["GET"])
def get_history():
    email = _get_email_from_token()
    if not email:
        return jsonify({"error": "Unauthorised"}), 401

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, child_name, screened_at, risk_score, risk_level,
                  has_re, diopters, severity, input_data
           FROM screenings WHERE email = ?
           ORDER BY screened_at DESC LIMIT 50""",
        (email,)
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id":          r["id"],
            "child_name":  r["child_name"],
            "screened_at": r["screened_at"],
            "risk_score":  r["risk_score"],
            "risk_level":  r["risk_level"],
            "has_re":      bool(r["has_re"]),
            "diopters":    r["diopters"],
            "severity":    r["severity"],
            "input_data":  json.loads(r["input_data"]) if r["input_data"] else {},
        })

    return jsonify(results), 200


@history_bp.route("/latest", methods=["GET"])
def get_latest():
    email = _get_email_from_token()
    if not email:
        return jsonify({"error": "Unauthorised"}), 401

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT id, child_name, screened_at, risk_score, risk_level,
                  has_re, diopters, severity, input_data
           FROM screenings WHERE email = ?
           ORDER BY screened_at DESC LIMIT 1""",
        (email,)
    ).fetchone()
    conn.close()

    if not row:
        return jsonify(None), 200

    return jsonify({
        "id":          row["id"],
        "child_name":  row["child_name"],
        "screened_at": row["screened_at"],
        "risk_score":  row["risk_score"],
        "risk_level":  row["risk_level"],
        "has_re":      bool(row["has_re"]),
        "diopters":    row["diopters"],
        "severity":    row["severity"],
        "input_data":  json.loads(row["input_data"]) if row["input_data"] else {},
    }), 200
