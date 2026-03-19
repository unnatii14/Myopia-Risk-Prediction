"""
Authentication Blueprint — /register and /login
Uses SQLite for user storage, bcrypt for password hashing, JWT for tokens.
"""

from flask import Blueprint, request, jsonify
import sqlite3
import bcrypt
import jwt
import datetime
import os
from google.auth.transport import requests
from google.oauth2 import id_token

auth_bp = Blueprint("auth", __name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "users.db")
JWT_SECRET = os.environ.get("JWT_SECRET", "myopia_dev_secret_key_2024")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            child_name    TEXT,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

_init_db()


def _make_token(name: str, email: str) -> str:
    payload = {
        "name" : name,
        "email": email,
        "exp"  : datetime.datetime.utcnow() + datetime.timedelta(days=30),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


@auth_bp.route("/signup", methods=["POST"])
def register():
    data       = request.get_json() or {}
    name       = (data.get("name") or "").strip()
    child_name = (data.get("childName") or "").strip()
    email      = (data.get("email") or "").strip().lower()
    password   = data.get("password") or ""

    if not name or not email or not password:
        return jsonify({"error": "Name, email and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO users (name, child_name, email, password_hash) VALUES (?, ?, ?, ?)",
            (name, child_name or None, email, pw_hash),
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409

    token = _make_token(name, email)
    return jsonify({"token": token, "name": name, "email": email}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data     = request.get_json() or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if not row or not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
        return jsonify({"error": "Invalid email or password"}), 401

    token = _make_token(row["name"], email)
    return jsonify({"token": token, "name": row["name"], "email": email}), 200


@auth_bp.route("/google", methods=["POST"])
def google_login():
    """
    Verify Google ID token and create/retrieve user.
    Frontend sends the Google JWT token from the GoogleLogin component.
    """
    data = request.get_json() or {}
    google_token = data.get("token")

    if not google_token:
        return jsonify({"error": "Google token is required"}), 400

    try:
        # Verify the Google JWT token
        GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
        if not GOOGLE_CLIENT_ID:
            return jsonify({"error": "Google Client ID not configured"}), 500

        # Verify the token signature and get claims
        idinfo = id_token.verify_oauth2_token(
            google_token,
            requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Extract user info from the verified token
        email = idinfo.get("email", "").strip().lower()
        name = idinfo.get("name", "")
        google_id = idinfo.get("sub", "")

        if not email:
            return jsonify({"error": "Email not found in Google token"}), 400

        # Check if user exists, if not create them
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if not user:
            # Create new user with a placeholder password hash for OAuth users
            # This ensures OAuth users can login via Google even if they try email/password
            placeholder_hash = bcrypt.hashpw(
                google_id.encode(),
                bcrypt.gensalt()
            ).decode()
            try:
                conn.execute(
                    "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                    (name, email, placeholder_hash),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                conn.close()
                return jsonify({"error": "Failed to create user"}), 500

        conn.close()

        # Generate JWT token for our app
        token = _make_token(name, email)
        return jsonify({
            "token": token,
            "name": name,
            "email": email
        }), 200

    except ValueError as e:
        # Invalid token
        return jsonify({"error": f"Invalid Google token: {str(e)}"}), 401
    except Exception as e:
        return jsonify({"error": f"Authentication failed: {str(e)}"}), 500
