"""
Authentication Blueprint — /register and /login
Uses SQLite for user storage, bcrypt for password hashing, JWT for tokens.
"""

from flask import Blueprint, request, jsonify
import bcrypt
import jwt
import datetime
import os
from google.auth.transport import requests
from google.oauth2 import id_token
from config import DEFAULT_JWT_SECRET
from db import get_conn, PH, row_as_dict, is_integrity_error, init_users_table

auth_bp = Blueprint("auth", __name__)

JWT_SECRET = os.environ.get("JWT_SECRET", "myopia_dev_secret_key_2024")

if os.environ.get("FLASK_ENV", "development").lower() == "production" and JWT_SECRET == DEFAULT_JWT_SECRET:
    raise RuntimeError("JWT_SECRET must be set to a non-default value in production")

init_users_table()


def _make_token(name: str, email: str) -> str:
    payload = {
        "name" : name,
        "email": email,
        "exp"  : datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24),
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
        conn = get_conn(); cur = conn.cursor()
        cur.execute(
            f"INSERT INTO users (name, child_name, email, password_hash) VALUES ({PH}, {PH}, {PH}, {PH})",
            (name, child_name or None, email, pw_hash),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        if is_integrity_error(e):
            return jsonify({"error": "Email already registered"}), 409
        raise

    token = _make_token(name, email)
    return jsonify({"token": token, "name": name, "email": email}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    data     = request.get_json() or {}
    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    conn = get_conn(); cur = conn.cursor()
    cur.execute(f"SELECT * FROM users WHERE email = {PH}", (email,))
    row = row_as_dict(cur)
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
        if not email:
            return jsonify({"error": "Email not found in Google token"}), 400

        conn = get_conn(); cur = conn.cursor()
        cur.execute(f"SELECT * FROM users WHERE email = {PH}", (email,))
        user = row_as_dict(cur)

        if not user:
            placeholder_hash = bcrypt.hashpw(
                os.urandom(32),
                bcrypt.gensalt()
            ).decode()
            try:
                cur.execute(
                    f"INSERT INTO users (name, email, password_hash) VALUES ({PH}, {PH}, {PH})",
                    (name, email, placeholder_hash),
                )
                conn.commit()
            except Exception:
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
