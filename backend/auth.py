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
import smtplib
import ssl
from email.message import EmailMessage
import random
import secrets
from google.auth.transport import requests
from google.oauth2 import id_token
from config import DEFAULT_JWT_SECRET

auth_bp = Blueprint("auth", __name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.path.join(BASE_DIR, "users.db")
JWT_SECRET = os.environ.get("JWT_SECRET", "myopia_dev_secret_key_2024")

if os.environ.get("FLASK_ENV", "development").lower() == "production" and JWT_SECRET == DEFAULT_JWT_SECRET:
    raise RuntimeError("JWT_SECRET must be set to a non-default value in production")


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
    # OTP storage for email-based login (one-time codes)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS otps (
            email       TEXT PRIMARY KEY,
            otp_hash    TEXT NOT NULL,
            expires_at  TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)
        # Password reset tokens
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                email       TEXT PRIMARY KEY,
                token_hash  TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
    conn.commit()
    conn.close()

_init_db()


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
        if not email:
            return jsonify({"error": "Email not found in Google token"}), 400

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if not user:
            placeholder_hash = bcrypt.hashpw(
                os.urandom(32),
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


@auth_bp.route("/request-otp", methods=["POST"])
def request_otp():
    """Generate a one-time numeric OTP, store a hashed copy and email it to the user.
    For security return a generic 200 response so attackers can't enumerate emails.
    """
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required"}), 400

    # Generate 6-digit OTP
    otp = f"{random.randint(0, 999999):06d}"
    otp_hash = bcrypt.hashpw(otp.encode(), bcrypt.gensalt()).decode()
    expires_at = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)).isoformat()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO otps (email, otp_hash, expires_at) VALUES (?, ?, ?)", (email, otp_hash, expires_at))
        conn.commit()
        conn.close()
    except Exception:
        # Do not leak DB errors to callers
        pass

    # Attempt to send email. Failures here should not expose whether the email exists.
    try:
        smtp_host = os.environ.get("SMTP_HOST", "")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASSWORD", "")
        from_addr = os.environ.get("SMTP_FROM", os.environ.get("FROM_EMAIL", "no-reply@myopiaguard.local"))

        if smtp_host and smtp_user and smtp_pass:
            msg = EmailMessage()
            msg["Subject"] = "Your MyopiaGuard sign-in code"
            msg["From"] = from_addr
            msg["To"] = email
            msg.set_content(f"Your MyopiaGuard sign-in code is: {otp}\n\nThis code expires in 10 minutes.")

            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
    except Exception:
        # Log could be added here; swallow to avoid information leaks
        pass

    # Always return success to avoid email enumeration
    return jsonify({"status": "ok"}), 200


@auth_bp.route("/verify-otp", methods=["POST"])
def verify_otp():
    """Verify OTP for email; create user if necessary and return JWT token."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    otp = (data.get("otp") or "").strip()

    if not email or not otp:
        return jsonify({"error": "Email and OTP are required"}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT otp_hash, expires_at FROM otps WHERE email = ?", (email,)).fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "Invalid or expired code"}), 401

        expires_at = datetime.datetime.fromisoformat(row["expires_at"])
        if datetime.datetime.now(datetime.timezone.utc) > expires_at:
            conn.execute("DELETE FROM otps WHERE email = ?", (email,))
            conn.commit()
            conn.close()
            return jsonify({"error": "Code expired"}), 401

        if not bcrypt.checkpw(otp.encode(), row["otp_hash"].encode()):
            conn.close()
            return jsonify({"error": "Invalid code"}), 401

        # OTP is valid — remove it and ensure user exists
        conn.execute("DELETE FROM otps WHERE email = ?", (email,))
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            # Create a lightweight placeholder user
            placeholder_hash = bcrypt.hashpw(os.urandom(32), bcrypt.gensalt()).decode()
            display_name = email.split("@")[0]
            conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (display_name, email, placeholder_hash),
            )
            conn.commit()
            name = display_name
        else:
            name = user["name"]

        conn.close()

        token = _make_token(name, email)
        return jsonify({"token": token, "name": name, "email": email}), 200

    except Exception as e:
        return jsonify({"error": "Verification failed"}), 500


@auth_bp.route("/request-password-reset", methods=["POST"])
def request_password_reset():
    """Generate a password reset token, store it hashed and email a short link or token."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required"}), 400

    # Generate secure token (URL-safe)
    token = secrets.token_urlsafe(32)
    token_hash = bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()
    expires_at = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)).isoformat()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO password_resets (email, token_hash, expires_at) VALUES (?, ?, ?)", (email, token_hash, expires_at))
        conn.commit()
        conn.close()
    except Exception:
        pass

    # Send email with token (or link) if SMTP configured
    try:
        smtp_host = os.environ.get("SMTP_HOST", "")
        smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        smtp_user = os.environ.get("SMTP_USER", "")
        smtp_pass = os.environ.get("SMTP_PASSWORD", "")
        from_addr = os.environ.get("SMTP_FROM", os.environ.get("FROM_EMAIL", "no-reply@myopiaguard.local"))
        frontend_base = os.environ.get("FRONTEND_URL", "http://localhost:5173").rstrip('/')

        if smtp_host and smtp_user and smtp_pass:
            reset_link = f"{frontend_base}/reset-password?email={email}&token={token}"
            msg = EmailMessage()
            msg["Subject"] = "Reset your MyopiaGuard password"
            msg["From"] = from_addr
            msg["To"] = email
            msg.set_content(f"Use this link to reset your MyopiaGuard password:\n\n{reset_link}\n\nOr use this code: {token}\n\nThis link/code expires in 1 hour.")

            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls(context=context)
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
    except Exception:
        pass

    # Generic response
    return jsonify({"status": "ok"}), 200


@auth_bp.route("/reset-password", methods=["POST"])
def reset_password():
    """Verify reset token and set new password."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    token = (data.get("token") or "").strip()
    new_password = data.get("newPassword") or ""

    if not email or not token or not new_password:
        return jsonify({"error": "Email, token and new password are required"}), 400
    if len(new_password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT token_hash, expires_at FROM password_resets WHERE email = ?", (email,)).fetchone()
        if not row:
            conn.close()
            return jsonify({"error": "Invalid or expired token"}), 401

        expires_at = datetime.datetime.fromisoformat(row["expires_at"])
        if datetime.datetime.now(datetime.timezone.utc) > expires_at:
            conn.execute("DELETE FROM password_resets WHERE email = ?", (email,))
            conn.commit()
            conn.close()
            return jsonify({"error": "Token expired"}), 401

        if not bcrypt.checkpw(token.encode(), row["token_hash"].encode()):
            conn.close()
            return jsonify({"error": "Invalid token"}), 401

        # Update password
        new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, email))
        conn.execute("DELETE FROM password_resets WHERE email = ?", (email,))
        conn.commit()
        conn.close()

        return jsonify({"status": "ok"}), 200
    except Exception:
        return jsonify({"error": "Reset failed"}), 500
