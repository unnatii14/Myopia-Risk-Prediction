import sys
from pathlib import Path

from flask import Flask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auth


def create_test_client(tmp_path):
    test_db = tmp_path / "users_test.db"
    auth.DB_PATH = str(test_db)
    auth._init_db()

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(auth.auth_bp, url_prefix="/auth")
    return app.test_client()


def test_signup_and_login_success(tmp_path):
    client = create_test_client(tmp_path)

    signup_payload = {
        "name": "Test User",
        "childName": "Kid",
        "email": "test@example.com",
        "password": "Password123!",
    }

    signup_response = client.post("/auth/signup", json=signup_payload)
    assert signup_response.status_code == 201
    signup_data = signup_response.get_json()
    assert "token" in signup_data
    assert signup_data["email"] == "test@example.com"

    login_response = client.post(
        "/auth/login",
        json={"email": "test@example.com", "password": "Password123!"},
    )
    assert login_response.status_code == 200
    login_data = login_response.get_json()
    assert "token" in login_data
    assert login_data["name"] == "Test User"


def test_signup_duplicate_email_returns_conflict(tmp_path):
    client = create_test_client(tmp_path)

    payload = {
        "name": "Duplicate User",
        "childName": "Kid",
        "email": "dupe@example.com",
        "password": "Password123!",
    }

    first = client.post("/auth/signup", json=payload)
    assert first.status_code == 201

    second = client.post("/auth/signup", json=payload)
    assert second.status_code == 409
    assert "error" in second.get_json()


def test_login_invalid_password_returns_unauthorized(tmp_path):
    client = create_test_client(tmp_path)

    signup_payload = {
        "name": "Auth User",
        "childName": "Kid",
        "email": "auth@example.com",
        "password": "Password123!",
    }
    client.post("/auth/signup", json=signup_payload)

    login_response = client.post(
        "/auth/login",
        json={"email": "auth@example.com", "password": "wrong-password"},
    )
    assert login_response.status_code == 401
    assert "error" in login_response.get_json()
