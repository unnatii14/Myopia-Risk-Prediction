from api import app


def create_client():
    app.config["TESTING"] = True
    return app.test_client()


def test_predict_happy_path_returns_risk_payload():
    client = create_client()
    payload = {
        "age": 12,
        "sex": "male",
        "height": 150,
        "weight": 42,
        "screenTime": 4,
        "nearWork": 3,
        "outdoorTime": 2,
        "sports": "occasional",
        "familyHistory": True,
        "parentsMyopic": "one",
        "vitaminD": False,
        "state": "Andhra Pradesh",
        "locationType": "urban",
        "schoolType": "private",
        "tuition": False,
        "competitiveExam": False,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.get_json()
    assert "risk_score" in body
    assert "risk_level" in body
    assert "risk_probability" in body
    assert "has_re" in body
    assert "re_probability" in body


def test_predict_missing_required_fields_returns_400():
    client = create_client()

    response = client.post("/predict", json={"age": 12})
    assert response.status_code == 400

    body = response.get_json()
    assert "error" in body
