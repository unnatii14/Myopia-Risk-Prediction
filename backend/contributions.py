"""
Image contribution blueprint — lets a logged-in user donate their retinal
image to help improve the model, WITH explicit consent.

Design notes (why it works this way):
  * Consent is required and opt-in — nothing is stored without it.
  * The image must be a readable file; quality is judged by a human reviewer.
  * Each donation is saved as review_status='pending'. NOTHING is auto-added to
    the training set — a human must review/label it later. This avoids training
    on the model's own guesses (a feedback loop) or on junk uploads.
  * The user can list and delete (withdraw) their own donations (DPDP right).

Routes:
  POST   /contribute-image     — donate an image (multipart: image, consent, reported_label)
  GET    /contribute/mine      — list my donations (transparency)
  DELETE /contribute/<id>      — withdraw one of my donations
"""
from flask import Blueprint, request, jsonify
import base64
import io
import os
import jwt

from db import get_conn, PH, init_contrib_table

try:
    from PIL import Image
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False

contrib_bp = Blueprint("contrib", __name__)

JWT_SECRET = os.environ.get("JWT_SECRET", "myopia_dev_secret_key_2024")

# Bump this string whenever the consent wording changes, so we know which
# version each user agreed to.
CONSENT_VERSION = "2026-07-04.v1"

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_REPORTED = {"myopia", "normal", "unknown"}

init_contrib_table()


def _email_from_token():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload.get("email")
    except Exception:
        return None


@contrib_bp.route("/contribute-image", methods=["POST"])
def contribute_image():
    email = _email_from_token()
    if not email:
        return jsonify({"error": "You must be signed in to donate an image."}), 401

    # Consent is mandatory and must be explicitly true.
    consent_raw = (request.form.get("consent", "") or "").strip().lower()
    if consent_raw not in ("true", "1", "yes", "on"):
        return jsonify({"error": "Consent is required to donate an image."}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded (form field 'image')."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "Empty upload."}), 400

    raw = file.read()
    if len(raw) == 0:
        return jsonify({"error": "Empty image."}), 400
    if len(raw) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image too large (max 5 MB)."}), 413

    # Allow-but-warn philosophy: accept any readable image. A human reviews every
    # donation before it can be used, so we don't hard-reject on the fundus check
    # here — we just make sure the file is a valid, openable image.
    if _PIL_OK:
        try:
            Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return jsonify({"error": "Could not read the image file."}), 400

    reported = (request.form.get("reported_label", "") or "unknown").strip().lower()
    if reported not in ALLOWED_REPORTED:
        reported = "unknown"

    model_prediction = (request.form.get("model_prediction", "") or None)
    model_confidence = request.form.get("model_confidence")
    try:
        model_confidence = float(model_confidence) if model_confidence not in (None, "") else None
    except ValueError:
        model_confidence = None

    image_b64 = base64.b64encode(raw).decode("ascii")
    mime = file.mimetype or "image/png"

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""INSERT INTO contributions
            (email, image_b64, image_mime, model_prediction, model_confidence,
             reported_label, consent_version, review_status)
            VALUES ({PH},{PH},{PH},{PH},{PH},{PH},{PH},'pending')""",
        (email, image_b64, mime, model_prediction, model_confidence,
         reported, CONSENT_VERSION),
    )
    conn.commit()
    conn.close()

    return jsonify({
        "message": "Thank you — your image was submitted for review. "
                   "You can withdraw it any time from your account.",
        "status": "pending",
    }), 201


@contrib_bp.route("/contribute/mine", methods=["GET"])
def my_contributions():
    email = _email_from_token()
    if not email:
        return jsonify({"error": "Unauthorised"}), 401

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""SELECT id, model_prediction, reported_label, review_status, created_at
            FROM contributions WHERE email = {PH} ORDER BY id DESC""",
        (email,),
    )
    rows = cur.fetchall()
    conn.close()
    items = [
        {
            "id": r[0],
            "model_prediction": r[1],
            "reported_label": r[2],
            "review_status": r[3],
            "created_at": str(r[4]),
        }
        for r in rows
    ]
    return jsonify({"contributions": items})


@contrib_bp.route("/contribute/<int:contrib_id>", methods=["DELETE"])
def withdraw_contribution(contrib_id):
    email = _email_from_token()
    if not email:
        return jsonify({"error": "Unauthorised"}), 401

    conn = get_conn()
    cur = conn.cursor()
    # Only allow deleting your OWN contribution.
    cur.execute(
        f"DELETE FROM contributions WHERE id = {PH} AND email = {PH}",
        (contrib_id, email),
    )
    deleted = cur.rowcount
    conn.commit()
    conn.close()
    if deleted == 0:
        return jsonify({"error": "Not found or not yours."}), 404
    return jsonify({"message": "Your donated image was deleted."})
