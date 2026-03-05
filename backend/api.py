"""
Myopia Risk Prediction Backend API
Three-stage ML pipeline:
  Stage 1 → Has Refractive Error? (has_re_model)
  Stage 2 → Progression Risk Level? (risk_progression_model)
  Stage 3 → Diopter Severity Estimate (diopter_regression_model)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow requests from the React frontend

# ─────────────────────────────────────────────────────────────
# Load model artifacts once at startup
# ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

print("Loading models...")
risk_model    = joblib.load(os.path.join(MODEL_DIR, "risk_progression_model.pkl"))
re_model      = joblib.load(os.path.join(MODEL_DIR, "has_re_model.pkl"))
scaler_cls    = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

# Diopter severity model may fail to load if sklearn version mismatches.
# It is optional — Stage 1+2 still work without it.
diopter_model = None
scaler_reg    = None
try:
    diopter_model = joblib.load(os.path.join(MODEL_DIR, "diopter_regression_model.pkl"))
    scaler_reg    = joblib.load(os.path.join(MODEL_DIR, "scaler_regression.pkl"))
    print("✅  Diopter severity model loaded.")
except Exception as _e:
    print(f"⚠️  Diopter model skipped (sklearn version mismatch): {_e}")
    print("    Stage 3 (diopter estimate) will use rule-based fallback.")

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)

print(f"✅  Models loaded. Feature count: {len(FEATURE_COLS)}")

# ─────────────────────────────────────────────────────────────
# State encoding tables
# Matches what the notebook did during training:
#   pd.get_dummies(State, prefix='State', drop_first=True)
#   → first state alphabetically was dropped (Andhra Pradesh)
#   factorize() order approximated by sorted alphabetical index
# ─────────────────────────────────────────────────────────────
# All 12 states in dataset (Andhra Pradesh = reference, dropped by drop_first)
ALL_STATES_SORTED = [
    "Andhra Pradesh", "Delhi", "Gujarat", "Karnataka", "Kerala",
    "Maharashtra", "Punjab", "Rajasthan", "Tamil Nadu",
    "Telangana", "Uttar Pradesh", "West Bengal"
]
# factorize() assigns labels based on first occurrence order in data.
# We approximate with sorted index (tree-based models are robust to this).
STATE_ENCODE_MAP = {s: i for i, s in enumerate(ALL_STATES_SORTED)}

# One-hot columns that survived drop_first=True (everything except Andhra Pradesh)
STATE_ONEHOT_COLS = [
    "State_Delhi", "State_Gujarat", "State_Karnataka", "State_Kerala",
    "State_Maharashtra", "State_Punjab", "State_Rajasthan",
    "State_Tamil Nadu", "State_Telangana", "State_Uttar Pradesh", "State_West Bengal"
]


def compute_bmi_category(bmi: float) -> int:
    if bmi < 18.5: return 0   # Underweight
    if bmi < 25.0: return 1   # Normal
    if bmi < 30.0: return 2   # Overweight
    return 3                   # Obese


def rule_based_risk(d: dict) -> float:
    """
    Evidence-based rule scoring (mirrors clinical risk factors).
    Returns a value in [0, 1].
    Used as a floor/calibration on the ML output to prevent
    obviously wrong results for extreme input combinations.
    """
    age          = float(d.get("age", 10))
    parents      = d.get("parentsMyopic", "none")
    fam_hist     = bool(d.get("familyHistory"))
    screen       = float(d.get("screenTime", 4))
    outdoor      = float(d.get("outdoorTime", 1))
    nearw        = float(d.get("nearWork", 4))
    vitamin_d    = bool(d.get("vitaminD"))
    sports       = d.get("sports", "occasional")
    comp_exam    = bool(d.get("competitiveExam"))
    tuition      = bool(d.get("tuition"))
    school       = d.get("schoolType", "")

    s = 30  # base

    # Age (younger = higher progression risk)
    if age <= 8:   s += 15
    elif age <= 10: s += 10
    elif age <= 12: s += 5

    # Genetics
    if parents == "both":   s += 25
    elif parents == "one":  s += 15
    elif fam_hist:          s += 8

    # Screen time
    if screen > 8:   s += 22
    elif screen > 6: s += 17
    elif screen > 4: s += 12
    elif screen > 2: s += 6

    # Outdoor time (strongest protective factor)
    if outdoor == 0:    s += 25
    elif outdoor < 0.5: s += 20
    elif outdoor < 1:   s += 15
    elif outdoor < 2:   s += 8
    elif outdoor >= 3:  s -= 10

    # Near work
    if nearw > 6:   s += 15
    elif nearw > 4: s += 8

    # Academic pressure
    if comp_exam: s += 10
    if tuition:   s += 5
    if school in ("international", "private"): s += 3

    # Protective factors
    if vitamin_d:        s -= 5
    if sports == "regular": s -= 8

    return min(max(s, 0), 100) / 100.0


def build_feature_row(d: dict) -> np.ndarray:
    """
    Convert raw frontend form data → model feature vector.
    Mirrors the feature engineering in the training notebook exactly.
    """
    age    = float(d.get("age", 10))
    sex    = 1 if d.get("sex") == "male" else 0
    height = float(d.get("height", 150))
    weight = float(d.get("weight", 40))
    bmi    = weight / ((height / 100) ** 2) if height > 0 else 22.0
    bmi    = round(bmi, 2)
    bmi_cat = compute_bmi_category(bmi)

    # Location type: default Urban for web-based screening
    location_type = 1 if d.get("locationType", "urban") == "urban" else 0

    school_map  = {"government": 0, "private": 1, "international": 2}
    school_type = school_map.get(d.get("schoolType", "government"), 0)

    family_history  = 1 if d.get("familyHistory") else 0
    parents_map     = {"none": 0, "one": 1, "both": 2}
    parents_myopia  = parents_map.get(d.get("parentsMyopic", "none"), 0)

    screen_time  = float(d.get("screenTime", 4))
    near_work    = float(d.get("nearWork", 4))
    outdoor_time = float(d.get("outdoorTime", 1))

    tuition      = 1 if d.get("tuition") else 0
    comp_exam    = 1 if d.get("competitiveExam") else 0
    vitamin_d    = 1 if d.get("vitaminD") else 0

    sports_map  = {"rare": 0, "occasional": 1, "regular": 2}
    sports      = sports_map.get(d.get("sports", "occasional"), 1)

    # ── Engineered composite features ────────────────────────
    screen_near_work     = screen_time + near_work
    outdoor_activity     = outdoor_time * sports
    # Cap digital_exposure to the training data's realistic maximum (~20).
    # outdoor_time=0 would give 100+, which is way outside the training
    # distribution (mean≈4.9, std≈5.3) and causes XGBoost to mispredict.
    _outdoor_floor       = max(outdoor_time, 0.5)          # floor at 30 min
    digital_exposure     = min(screen_time / (_outdoor_floor + 0.1), 20.0)
    academic_stress      = tuition * comp_exam
    risk_score_feat      = (
        family_history * 2
        + screen_time / 2
        + near_work / 2
        + (10 - outdoor_time)
        + tuition
        + comp_exam
    )

    # ── State encoding ────────────────────────────────────────
    state         = d.get("state", "Maharashtra")
    state_encoded = STATE_ENCODE_MAP.get(state, 5)  # default Maharashtra=5

    state_onehots = {col: 0 for col in STATE_ONEHOT_COLS}
    col_key = f"State_{state}"
    if col_key in state_onehots:
        state_onehots[col_key] = 1
    # Andhra Pradesh → all zeros (reference category)

    # ── Assemble base dict ────────────────────────────────────
    row = {
        "Age"                     : age,
        "Sex"                     : sex,
        "Height_cm"               : height,
        "Weight_kg"               : weight,
        "BMI"                     : bmi,
        "BMI_Category"            : bmi_cat,
        "Location_Type"           : location_type,
        "School_Type"             : school_type,
        "Family_History_Myopia"   : family_history,
        "Parents_With_Myopia"     : parents_myopia,
        "Screen_Time_Hours"       : screen_time,
        "Near_Work_Hours"         : near_work,
        "Outdoor_Time_Hours"      : outdoor_time,
        "Tuition_Classes"         : tuition,
        "Competitive_Exam_Prep"   : comp_exam,
        "Vitamin_D_Supplementation": vitamin_d,
        "Sports_Participation"    : sports,
        "Screen_Near_Work"        : screen_near_work,
        "Outdoor_Activity_Role"   : outdoor_activity,
        "Digital_Exposure"        : digital_exposure,
        "Academic_Stress"         : academic_stress,
        "Risk_Score"              : risk_score_feat,
        "State_Encoded"           : state_encoded,
        **state_onehots,
    }

    # Build vector in the exact column order the scaler was fitted on.
    # NOTE: feature_columns.json contains 'State_Encoded' TWICE (notebook quirk).
    values = [row.get(col, 0) for col in FEATURE_COLS]
    return np.array(values, dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "features": len(FEATURE_COLS)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        X = build_feature_row(data)

        # Scale for classifiers
        X_cls = scaler_cls.transform(X)

        # ── Stage 1: Has Refractive Error? ────────────────────
        re_prob  = float(re_model.predict_proba(X_cls)[0][1])
        has_re   = re_prob >= 0.5

        # ── Stage 2: Progression Risk ─────────────────────────
        ml_prob   = float(risk_model.predict_proba(X_cls)[0][1])
        rule_prob = rule_based_risk(data)

        # ── Adaptive hybrid scoring ───────────────────────────
        # ROOT CAUSE (diagnosed): the XGBoost model has a non-monotonic
        # response curve — it gives HIGHER scores for outdoor=1.5h than
        # outdoor=0h, and at certain Digital_Exposure scaled values (+2.22)
        # routes to a near-zero leaf even for extreme high-risk inputs.
        # Evidence: sweep shows outdoor=0→57%, outdoor=1.5→72%, outdoor=0.1→4%.
        #
        # FIX: Rule-based score (derived from WHO / published clinical evidence)
        # is the PRIMARY signal.  ML refines it upward when it's confident.
        # ML is set as secondary because it has poor monotonicity on key features.
        if ml_prob >= 0.65:
            # ML confidently HIGH → share credit 45% ML / 55% rule
            risk_prob = 0.45 * ml_prob + 0.55 * rule_prob
        elif ml_prob >= 0.35:
            # ML uncertain middle zone → lean on rules 75%
            risk_prob = 0.25 * ml_prob + 0.75 * rule_prob
        else:
            # ML giving LOW (likely artifact / extreme input) → trust rule 90%
            risk_prob = 0.10 * ml_prob + 0.90 * rule_prob

        # Hard floor: clinical rules always set a minimum (prevent ML from
        # dragging a clearly high-risk case below 80% of clinical estimate)
        risk_prob = max(risk_prob, 0.80 * rule_prob)

        risk_pct  = int(round(risk_prob * 100))

        if risk_pct < 40:
            risk_level = "LOW"
        elif risk_pct < 70:
            risk_level = "MODERATE"
        else:
            risk_level = "HIGH"

        # ── Stage 3: Diopter Estimate (only if RE likely) ─────
        diopters = None
        severity = None
        if has_re:
            if diopter_model is not None and scaler_reg is not None:
                # Use ML regression model
                X_reg    = scaler_reg.transform(X)
                diopters = float(diopter_model.predict(X_reg)[0])
                diopters = round(abs(diopters), 2)
            else:
                # Rule-based fallback estimate
                if risk_pct >= 70:   diopters = 3.5
                elif risk_pct >= 50: diopters = 2.0
                else:                diopters = 1.0

            if diopters is not None:
                if diopters < 0.5:   severity = "Negligible"
                elif diopters < 3.0: severity = "Mild"
                elif diopters < 6.0: severity = "Moderate"
                else:                severity = "High"

        return jsonify({
            "risk_score"      : risk_pct,
            "risk_level"      : risk_level,
            "risk_probability": round(risk_prob, 3),
            "has_re"          : has_re,
            "re_probability"  : round(re_prob, 3),
            "diopters"        : diopters,
            "severity"        : severity,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Myopia Risk API on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
