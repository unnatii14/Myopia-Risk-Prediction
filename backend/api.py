"""
Myopia Risk Prediction Backend API
Three-stage ML pipeline:
  Stage 1 → Has Refractive Error? (has_re_model)
  Stage 2 → Progression Risk Level? (risk_progression_model)
  Stage 3 → Diopter Severity Estimate (diopter_regression_model)
"""

from flask import Flask, request, jsonify, g
from flask_cors import CORS
import joblib
import json
import os
import time
import numpy as np
import pandas as pd
from logger import setup_logger, RequestLogger
from auth import auth_bp

app = Flask(__name__)
CORS(app)  # Allow requests from the React frontend
app.register_blueprint(auth_bp, url_prefix="/auth")

# Setup logging
logger = setup_logger('api', log_file='logs/api.log', level='INFO')
request_logger = RequestLogger(logger)

# ─────────────────────────────────────────────────────────────
# Load model artifacts once at startup
# ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

print("Loading models...")
risk_model    = joblib.load(os.path.join(MODEL_DIR, "risk_progression_model.pkl"))
scaler_cls    = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

# Load IMPROVED Stage 1 model (AUC 0.94!)
re_model      = joblib.load(os.path.join(MODEL_DIR, "has_re_model_improved.pkl"))
re_scaler     = joblib.load(os.path.join(MODEL_DIR, "has_re_scaler.pkl"))

with open(os.path.join(MODEL_DIR, "has_re_features.json")) as f:
    RE_FEATURE_META = json.load(f)
    RE_FEATURE_COLS = RE_FEATURE_META['feature_columns']
    
print(f"[OK]  Stage 1 (Has_RE) model: {RE_FEATURE_META.get('model_type','XGBoost')}  AUC={RE_FEATURE_META['metrics']['auc']:.4f}")

# Diopter severity model may fail to load if sklearn version mismatches.
# It is optional — Stage 1+2 still work without it.
diopter_model = None
scaler_reg    = None
try:
    diopter_model = joblib.load(os.path.join(MODEL_DIR, "diopter_regression_model.pkl"))
    scaler_reg    = joblib.load(os.path.join(MODEL_DIR, "scaler_regression.pkl"))
    print("[OK]  Diopter severity model loaded.")
except Exception as _e:
    print(f"[WARN]  Diopter model skipped (sklearn version mismatch): {_e}")
    print("    Stage 3 (diopter estimate) will use rule-based fallback.")

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)

print(f"[OK]  Models loaded. Feature count: {len(FEATURE_COLS)}")

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


def build_stage1_features(d: dict) -> np.ndarray:
    """
    Build feature vector for IMPROVED Stage 1 (Has_RE) model.
    Includes enhanced interaction features that boosted AUC 0.50 → 0.94.
    """
    age          = float(d.get("age", 10))
    height       = float(d.get("height", 150))
    weight       = float(d.get("weight", 40))
    bmi          = weight / ((height / 100) ** 2) if height > 0 else 22.0
    bmi          = round(bmi, 2)
    
    screen_time  = float(d.get("screenTime", 4))
    near_work    = float(d.get("nearWork", 4))
    outdoor_time = float(d.get("outdoorTime", 1))
    
    family_hist  = 1 if d.get("familyHistory") else 0
    parents_map  = {"none": 0, "one": 1, "both": 2}
    parents_myopic = parents_map.get(d.get("parentsMyopic", "none"), 0)
    
    location_urban = 1 if d.get("locationType", "urban") == "urban" else 0
    school_map     = {"government": 0, "private": 1, "international": 2}
    school_type    = school_map.get(d.get("schoolType", "government"), 0)
    
    tuition       = 1 if d.get("tuition") else 0
    comp_exam     = 1 if d.get("competitiveExam") else 0
    vitamin_d     = 1 if d.get("vitaminD") else 0
    
    sports_map = {"rare": 0, "occasional": 1, "regular": 2}
    sports     = sports_map.get(d.get("sports", "occasional"), 1)
    
    # ── ENHANCED FEATURES (THE BREAKTHROUGH!) ────────────────
    # These interaction features improved AUC from 0.50 → 0.94
    age_screen           = age * screen_time
    screen_near_total    = screen_time + near_work
    outdoor_deficit      = max(0, 2 - outdoor_time)
    screen_outdoor_ratio = screen_time / (outdoor_time + 0.1)
    
    # Family/genetic load
    family_load = parents_myopic * 2 + family_hist
    
    # State encoding
    state = d.get("state", "Maharashtra")
    state_onehots = {col: 0 for col in STATE_ONEHOT_COLS}
    col_key = f"State_{state}"
    if col_key in state_onehots:
        state_onehots[col_key] = 1
    
    # Assemble feature dict in order
    row = {
        'Age': age,
        'BMI': bmi,
        'Screen_Time_Hours': screen_time,
        'Near_Work_Hours': near_work,
        'Outdoor_Time_Hours': outdoor_time,
        
        # Enhanced features
        'Age_Screen': age_screen,
        'Screen_Near_Total': screen_near_total,
        'Outdoor_Deficit': outdoor_deficit,
        'Screen_Outdoor_Ratio': screen_outdoor_ratio,
        
        # Family/genetic features
        'High_Risk_Parent': parents_myopic,
        'Family_History_Binary': family_hist,
        'Family_Load': family_load,
        
        # Encoded categorical
        'Location_Type_Urban': location_urban,
        'School_Type_Encoded': school_type,
        'Tuition_Binary': tuition,
        'Comp_Exam_Binary': comp_exam,
        'Vitamin_D_Binary': vitamin_d,
        'Sports_Encoded': sports,
        
        **state_onehots,
    }
    
    # Build vector in exact order
    values = [row.get(col, 0) for col in RE_FEATURE_COLS]
    return np.array(values, dtype=float).reshape(1, -1)


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
    Build feature vector for Stage 2 (Risk Progression) model.
    Matches risk_cols used in retrain_improved.py — GradientBoosting model.
    """
    age          = float(d.get("age", 10))
    height       = float(d.get("height", 150))
    weight       = float(d.get("weight", 40))
    bmi          = weight / ((height / 100) ** 2) if height > 0 else 22.0
    bmi          = round(bmi, 2)
    bmi_cat      = compute_bmi_category(bmi)

    screen_time  = float(d.get("screenTime", 4))
    near_work    = float(d.get("nearWork", 4))
    outdoor_time = float(d.get("outdoorTime", 1))

    family_hist    = 1 if d.get("familyHistory") else 0
    parents_map    = {"none": 0, "one": 1, "both": 2}
    parents_myopic = parents_map.get(d.get("parentsMyopic", "none"), 0)

    location_urban = 1 if d.get("locationType", "urban") == "urban" else 0
    school_map     = {"government": 0, "private": 1, "international": 2}
    school_type    = school_map.get(d.get("schoolType", "government"), 0)

    tuition      = 1 if d.get("tuition") else 0
    comp_exam    = 1 if d.get("competitiveExam") else 0
    vitamin_d    = 1 if d.get("vitaminD") else 0

    sports_map = {"rare": 0, "occasional": 1, "regular": 2}
    sports     = sports_map.get(d.get("sports", "occasional"), 1)

    # ── Enhanced interaction features (same as retrain_improved.py) ──
    age_screen           = age * screen_time
    screen_near_total    = screen_time + near_work
    outdoor_deficit      = max(0, 2 - outdoor_time)
    screen_outdoor_ratio = screen_time / (outdoor_time + 0.1)

    # Genetic / family load
    high_risk_parent = 1 if parents_myopic >= 2 else 0
    family_load      = parents_myopic * 2 + family_hist

    # ── State one-hots ────────────────────────────────────────
    state = d.get("state", "Maharashtra")
    state_onehots = {col: 0 for col in STATE_ONEHOT_COLS}
    col_key = f"State_{state}"
    if col_key in state_onehots:
        state_onehots[col_key] = 1

    row = {
        "Age"                 : age,
        "BMI"                 : bmi,
        "BMI_Category_Encoded": bmi_cat,
        "Screen_Time_Hours"   : screen_time,
        "Near_Work_Hours"     : near_work,
        "Outdoor_Time_Hours"  : outdoor_time,
        "Age_Screen"          : age_screen,
        "Screen_Near_Total"   : screen_near_total,
        "Outdoor_Deficit"     : outdoor_deficit,
        "Screen_Outdoor_Ratio": screen_outdoor_ratio,
        "High_Risk_Parent"    : high_risk_parent,
        "Family_History_Binary": family_hist,
        "Family_Load"         : family_load,
        "Location_Type_Urban" : location_urban,
        "School_Type_Encoded" : school_type,
        "Tuition_Binary"      : tuition,
        "Comp_Exam_Binary"    : comp_exam,
        "Vitamin_D_Binary"    : vitamin_d,
        "Sports_Encoded"      : sports,
        **state_onehots,
    }

    values = [row.get(col, 0) for col in FEATURE_COLS]
    return np.array(values, dtype=float).reshape(1, -1)


# ─────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────

@app.before_request
def before_request():
    """Log request start time"""
    g.start_time = time.time()
    request_logger.log_request(request)


@app.after_request
def after_request(response):
    """Log response details"""
    if hasattr(g, 'start_time'):
        duration_ms = (time.time() - g.start_time) * 1000
        request_logger.log_response(response, duration_ms)
    return response


@app.route("/health", methods=["GET"])
def health():
    logger.info("Health check endpoint called")
    return jsonify({"status": "ok", "features": len(FEATURE_COLS)})


@app.route("/", methods=["GET"])
def index():
    """Basic index route to help manual browser checks."""
    return jsonify(
        {
            "service": "Myopia Risk API",
            "status": "ok",
            "routes": ["/health", "/predict", "/auth/signup", "/auth/login"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        # Handle JSON parsing errors gracefully
        try:
            data = request.get_json(force=True)
        except Exception as json_err:
            logger.error(f"JSON parsing error: {json_err}")
            return jsonify({"error": "Invalid JSON format"}), 400
            
        if not data:
            logger.warning("Empty JSON body received")
            return jsonify({"error": "No JSON body received"}), 400
        
        logger.info(f"Prediction request received for age={data.get('age')}, sex={data.get('sex')}")
        
        # ── Input Validation ──
        from validation import validate_screening_data, ValidationError
        try:
            validated_data = validate_screening_data(data)
            logger.debug("Input validation passed")
        except ValidationError as ve:
            logger.warning(f"Validation failed: {str(ve)}")
            return jsonify({"error": f"Validation failed: {str(ve)}"}), 400
        
        # Use validated data for processing
        data = validated_data

        # Build features for Stage 2 (risk progression)
        X = build_feature_row(data)
        X_cls = scaler_cls.transform(X)

        # ── Stage 1: Has Refractive Error? (IMPROVED MODEL) ───
        X_re = build_stage1_features(data)
        X_re_scaled = re_scaler.transform(X_re)
        re_prob  = float(re_model.predict_proba(X_re_scaled)[0][1])
        has_re   = re_prob >= 0.5

        # ── Stage 2: Progression Risk ─────────────────────────
        ml_prob   = float(risk_model.predict_proba(X_cls)[0][1])
        rule_prob = rule_based_risk(data)

        # ── Adaptive hybrid scoring ───────────────────────────
        # Model: GradientBoosting (AUC=0.893). More reliable than old XGBoost.
        # Hybrid still used since clinical rules encode WHO evidence directly.
        if ml_prob >= 0.65:
            # ML confidently HIGH → trust ML 60%, rule 40%
            risk_prob = 0.60 * ml_prob + 0.40 * rule_prob
        elif ml_prob >= 0.35:
            # ML uncertain → balanced 50/50
            risk_prob = 0.50 * ml_prob + 0.50 * rule_prob
        else:
            # ML giving LOW → lean on rules 80%
            risk_prob = 0.20 * ml_prob + 0.80 * rule_prob

        # Hard floor: clinical rules set a minimum
        risk_prob = max(risk_prob, 0.75 * rule_prob)

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
                # IMPORTANT: Regression model uses subset of 27 features (not all 30)
                # These match what retrain_all_models.py trained on
                regression_feature_names = [
                    'Age', 'BMI', 'Screen_Time_Hours', 'Near_Work_Hours', 'Outdoor_Time_Hours',
                    'Age_Screen', 'Screen_Near_Total', 'Screen_Outdoor_Ratio',
                    'High_Risk_Parent', 'Family_Load',
                    'Location_Type_Urban', 'School_Type_Encoded',
                    'Tuition_Binary', 'Comp_Exam_Binary', 'Vitamin_D_Binary', 'Sports_Encoded',
                    'State_Delhi', 'State_Gujarat', 'State_Karnataka', 'State_Kerala',
                    'State_Maharashtra', 'State_Punjab', 'State_Rajasthan', 'State_Tamil Nadu',
                    'State_Telangana', 'State_Uttar Pradesh', 'State_West Bengal'
                ]
                
                # Build feature dict
                row_dict = {}
                age    = float(data.get("age", 10))
                height = float(data.get("height", 150))
                weight = float(data.get("weight", 40))
                bmi    = weight / ((height / 100) ** 2) if height > 0 else 22.0
                
                screen_time  = float(data.get("screenTime", 4))
                near_work    = float(data.get("nearWork", 4))
                outdoor_time = float(data.get("outdoorTime", 1))
                
                family_history = 1 if data.get("familyHistory") else 0
                parents_map    = {"none": 0, "one": 1, "both": 2}
                parents_myopia = parents_map.get(data.get("parentsMyopic", "none"), 0)
                
                row_dict['Age'] = age
                row_dict['BMI'] = round(bmi, 2)
                row_dict['Screen_Time_Hours'] = screen_time
                row_dict['Near_Work_Hours'] = near_work
                row_dict['Outdoor_Time_Hours'] = outdoor_time
                row_dict['Age_Screen'] = age * screen_time
                row_dict['Screen_Near_Total'] = screen_time + near_work
                _outdoor_floor = max(outdoor_time, 0.1)
                row_dict['Screen_Outdoor_Ratio'] = screen_time / _outdoor_floor
                row_dict['High_Risk_Parent'] = 1 if parents_myopia >= 2 else 0
                row_dict['Family_Load'] = family_history + parents_myopia
                row_dict['Location_Type_Urban'] = 1 if data.get("locationType", "urban") == "urban"  else 0
                
                school_map = {"government": 0, "private": 1, "international": 2}
                row_dict['School_Type_Encoded'] = school_map.get(data.get("schoolType", "government"), 0)
                row_dict['Tuition_Binary'] = 1 if data.get("tuition") else 0
                row_dict['Comp_Exam_Binary'] = 1 if data.get("competitiveExam") else 0
                row_dict['Vitamin_D_Binary'] = 1 if data.get("vitaminD") else 0
                
                sports_map = {"rare": 0, "occasional": 1, "regular": 2}
                row_dict['Sports_Encoded'] = sports_map.get(data.get("sports", "occasional"), 1)
                
                # State one-hot
                state = data.get("state", "Maharashtra")
                for state_col in regression_feature_names:
                    if state_col.startswith("State_"):
                        state_name = state_col.replace("State_", "")
                        row_dict[state_col] = 1 if state == state_name else 0
                
                # Build array in correct order
                X_reg_values = [row_dict.get(col, 0) for col in regression_feature_names]
                X_reg_input = np.array(X_reg_values, dtype=float).reshape(1, -1)
                
                X_reg_scaled = scaler_reg.transform(X_reg_input)
                diopters = float(diopter_model.predict(X_reg_scaled)[0])
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

        result = {
            "risk_score"      : risk_pct,
            "risk_level"      : risk_level,
            "risk_probability": round(risk_prob, 3),
            "has_re"          : has_re,
            "re_probability"  : round(re_prob, 3),
            "diopters"        : diopters,
            "severity"        : severity,
        }
        
        # Log prediction details
        duration_ms = (time.time() - start_time) * 1000
        request_logger.log_prediction(data, result, duration_ms)
        
        logger.info(
            f"Prediction complete: Risk={risk_level} ({risk_pct}%), "
            f"Has_RE={has_re} ({re_prob:.3f}), "
            f"Diopters={diopters}, Duration={duration_ms:.2f}ms"
        )
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Prediction error: {type(e).__name__}: {str(e)}", exc_info=True)
        request_logger.log_error(e, context="predict endpoint")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Myopia Risk API on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
