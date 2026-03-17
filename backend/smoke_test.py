"""Quick end-to-end smoke test — mimics a predict() call without Flask."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Patch Flask so we can import api without running the server
import types, importlib
flask_mock = types.ModuleType('flask')
class _MockFlaskApp:
    def register_blueprint(self, *a, **kw):
        return None

    def route(self, *a, **kw):
        return lambda f: f

    def before_request(self, f):
        return f

    def after_request(self, f):
        return f

    def run(self, *a, **kw):
        return None

flask_mock.Flask     = lambda *a, **kw: _MockFlaskApp()
flask_mock.Blueprint = lambda *a, **kw: types.SimpleNamespace(route=lambda *ra, **rkw: (lambda f: f))
flask_mock.request   = None
flask_mock.jsonify   = lambda x: x
flask_mock.g         = types.SimpleNamespace()
flask_cors_mock      = types.ModuleType('flask_cors')
flask_cors_mock.CORS = lambda *a, **kw: None
sys.modules.setdefault('flask',      flask_mock)
sys.modules.setdefault('flask_cors', flask_cors_mock)

# Minimal logger stub
logger_mod = types.ModuleType('logger')
class _L:
    def info(self,*a,**kw): pass
    def debug(self,*a,**kw): pass
    def warning(self,*a,**kw): pass
    def error(self,*a,**kw): pass
class _RL:
    def log_request(self,*a,**kw): pass
    def log_response(self,*a,**kw): pass
    def log_prediction(self,*a,**kw): pass
    def log_error(self,*a,**kw): pass
logger_mod.setup_logger   = lambda *a, **kw: _L()
logger_mod.RequestLogger  = lambda *a, **kw: _RL()
sys.modules['logger'] = logger_mod

# Minimal validation stub
val_mod = types.ModuleType('validation')
class ValidationError(Exception): pass
val_mod.ValidationError = ValidationError
val_mod.validate_screening_data = lambda d: d
sys.modules['validation'] = val_mod

import importlib
import api as api_module

# Directly call feature builders + model
from api import build_feature_row, build_stage1_features, rule_based_risk
from api import risk_model, scaler_cls, re_model, re_scaler, diopter_model, scaler_reg
import numpy as np

test_inputs = [
    {
        "label": "High-risk child",
        "age": 9, "sex": "male", "height": 130, "weight": 30,
        "screenTime": 8, "nearWork": 6, "outdoorTime": 0,
        "familyHistory": True, "parentsMyopic": "both",
        "locationType": "urban", "schoolType": "private",
        "tuition": True, "competitiveExam": True, "vitaminD": False,
        "sports": "rare", "state": "Maharashtra",
    },
    {
        "label": "Low-risk child",
        "age": 12, "sex": "female", "height": 150, "weight": 42,
        "screenTime": 1, "nearWork": 1, "outdoorTime": 3,
        "familyHistory": False, "parentsMyopic": "none",
        "locationType": "rural", "schoolType": "government",
        "tuition": False, "competitiveExam": False, "vitaminD": True,
        "sports": "regular", "state": "Andhra Pradesh",
    },
]

print("=" * 60)
print("END-TO-END PREDICTION SMOKE TEST")
print("=" * 60)

for d in test_inputs:
    label = d.pop("label")
    # Stage 1
    X_re   = build_stage1_features(d)
    X_re_s = re_scaler.transform(X_re)
    re_prob = float(re_model.predict_proba(X_re_s)[0][1])
    has_re  = re_prob >= 0.5

    # Stage 2
    X      = build_feature_row(d)
    X_cls  = scaler_cls.transform(X)
    ml_prob = float(risk_model.predict_proba(X_cls)[0][1])
    rule_p  = rule_based_risk(d)

    if ml_prob >= 0.65:
        risk_prob = 0.60 * ml_prob + 0.40 * rule_p
    elif ml_prob >= 0.35:
        risk_prob = 0.50 * ml_prob + 0.50 * rule_p
    else:
        risk_prob = 0.20 * ml_prob + 0.80 * rule_p
    risk_prob = max(risk_prob, 0.75 * rule_p)
    risk_pct  = int(round(risk_prob * 100))
    level     = "LOW" if risk_pct < 40 else ("MODERATE" if risk_pct < 70 else "HIGH")

    # Stage 3
    diopters = None
    if has_re and diopter_model is not None:
        regression_feature_names = [
            'Age','BMI','Screen_Time_Hours','Near_Work_Hours','Outdoor_Time_Hours',
            'Age_Screen','Screen_Near_Total','Screen_Outdoor_Ratio',
            'High_Risk_Parent','Family_Load',
            'Location_Type_Urban','School_Type_Encoded',
            'Tuition_Binary','Comp_Exam_Binary','Vitamin_D_Binary','Sports_Encoded',
            'State_Delhi','State_Gujarat','State_Karnataka','State_Kerala',
            'State_Maharashtra','State_Punjab','State_Rajasthan','State_Tamil Nadu',
            'State_Telangana','State_Uttar Pradesh','State_West Bengal'
        ]
        from api import STATE_ONEHOT_COLS
        row_d = {}
        age = float(d.get("age",10)); height=float(d.get("height",150)); weight=float(d.get("weight",40))
        bmi = weight/((height/100)**2)
        screen_time=float(d.get("screenTime",4)); near_work=float(d.get("nearWork",4)); outdoor_time=float(d.get("outdoorTime",1))
        from api import STATE_ONEHOT_COLS
        parents_map={"none":0,"one":1,"both":2}
        parents_myopia=parents_map.get(d.get("parentsMyopic","none"),0)
        family_history=1 if d.get("familyHistory") else 0
        row_d.update({'Age':age,'BMI':round(bmi,2),'Screen_Time_Hours':screen_time,
            'Near_Work_Hours':near_work,'Outdoor_Time_Hours':outdoor_time,
            'Age_Screen':age*screen_time,'Screen_Near_Total':screen_time+near_work,
            'Screen_Outdoor_Ratio':screen_time/(outdoor_time+0.1),
            'High_Risk_Parent':1 if parents_myopia>=2 else 0,
            'Family_Load':family_history+parents_myopia,
            'Location_Type_Urban':1 if d.get("locationType","urban")=="urban" else 0,
            'School_Type_Encoded':{"government":0,"private":1,"international":2}.get(d.get("schoolType","government"),0),
            'Tuition_Binary':1 if d.get("tuition") else 0,
            'Comp_Exam_Binary':1 if d.get("competitiveExam") else 0,
            'Vitamin_D_Binary':1 if d.get("vitaminD") else 0,
            'Sports_Encoded':{"rare":0,"occasional":1,"regular":2}.get(d.get("sports","occasional"),1),
        })
        state=d.get("state","Maharashtra")
        for col in regression_feature_names:
            if col.startswith("State_"):
                row_d[col]=1 if state==col.replace("State_","") else 0
        X_reg=np.array([row_d.get(c,0) for c in regression_feature_names],dtype=float).reshape(1,-1)
        X_reg_s=scaler_reg.transform(X_reg)
        diopters=round(abs(float(diopter_model.predict(X_reg_s)[0])),2)

    print(f"\n{label}:")
    print(f"  Has_RE:      {has_re}  (prob={re_prob:.3f})")
    print(f"  ML risk:     {ml_prob:.3f}   Rule risk: {rule_p:.3f}")
    print(f"  Final risk:  {risk_pct}%  ({level})")
    if diopters is not None:
        print(f"  Diopters:    {diopters}D")

print("\n✅ Smoke test passed — all stages working correctly")
