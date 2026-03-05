import warnings; warnings.filterwarnings("ignore")
import joblib, json, numpy as np, os, sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
risk_model = joblib.load(os.path.join(MODEL_DIR, "risk_progression_model.pkl"))
scaler_cls = joblib.load(os.path.join(MODEL_DIR, "scaler_classification.pkl"))

with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLS = json.load(f)

idx_de = FEATURE_COLS.index("Digital_Exposure")
print(f"Digital_Exposure index: {idx_de}")
print(f"Scaler mean[{idx_de}]: {scaler_cls.mean_[idx_de]:.4f}")
print(f"Scaler scale[{idx_de}]: {scaler_cls.scale_[idx_de]:.4f}")

def test(label, age, screen, nearw, outdoor, parents=2, fam=1, tuition=1, comp=1, sports=0):
    _outdoor_floor = max(outdoor, 0.5)
    de = min(screen / (_outdoor_floor + 0.1), 20.0)   # same formula as fixed API
    row = [age, 1, 140, 35, 17.8, 0, 1, 1, fam, parents, screen, nearw, outdoor,
           tuition, comp, 0, sports, screen+nearw, outdoor*sports, de, tuition*comp,
           fam*2+screen/2+nearw/2+(10-outdoor)+tuition+comp, 5, 5,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    X = np.array(row, dtype=float).reshape(1, -1)
    Xs = scaler_cls.transform(X)
    scaled_de = Xs[0][idx_de]
    prob = risk_model.predict_proba(Xs)[0][1]
    print(f"\n{label}")
    print(f"  digital_exp raw={de:.1f}  scaled={scaled_de:.2f}  -> prob={prob:.3f}  score={round(prob*100)}%")

test("TEST A: outdoor=0.5 screen=8  (first test)", 12, 8, 5, 0.5)
test("TEST B: outdoor=0   screen=10 (second test - extreme)", 8, 10, 8, 0.0)
test("TEST C: outdoor=1   screen=8", 10, 8, 5, 1.0)
test("TEST D: outdoor=2   screen=4  (low risk)", 14, 4, 2, 2.0, parents=0, fam=0, tuition=0, comp=0, sports=2)
test("TEST E: outdoor=0.1 screen=8  (near zero outdoor)", 8, 8, 6, 0.1)
