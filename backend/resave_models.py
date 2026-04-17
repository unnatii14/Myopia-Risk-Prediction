"""
Resave all .pkl models using current environment to fix cross-platform pickle issues.
Run this locally: python backend/resave_models.py
"""
import joblib
import pickle
import os
import numpy as np
import sklearn
import xgboost

print(f"Python: {__import__('sys').version}")
print(f"numpy: {np.__version__}")
print(f"sklearn: {sklearn.__version__}")
print(f"xgboost: {xgboost.__version__}")
print(f"joblib: {joblib.__version__}")
print()

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
MODEL_DIR = os.path.realpath(MODEL_DIR)
print(f"Models dir: {MODEL_DIR}")

models_to_resave = [
    'risk_progression_model.pkl',
    'scaler_classification.pkl',
    'has_re_model_improved.pkl',
    'has_re_model.pkl',
    'has_re_scaler.pkl',
    'diopter_regression_model.pkl',
    'scaler_regression.pkl',
]

for filename in models_to_resave:
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"[SKIP] {filename} — not found")
        continue
    try:
        obj = joblib.load(path)
        # Resave with protocol=4 (Python 3.8+ compatible, works cross-platform)
        joblib.dump(obj, path, protocol=4)
        print(f"[OK]   {filename} resaved")
    except Exception as e:
        print(f"[ERR]  {filename} — {e}")

print()
print("Done! Now commit the updated .pkl files and push to git.")
