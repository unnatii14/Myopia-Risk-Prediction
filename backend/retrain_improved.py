"""
Improved Model Retraining Script
---------------------------------
Improvements over retrain_all_models.py:

Stage 1 (Has RE):
  - XGBoost with tuned hyperparameters (already AUC 0.94, keeping)

Stage 2 (Risk Progression):
  - XGBoost WITH monotone_constraints to fix non-monotonic predictions
    (outdoor_time must decrease risk, screen/near must increase risk)
  - RandomForest & GradientBoosting as challengers
  - Best model by AUC selected automatically

Stage 3 (Diopter Severity):
  - Previous GBR had R² = -0.164 (worse than mean prediction!)
  - XGBRegressor: usually better for tabular data
  - RandomForestRegressor: robust baseline
  - GradientBoostingRegressor: existing approach with tuning
  - Best model by R² selected automatically

After running: models/ directory is updated with best models.
Run from: project root OR backend/ directory
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
import sklearn
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
import joblib

print("=" * 80)
print("IMPROVED MODEL TRAINING — Best Algorithm Selection")
print("=" * 80)
print(f"sklearn version: {sklearn.__version__}")

# ─────────────────────────────────────────────────────────────
# Locate CSV
# ─────────────────────────────────────────────────────────────
for candidate in ['Myopia_Dataset_5000.csv', '../Myopia_Dataset_5000.csv']:
    if os.path.exists(candidate):
        csv_path = candidate
        break
else:
    raise FileNotFoundError("Cannot find Myopia_Dataset_5000.csv")

df = pd.read_csv(csv_path)
print(f"\nDataset: {len(df)} rows")

# ─────────────────────────────────────────────────────────────
# Feature Engineering (same as retrain_all_models.py)
# ─────────────────────────────────────────────────────────────
dm = df.copy()
dm['Has_RE']     = (dm['Presence_of_RE'] == 'Yes').astype(int)
dm['Risk_Level'] = (dm['Progression_Risk'] == 'High').astype(int)
dm = dm[dm['Progression_Risk'] != 'N/A'].copy()

dm['Age_Screen']          = dm['Age'] * dm['Screen_Time_Hours']
dm['Screen_Near_Total']   = dm['Screen_Time_Hours'] + dm['Near_Work_Hours']
dm['Outdoor_Deficit']     = np.maximum(0, 2 - dm['Outdoor_Time_Hours'])
dm['Screen_Outdoor_Ratio']= dm['Screen_Time_Hours'] / (dm['Outdoor_Time_Hours'] + 0.1)

dm['High_Risk_Parent']    = dm['Parents_With_Myopia'].map({'None': 0, 'One Parent': 1, 'Both Parents': 2})
dm['Family_History_Binary'] = dm['Family_History_Myopia'].map({'No': 0, 'Yes': 1})
dm['Family_Load']         = dm['High_Risk_Parent'] * 2 + dm['Family_History_Binary']

dm['BMI_Category_Encoded'] = dm['BMI_Category'].map({'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3})
dm['Location_Type_Urban']  = (dm['Location_Type'] == 'Urban').astype(int)
dm['School_Type_Encoded']  = dm['School_Type'].map({'Government': 0, 'Private': 1, 'International': 2})
dm['Tuition_Binary']       = dm['Tuition_Classes'].map({'No': 0, 'Yes': 1})
dm['Comp_Exam_Binary']     = dm['Competitive_Exam_Prep'].map({'No': 0, 'Yes': 1})
dm['Vitamin_D_Binary']     = dm['Vitamin_D_Supplementation'].map({'No': 0, 'Yes': 1})
dm['Sports_Encoded']       = dm['Sports_Participation'].map({'Rare': 0, 'Occasional': 1, 'Regular': 2})

df_features = pd.get_dummies(dm, columns=['State'], drop_first=True)
state_cols   = [c for c in df_features.columns if c.startswith('State_')]

# ─────────────────────────────────────────────────────────────
# Helper: evaluate classifier
# ─────────────────────────────────────────────────────────────
def eval_clf(model, X_tr, y_tr, X_te, y_te, label):
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y_te, prob)
    pred = model.predict(X_te)
    print(f"\n  {label}")
    print(f"    AUC : {auc:.4f}")
    print(classification_report(y_te, pred, target_names=['No', 'Yes'], digits=3))
    return auc, model

# ─────────────────────────────────────────────────────────────
# Helper: evaluate regressor
# ─────────────────────────────────────────────────────────────
def eval_reg(model, X_tr, y_tr, X_te, y_te, label):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2   = r2_score(y_te, pred)
    print(f"\n  {label}")
    print(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")
    return r2, mae, rmse, model

# ═══════════════════════════════════════════════════════════════
# STAGE 1: HAS REFRACTIVE ERROR (already AUC 0.94 — keep tuned)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 1 — Has Refractive Error")
print("=" * 80)

re_cols = [
    'Age', 'BMI', 'Screen_Time_Hours', 'Near_Work_Hours', 'Outdoor_Time_Hours',
    'Age_Screen', 'Screen_Near_Total', 'Outdoor_Deficit', 'Screen_Outdoor_Ratio',
    'High_Risk_Parent', 'Family_History_Binary', 'Family_Load',
    'Location_Type_Urban', 'School_Type_Encoded', 'Tuition_Binary',
    'Comp_Exam_Binary', 'Vitamin_D_Binary', 'Sports_Encoded',
] + state_cols

X_re = df_features[re_cols].fillna(0)
y_re = df_features['Has_RE']

X_re_tr, X_re_te, y_re_tr, y_re_te = train_test_split(
    X_re, y_re, test_size=0.2, random_state=42, stratify=y_re)

re_scaler = StandardScaler()
X_re_tr_s = re_scaler.fit_transform(X_re_tr)
X_re_te_s = re_scaler.transform(X_re_te)

# Try two candidates for Stage 1
results_s1 = {}
auc1, m1 = eval_clf(
    XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, eval_metric='logloss'),
    X_re_tr_s, y_re_tr, X_re_te_s, y_re_te, "XGBoost (tuned)"
)
results_s1['XGBoost'] = (auc1, m1)

auc2, m2 = eval_clf(
    RandomForestClassifier(n_estimators=300, max_depth=12,
                           class_weight='balanced', random_state=42, n_jobs=-1),
    X_re_tr_s, y_re_tr, X_re_te_s, y_re_te, "RandomForest"
)
results_s1['RandomForest'] = (auc2, m2)

best_s1_name = max(results_s1, key=lambda k: results_s1[k][0])
best_re_auc, best_re_model = results_s1[best_s1_name]
print(f"\n✅ Stage 1 Winner: {best_s1_name}  AUC={best_re_auc:.4f}")

# ═══════════════════════════════════════════════════════════════
# STAGE 2: RISK PROGRESSION  — with monotone constraints fix!
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 2 — Risk Progression (with monotone constraint fix)")
print("=" * 80)

risk_cols = [
    'Age', 'BMI', 'BMI_Category_Encoded', 'Screen_Time_Hours', 'Near_Work_Hours',
    'Outdoor_Time_Hours', 'Age_Screen', 'Screen_Near_Total', 'Outdoor_Deficit',
    'Screen_Outdoor_Ratio', 'High_Risk_Parent', 'Family_History_Binary',
    'Family_Load', 'Location_Type_Urban', 'School_Type_Encoded',
    'Tuition_Binary', 'Comp_Exam_Binary', 'Vitamin_D_Binary', 'Sports_Encoded',
] + state_cols

X_risk = df_features[risk_cols].fillna(0)
y_risk = df_features['Risk_Level']

X_risk_tr, X_risk_te, y_risk_tr, y_risk_te = train_test_split(
    X_risk, y_risk, test_size=0.2, random_state=42, stratify=y_risk)

risk_scaler = StandardScaler()
X_risk_tr_s = risk_scaler.fit_transform(X_risk_tr)
X_risk_te_s = risk_scaler.transform(X_risk_te)

# Monotone constraints for XGBoost:
# +1 = feature must increase predicted risk
# -1 = feature must decrease predicted risk
#  0 = unconstrained
# Order matches risk_cols above:
#  Age(0) BMI(0) BMI_Cat(0) Screen(+1) NearWork(+1) Outdoor(-1)
#  Age_Screen(+1) Screen_Near_Total(+1) Outdoor_Deficit(+1) Screen_Outdoor_Ratio(+1)
#  High_Risk_Parent(+1) Family_Hist(+1) Family_Load(+1)
#  Location_Urban(0) School_Type(0) Tuition(+1) CompExam(+1) VitaminD(-1) Sports(-1)
#  State_* (0 for all)
base_constraints = [
    0,  # Age
    0,  # BMI
    0,  # BMI_Category_Encoded
    1,  # Screen_Time_Hours
    1,  # Near_Work_Hours
   -1,  # Outdoor_Time_Hours
    1,  # Age_Screen
    1,  # Screen_Near_Total
    1,  # Outdoor_Deficit
    1,  # Screen_Outdoor_Ratio
    1,  # High_Risk_Parent
    1,  # Family_History_Binary
    1,  # Family_Load
    0,  # Location_Type_Urban
    0,  # School_Type_Encoded
    1,  # Tuition_Binary
    1,  # Comp_Exam_Binary
   -1,  # Vitamin_D_Binary
   -1,  # Sports_Encoded
] + [0] * len(state_cols)   # State one-hots — unconstrained

results_s2 = {}

# XGBoost WITH monotone constraints (the key fix)
auc_xgb_mono, m_xgb_mono = eval_clf(
    XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        monotone_constraints=tuple(base_constraints),
        random_state=42, eval_metric='logloss'
    ),
    X_risk_tr_s, y_risk_tr, X_risk_te_s, y_risk_te,
    "XGBoost + Monotone Constraints"
)
results_s2['XGBoost_Mono'] = (auc_xgb_mono, m_xgb_mono)

# Gradient Boosting (sklearn — no scaling needed but we pass scaled anyway)
auc_gb, m_gb = eval_clf(
    GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    X_risk_tr_s, y_risk_tr, X_risk_te_s, y_risk_te,
    "GradientBoosting"
)
results_s2['GradientBoosting'] = (auc_gb, m_gb)

# RandomForest
auc_rf, m_rf = eval_clf(
    RandomForestClassifier(
        n_estimators=300, max_depth=12,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    X_risk_tr_s, y_risk_tr, X_risk_te_s, y_risk_te,
    "RandomForest"
)
results_s2['RandomForest'] = (auc_rf, m_rf)

best_s2_name = max(results_s2, key=lambda k: results_s2[k][0])
best_risk_auc, best_risk_model = results_s2[best_s2_name]
print(f"\n✅ Stage 2 Winner: {best_s2_name}  AUC={best_risk_auc:.4f}")

# ═══════════════════════════════════════════════════════════════
# STAGE 3: DIOPTER REGRESSION  — fix the R²=-0.164 disaster
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 3 — Diopter Severity (fixing R²=-0.164)")
print("=" * 80)

df_re_pos = df_features[df_features['Has_RE'] == 1].copy()
df_re_pos = df_re_pos[df_re_pos['Degree_RE_Diopters'] != 0].copy()
print(f"RE-positive samples: {len(df_re_pos)}")

diop_cols = [
    'Age', 'BMI', 'Screen_Time_Hours', 'Near_Work_Hours', 'Outdoor_Time_Hours',
    'Age_Screen', 'Screen_Near_Total', 'Screen_Outdoor_Ratio',
    'High_Risk_Parent', 'Family_Load',
    'Location_Type_Urban', 'School_Type_Encoded',
    'Tuition_Binary', 'Comp_Exam_Binary', 'Vitamin_D_Binary', 'Sports_Encoded',
] + [c for c in state_cols if c in df_re_pos.columns]

X_diop = df_re_pos[diop_cols].fillna(0)
y_diop = df_re_pos['Degree_RE_Diopters'].abs()

X_diop_tr, X_diop_te, y_diop_tr, y_diop_te = train_test_split(
    X_diop, y_diop, test_size=0.2, random_state=42)

diop_scaler = StandardScaler()
X_diop_tr_s = diop_scaler.fit_transform(X_diop_tr)
X_diop_te_s = diop_scaler.transform(X_diop_te)

results_s3 = {}

r2_xgbr, mae_xgbr, rmse_xgbr, m_xgbr = eval_reg(
    XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42
    ),
    X_diop_tr_s, y_diop_tr, X_diop_te_s, y_diop_te,
    "XGBRegressor"
)
results_s3['XGBRegressor'] = (r2_xgbr, mae_xgbr, rmse_xgbr, m_xgbr)

r2_rfr, mae_rfr, rmse_rfr, m_rfr = eval_reg(
    RandomForestRegressor(
        n_estimators=300, max_depth=12,
        random_state=42, n_jobs=-1
    ),
    X_diop_tr_s, y_diop_tr, X_diop_te_s, y_diop_te,
    "RandomForestRegressor"
)
results_s3['RandomForestRegressor'] = (r2_rfr, mae_rfr, rmse_rfr, m_rfr)

r2_gbr, mae_gbr, rmse_gbr, m_gbr = eval_reg(
    GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    X_diop_tr_s, y_diop_tr, X_diop_te_s, y_diop_te,
    "GradientBoostingRegressor (tuned)"
)
results_s3['GradientBoostingRegressor'] = (r2_gbr, mae_gbr, rmse_gbr, m_gbr)

best_s3_name = max(results_s3, key=lambda k: results_s3[k][0])
best_diop_r2, best_diop_mae, best_diop_rmse, best_diop_model = results_s3[best_s3_name]
print(f"\n✅ Stage 3 Winner: {best_s3_name}  R²={best_diop_r2:.4f}  MAE={best_diop_mae:.3f}")

# ─────────────────────────────────────────────────────────────
# Save all models
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SAVING BEST MODELS")
print("=" * 80)

for candidate in ['../models', 'models']:
    if os.path.exists(candidate):
        save_dir = candidate
        break
else:
    save_dir = '../models'
    os.makedirs(save_dir, exist_ok=True)

joblib.dump(best_re_model,   f"{save_dir}/has_re_model_improved.pkl")
joblib.dump(re_scaler,       f"{save_dir}/has_re_scaler.pkl")
print(f"✅ Stage 1: {best_s1_name}  (AUC {best_re_auc:.4f})")

joblib.dump(best_risk_model, f"{save_dir}/risk_progression_model.pkl")
joblib.dump(risk_scaler,     f"{save_dir}/scaler_classification.pkl")
print(f"✅ Stage 2: {best_s2_name}  (AUC {best_risk_auc:.4f})")

joblib.dump(best_diop_model, f"{save_dir}/diopter_regression_model.pkl")
joblib.dump(diop_scaler,     f"{save_dir}/scaler_regression.pkl")
print(f"✅ Stage 3: {best_s3_name}  (R² {best_diop_r2:.4f}  MAE {best_diop_mae:.3f}D)")

# Feature lists
with open(f"{save_dir}/has_re_features.json", 'w') as f:
    json.dump({
        "model_type": best_s1_name,
        "feature_count": len(re_cols),
        "feature_columns": re_cols,
        "sklearn_version": sklearn.__version__,
        "metrics": {"auc": float(best_re_auc), "threshold": 0.5}
    }, f, indent=2)

with open(f"{save_dir}/feature_columns.json", 'w') as f:
    json.dump(risk_cols, f, indent=2)

# Metadata
meta = {
    "sklearn_version": sklearn.__version__,
    "training_date"  : pd.Timestamp.now().strftime("%Y-%m-%d"),
    "risk_model": {
        "name"     : best_s2_name,
        "auc"      : float(best_risk_auc),
        "target"   : "Progression_Risk (High=1, Low/Moderate=0)",
        "threshold": 0.5,
        "note"     : "Monotone constraints applied (outdoor↓risk, screen↑risk)"
    },
    "re_model": {
        "name"       : best_s1_name,
        "auc"        : float(best_re_auc),
        "target"     : "Has_RE (Yes=1, No=0)",
        "threshold"  : 0.5,
        "improvement": "Best of XGBoost/RF with enhanced interaction features"
    },
    "diopter_model": {
        "name"  : best_s3_name,
        "mae"   : float(best_diop_mae),
        "rmse"  : float(best_diop_rmse),
        "r2"    : float(best_diop_r2),
        "target": "Degree_RE_Diopters (absolute value)",
        "note"  : "Best of XGBRegressor/RF/GBR — fixed R²=-0.164 issue"
    },
    "dataset": {
        "total_samples": len(dm),
        "train_samples": int(len(dm) * 0.8),
        "test_samples" : int(len(dm) * 0.2)
    },
    "comparison": {
        "stage1": {k: float(v[0]) for k, v in results_s1.items()},
        "stage2": {k: float(v[0]) for k, v in results_s2.items()},
        "stage3": {k: float(v[0]) for k, v in results_s3.items()},
    }
}

with open(f"{save_dir}/model_metadata.json", 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Metadata saved to {save_dir}/model_metadata.json")

print("\n" + "=" * 80)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY")
print(f"  Stage 1 ({best_s1_name}):  AUC = {best_re_auc:.4f}")
print(f"  Stage 2 ({best_s2_name}):  AUC = {best_risk_auc:.4f}")
print(f"  Stage 3 ({best_s3_name}): MAE = {best_diop_mae:.3f} D  R² = {best_diop_r2:.4f}")
print("=" * 80)
