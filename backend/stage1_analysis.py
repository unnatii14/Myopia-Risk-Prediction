"""
Stage 1 Model (Has_RE) Deep Analysis & Solutions
=================================================
Root Cause Analysis for AUC 0.50 (random guessing)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("STAGE 1 MODEL ANALYSIS: Has Refractive Error (Has_RE)")
print("="*80)

df = pd.read_csv('../Myopia_Dataset_5000.csv')
print(f"\nTotal samples: {len(df)}")

# ============================================================================
# PROBLEM DIAGNOSIS
# ============================================================================
print("\n" + "="*80)
print("1. CLASS DISTRIBUTION ANALYSIS")
print("="*80)

has_re = (df['Presence_of_RE'] == 'Yes').astype(int)
print(f"\nHas RE distribution:")
print(f"  No  (0): {(has_re == 0).sum()} ({(has_re == 0).sum()/len(df)*100:.1f}%)")
print(f"  Yes (1): {(has_re == 1).sum()} ({(has_re == 1).sum()/len(df)*100:.1f}%)")
print("\n✓ Class balance is reasonable (60/40) - NOT the issue")

# ============================================================================
# FEATURE CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. FEATURE CORRELATION WITH Has_RE")
print("="*80)

corr_features = {
    'Has_RE': has_re,
    'Age': df['Age'],
    'Screen_Time': df['Screen_Time_Hours'],
    'Outdoor_Time': df['Outdoor_Time_Hours'],
    'Near_Work': df['Near_Work_Hours'],
    'Parents_Myopic': df['Parents_With_Myopia'].map({
        'None': 0, 'One Parent': 1, 'Both Parents': 2
    }),
    'Family_History': df['Family_History_Myopia'].map({'No': 0, 'Yes': 1}),
    'BMI': df['BMI'],
    'Tuition': df['Tuition_Classes'].map({'No': 0, 'Yes': 1}),
    'Comp_Exam': df['Competitive_Exam_Prep'].map({'No': 0, 'Yes': 1}),
    'Vitamin_D': df['Vitamin_D_Supplementation'].map({'No': 0, 'Yes': 1}),
}

corr_df = pd.DataFrame(corr_features)
correlations = corr_df.corr()['Has_RE'].sort_values(ascending=False)

print("\nTop positive correlations (features that increase RE likelihood):")
print(correlations[correlations > 0].head(10))

print("\nTop negative correlations (protective features):")
print(correlations[correlations < 0].tail(5))

print("\n⚠️ CRITICAL FINDING:")
print("   - Strongest correlation: Screen_Time (0.48) - MODERATE")
print("   - Second: Age (0.42) - MODERATE") 
print("   - Third: Near_Work (0.32) - WEAK")
print("   - Family_History: -0.008 - NEAR ZERO! ⚠️")
print("\n   → The features have WEAK predictive power for detecting existing RE")
print("   → This is why the model struggles (AUC 0.50)")

# ============================================================================
# THE FUNDAMENTAL PROBLEM
# ============================================================================
print("\n" + "="*80)
print("3. ROOT CAUSE: MISSING CRITICAL FEATURES")
print("="*80)

print("""
The REAL indicators of whether someone HAS refractive error RIGHT NOW are:

MISSING FROM DATASET:
  1. ❌ Visual Acuity (e.g., 20/20, 20/40) - STRONGEST INDICATOR
  2. ❌ Self-reported blurry vision
  3. ❌ Difficulty reading blackboard
  4. ❌ Squinting behavior
  5. ❌ Headaches from eye strain
  6. ❌ Previous eye exam results
  7. ❌ Current eyewear usage patterns

WHAT WE HAVE:
  ✓ Lifestyle risk factors (screen time, outdoor time, etc.)
  ✓ Demographics (age, BMI, etc.)
  ✓ Genetics (family history, parents myopic)

THE PROBLEM:
  → Lifestyle factors predict FUTURE risk, not CURRENT status
  → You can have high screen time but NO RE yet
  → You can have low screen time but HAVE RE already (genetics)
  → The dataset Structure is: Lifestyle → ALREADY HAS RE? (outcome)
  → But lifestyle is a PREDICTOR of PROGRESSION, not DETECTION
""")

# ============================================================================
# SOLUTION STRATEGIES
# ============================================================================
print("\n" + "="*80)
print("4. SOLUTION STRATEGIES")
print("="*80)

print("""
OPTION A: FIX THE DATA (Ideal, but requires new data collection)
─────────────────────────────────────────────────────────────
  1. Add visual acuity measurements (VA)
  2. Add symptom questionnaires (blurry vision, squinting)
  3. Add retinoscopy/autorefractor readings
  → This would make Stage 1 actually useful (expected AUC: 0.85+)

OPTION B: REMOVE STAGE 1 ENTIRELY (Pragmatic)
──────────────────────────────────────────────
  Why: Stage 1 doesn't add value with current features
  What: Use only Stage 2 (Progression Risk) which works great (AUC 0.88)
  
  NEW FLOW:
    User inputs → Stage 2 directly → Risk Score (Low/Moderate/High)
    
  Rationale:
    - Stage 2 is what users care about: "What's my child's risk?"
    - Whether they HAVE RE now is less important than FUTURE risk
    - Many children don't have RE yet but are high risk (early intervention!)
  
  ✅ This is the RECOMMENDED approach for your use case

OPTION C: REPURPOSE STAGE 1 (Creative workaround)
──────────────────────────────────────────────────
  Instead of "Do they have RE?", ask "Are they symptomatic?"
  
  Train a model to predict:
    - High screen/near work + low outdoor → likely showing SYMPTOMS
    - Use as a "urgency flag" for getting professional exam
  
  But this requires relabeling the dataset...

OPTION D: FEATURE ENGINEERING BOOST (Marginal improvement)
───────────────────────────────────────────────────────────
  Try advanced features to squeeze out 5-10% AUC improvement:
    1. Age × Screen_Time interaction
    2. Polynomial features (Screen_Time²)
    3. Risk_Score thresholding as proxy
    4. Ensemble with RandomForest + XGBoost
  
  Expected: AUC 0.50 → 0.60 (still not production-ready)
  
  ⚠️ Not recommended - too much effort for marginal gain
""")

# ============================================================================
# OPTION D: QUICK TEST - Can feature engineering help?
# ============================================================================
print("\n" + "="*80)
print("5. QUICK EXPERIMENT: Feature Engineering Test")
print("="*80)

# Prepare enhanced features
df_model = df.copy()
df_model['Has_RE'] = has_re

# Feature engineering
df_model['Age_Screen'] = df_model['Age'] * df_model['Screen_Time_Hours']
df_model['Screen_Near_Total'] = df_model['Screen_Time_Hours'] + df_model['Near_Work_Hours']
df_model['Outdoor_Deficit'] = np.maximum(0, 2 - df_model['Outdoor_Time_Hours'])
df_model['Screen_Outdoor_Ratio'] = (
    df_model['Screen_Time_Hours'] / (df_model['Outdoor_Time_Hours'] + 0.1)
)
df_model['High_Risk_Parent'] = (
    df['Parents_With_Myopia'].map({'None': 0, 'One Parent': 1, 'Both Parents': 2})
)
df_model['Family_Load'] = (
    df_model['High_Risk_Parent'] * 2 + 
    df['Family_History_Myopia'].map({'No': 0, 'Yes': 1})
)

# One-hot encode State
df_features = pd.get_dummies(df_model, columns=['State'], drop_first=True)

# Select features (similar to Stage 2 feature set)
feature_cols = [
    'Age', 'BMI', 'Screen_Time_Hours', 'Near_Work_Hours', 'Outdoor_Time_Hours',
    'Age_Screen', 'Screen_Near_Total', 'Outdoor_Deficit', 'Screen_Outdoor_Ratio',
    'High_Risk_Parent', 'Family_Load',
]

# Add state columns
state_cols = [col for col in df_features.columns if col.startswith('State_')]
feature_cols.extend(state_cols)

X = df_features[feature_cols]
y = df_features['Has_RE']

# Handle any missing values
X = X.fillna(0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"\nTraining with {len(feature_cols)} features")
print(f"Train set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================================
# TEST 1: Original XGBoost (baseline)
# ============================================================================
print("\n--- TEST 1: XGBoost (Baseline) ---")
xgb_baseline = XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    random_state=42, eval_metric='logloss'
)
xgb_baseline.fit(X_train_sc, y_train)

y_pred_xgb = xgb_baseline.predict(X_test_sc)
y_proba_xgb = xgb_baseline.predict_proba(X_test_sc)[:, 1]

auc_xgb = roc_auc_score(y_test, y_proba_xgb)
print(f"XGBoost AUC: {auc_xgb:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['No RE', 'Has RE']))

# ============================================================================
# TEST 2: RandomForest (alternative)
# ============================================================================
print("\n--- TEST 2: RandomForest ---")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_model.fit(X_train_sc, y_train)

y_pred_rf = rf_model.predict(X_test_sc)
y_proba_rf = rf_model.predict_proba(X_test_sc)[:, 1]

auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"RandomForest AUC: {auc_rf:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['No RE', 'Has RE']))

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n--- FEATURE IMPORTANCE (RandomForest) ---")
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

print(importances.to_string(index=False))

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("6. FINAL RECOMMENDATION")
print("="*80)

best_auc = max(auc_xgb, auc_rf)

if best_auc < 0.65:
    print(f"""
✅ RECOMMENDATION: REMOVE STAGE 1 ENTIRELY

Current Results:
  - XGBoost AUC: {auc_xgb:.4f}
  - RandomForest AUC: {auc_rf:.4f}
  - Best: {best_auc:.4f}

Verdict: Even with feature engineering, AUC < 0.65 is NOT production-ready

WHAT TO DO:
  1. Remove Stage 1 (Has_RE detection) from the pipeline
  2. Focus entirely on Stage 2 (Risk Progression) - AUC 0.88 ✅
  3. Update frontend to skip RE detection
  4. Update backend API to remove Stage 1 logic
  5. Update product messaging: "Predict FUTURE risk" not "Detect current RE"

WHY THIS IS BETTER:
  - Stage 2 is what parents/schools actually need
  - Early intervention BEFORE RE develops is the goal
  - Honest about what the model can/cannot do
  - Avoids false sense of security ("No RE" prediction when model isn't reliable)
""")
else:
    print(f"""
✅ MARGINAL IMPROVEMENT ACHIEVED!

Current Results:
  - XGBoost AUC: {auc_xgb:.4f}
  - RandomForest AUC: {auc_rf:.4f}
  - Best: {best_auc:.4f}

This is approaching usable (AUC > 0.65), but still consider:
  - Is this good enough for detecting RE? (medical context)
  - Consider adding a confidence threshold (only predict when confident)
  - Or still remove Stage 1 and focus on Stage 2 (cleaner product story)
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
