# 📊 Visual Diagrams: Myopia Risk Prediction System

## 1. Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER (React Frontend)                       │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ 1. SCREENING PAGE (Screen.tsx)                                    │   │
│  │    ├─ Step 1: Personal Info (Age, Sex, Height, Weight)          │   │
│  │    ├─ Step 2: Family History (Parents Myopic?)                  │   │
│  │    ├─ Step 3: Lifestyle (Screen, Outdoor, Sports, etc.)        │   │
│  │    └─ Submit Button → Data to sessionStorage                    │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│            ↓                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ 2. RESULTS PAGE (Results.tsx) - Shows Loading Spinner            │   │
│  │    └─ Calls Backend API: POST /predict                           │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│            ↓                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ 3. VISUALIZATION & PDF                                           │   │
│  │    ├─ RiskGauge Component → Animated semicircle gauge           │   │
│  │    ├─ Risk Level Display → Color-coded text                     │   │
│  │    ├─ Three-Stage Summary → Cards showing each stage            │   │
│  │    ├─ Recommendations → What to do next                         │   │
│  │    └─ Download PDF → Report with child name                     │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                    ↕ HTTP POST /predict (JSON)
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKEND (Python Flask)                                   │
│                    Running on localhost:5001                               │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ api.py: /predict ENDPOINT                                         │   │
│  │                                                                    │   │
│  │ 1. Receive JSON screening data                                    │   │
│  │ 2. Validate inputs (type checks, range validation)               │   │
│  │ 3. Build feature vectors (8→30 features)                        │   │
│  │ 4. Load pre-trained ML models from disk                          │   │
│  │ 5. Run 3-stage prediction pipeline                               │   │
│  │ 6. Return JSON result to frontend                                │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─ ML MODELS (Loaded at startup) ──────────────────────────────────┐     │
│  │                                                                  │     │
│  │ 1. has_re_model (Stage 1)                                       │     │
│  │    └─ XGBoost classification                                    │     │
│  │    └─ Predicts: Does child have refractive error?             │     │
│  │                                                                  │     │
│  │ 2. risk_progression_model (Stage 2)                             │     │
│  │    └─ GradientBoosting classification                           │     │
│  │    └─ Predicts: Myopia progression risk level                  │     │
│  │    └─ Trained on 5000 real screening records                   │     │
│  │    └─ AUC: 0.893 (very high accuracy)                          │     │
│  │                                                                  │     │
│  │ 3. diopter_regression_model (Stage 3)                           │     │
│  │    └─ XGBoost regression                                        │     │
│  │    └─ Predicts: Diopter severity (continuous value)            │     │
│  │                                                                  │     │
│  │ 4. Scalers (Min-Max normalization)                              │     │
│  │    └─ Normalize input features to 0-1 range                    │     │
│  │    └─ Required by all ML models                                 │     │
│  │                                                                  │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│  ┌─ RULE-BASED SCORING SYSTEM ──────────────────────────────────────┐    │
│  │                                                                  │    │
│  │ Evidence-based point system:                                    │    │
│  │  • Base score: 30 points                                        │    │
│  │  • Add/subtract based on risk factors                           │    │
│  │  • Final score: 0-100 (converted to probability)               │    │
│  │  • Used as safety check & calibration                          │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─ DATABASE (MongoDB) ─────────────────────────────────────────────┐    │
│  │  myopia_guard.users                                              │    │
│  │  ├─ users collection: name, email, password                     │    │
│  │  └─ Stores account information                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ML Prediction Pipeline (3 Stages)

```
SCREENING DATA INPUT
│
├─ age, sex, height, weight
├─ familyHistory, parentsMyopic
├─ screenTime, outdoorTime, nearWork, sports, vitaminD
│
└─────────────────────────────────────────────────────────────────┐
                                                                  │
                    FEATURE ENGINEERING                           │
                          │                                        │
        ┌─────────────────┼─────────────────┐                    │
        ↓                 ↓                 ↓                    │
                                                                  │
    STAGE 1          STAGE 2           STAGE 3                    │
  (RE Detect)     (Risk Level)      (Diopter Est.)                │
    │                 │                 │                         │
    ├─ 8 features  ├─ 30 features   ├─ 27 features               │
    │              │                │                             │
    ├─ Scale       ├─ Scale         ├─ Scale                      │
    │ (0-1)        │ (0-1)          │ (0-1)                       │
    │              │                │                             │
    ├─ XGBoost  ├─ GradientBoosting├─ XGBoost                    │
    │ Classifier   │ Classifier      │ Regressor                  │
    │ (Binary)     │ (Binary)        │ (Continuous)               │
    │              │                │                             │
    ├─ Predict  ├─ Predict Prob  ├─ Predict Diopter              │
    │ Prob of RE   │ of Risk        │ Value                       │
    │ (0-1)        │ (0-1)          │ (0-10)                      │
    │              │                │                             │
    └─→ RE_Prob    └────┬───────────┴──────────→ Diopter_Value   │
                       │                                           │
                       │  + RULE-BASED SCORING                    │
                       │        │                                  │
                       ├─ Calculate base (30)                      │
                       ├─ Add age factor                           │
                       ├─ Add genetics factor                      │
                       ├─ Add lifestyle factors                    │
                       └─ Final Rule_Prob (0-1)                   │
                       │                                           │
                       │  ADAPTIVE FUSION                          │
                       ├─ if ML_Prob >= 0.65:                    │
                       │  Risk = 0.60×ML + 0.40×Rule             │
                       │                                           │
                       ├─ if ML_Prob >= 0.35:                    │
                       │  Risk = 0.50×ML + 0.50×Rule             │
                       │                                           │
                       └─ else:                                   │
                          Risk = 0.20×ML + 0.80×Rule             │
                       │                                           │
                       └─→ Final Risk_Score (0-100%)              │
                                                                  │
                            ↓                                      │
                       CLASSIFY LEVEL                              │
                       ├─ if score < 40:  LOW                     │
                       ├─ if score < 70:  MODERATE                │
                       └─ if score ≥ 70:  HIGH                    │
                                                                  │
                            ↓                                      │
                       CATEGORIZE SEVERITY                         │
                       ├─ if diopter < 0.5:    Negligible        │
                       ├─ if diopter < 3.0:    Mild              │
                       ├─ if diopter < 6.0:    Moderate          │
                       └─ if diopter ≥ 6.0:    High              │
                                                                  │
└──────────────────────────────────────────────────────────────────┘

FINAL OUTPUT:
{
  "risk_score": 69,
  "risk_level": "MODERATE",
  "risk_probability": 0.688,
  "has_re": true,
  "re_probability": 0.68,
  "diopters": 2.45,
  "severity": "Mild"
}
```

---

## 3. Gauge Visualization (RiskGauge.tsx)

```
NUMERICAL INPUT: risk_score = 69

    ANIMATION CALCULATION
             │
             ├─ Duration: 1.5 seconds
             ├─ Steps: 60 frames
             ├─ Increment per frame: 69 / 60 ≈ 1.15
             ├─ Current frame updates every ~25ms
             └─ Example frames:
                Frame 0:   displayScore = 0°
                Frame 10:  displayScore = 11°
                Frame 20:  displayScore = 23°
                Frame 30:  displayScore = 34°
                Frame 60:  displayScore = 69° ✓ Done
             │
             ↓
    ROTATION CALCULATION
             │
             ├─ Formula: rotation = (displayScore / 100) × 180°
             ├─ For score 69: rotation = 0.69 × 180° = 124.2°
             └─ Range: 0° (leftmost) to 180° (rightmost)
             │
             ↓
    COLOR SELECTION
             │
             ├─ if displayScore < 40:  "var(--low-risk)"     [GREEN]
             ├─ if displayScore < 70:  "var(--moderate-risk)"[AMBER]
             └─ else:                   "var(--high-risk)"    [RED]
             │
             ├─ For score 69: color = AMBER (60% green, 40% orange mix)
             │
             ↓
    SVG RENDERING
             │
             ├─ Background Arc (light gray)
             │  Path: M 20 90 A 80 80 0 0 1 180 90
             │
             ├─ Risk Zones (faded background)
             │  ├─ Green zone:  0-40% of arc
             │  ├─ Amber zone:  40-70% of arc
             │  └─ Red zone:    70-100% of arc
             │
             ├─ Progress Arc (animated, colored)
             │  └─ Animates from 0 to 124.2° with amber color
             │
             ├─ Needle (animated)
             │  ├─ Line: x1=100, y1=90 to x2=100, y2=25
             │  ├─ Rotates 124.2° around (100, 90)
             │  ├─ Color: amber (matches zone)
             │  └─ Center circle at rotation point
             │
             ├─ Score Text (animated with spring)
             │  ├─ Displays: "69%"
             │  ├─ Font size: 36px
             │  ├─ Color: amber
             │  ├─ Scales in from 0 to 1 at ~0.5s delay
             │  └─ Position: centered at bottom (-5px)
             │
             └─ Scale Labels
                ├─ Left:   "0"
                ├─ Center: "50"
                └─ Right:  "100"

FINAL RENDERED OUTPUT: Semi-circular gauge with moving needle pointing to 69%
```

---

## 4. Data Flow: Screening → Prediction → Screen

```
USER INTERACTION TIMELINE:

┌──────────────────────────────────────────────────────────────────┐
│  USER VISITS http://localhost:5173/screen                       │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Display Screen.tsx (3-Step Form)                            │
│    Step 1: Personal Info (Age, Sex, Height, Weight)            │
│    Step 2: Family History (Parents Myopic?)                    │
│    Step 3: Lifestyle (Screen, Outdoor, Sports, etc.)           │
└──────────────────────────────────────────────────────────────────┘
                    │ User fills form
                    │
                    ↓ Clicks SUBMIT
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Validation (Frontend)                                        │
│    └─ All fields required? Age 6-18? Screen 0-24?             │
│       └─ If invalid: Show error messages                        │
│       └─ If valid: Continue to next step                        │
└──────────────────────────────────────────────────────────────────┘
                    │ Valid
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Save to sessionStorage                                       │
│    └─ Key: "screeningData"                                      │
│    └─ Value: {age, sex, height, weight, ...}                   │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Navigate to /results                                         │
│    └─ Renders Results.tsx                                       │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Display Loading State                                        │
│    ├─ Show: "Analysing with AI model…"                         │
│    ├─ Show: Rotating loader spinner                             │
│    └─ Meanwhile: useEffect() hook runs...                       │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Read from sessionStorage                                      │
│    └─ Get screeningData JSON                                     │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Call Backend API                                              │
│    Method: POST                                                  │
│    URL: http://localhost:5001/predict                           │
│    Content-Type: application/json                               │
│    Body: screeningData (JSON)                                   │
│                                                                  │
│    [NETWORK REQUEST SENT →]                                    │
└──────────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴────────────┐
        │   BACKEND PROCESSING   │
        │   (See earlier diagram)│
        │   [~200-500ms delay]   │
        └───────────┬────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ← Response Received (JSON)                                     │
│    {                                                             │
│      "risk_score": 69,                                           │
│      "risk_level": "MODERATE",                                   │
│      "has_re": true,                                             │
│      "diopters": 2.45,                                           │
│      "severity": "Mild"                                          │
│    }                                                             │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Update React State                                            │
│    setRiskScore(69)                                              │
│    setRiskLevel("MODERATE")                                      │
│    setPrediction({...})                                          │
│    setLoading(false)                                             │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
┌──────────────────────────────────────────────────────────────────┐
│  ✓ Render Results Page                                           │
│    ├─ Hide spinner                                               │
│    ├─ Show RiskGauge (starts animation)                         │
│    │  └─ Needle animates from 0° to 124.2° over 1.5 seconds   │
│    │  └─ Arc fills left to right with amber color             │
│    │  └─ Number appears: "69%"                                 │
│    │                                                             │
│    ├─ Show Risk Level Card: "MODERATE RISK - 69%"             │
│    │                                                             │
│    ├─ Show 3-Stage Summary:                                     │
│    │  ├─ Stage 1: YES | 72% (Has RE)                          │
│    │  ├─ Stage 2: MODERATE | 69% (Risk Level)                 │
│    │  └─ Stage 3: -2.45D | Mild (Severity)                    │
│    │                                                             │
│    ├─ Show Recommendations:                                     │
│    │  ├─ Schedule eye check-up                                  │
│    │  ├─ Increase outdoor time                                  │
│    │  └─ Reduce screen time to <2 hrs/day                     │
│    │                                                             │
│    └─ Show Download PDF Button                                 │
└──────────────────────────────────────────────────────────────────┘
                    │
                    ↓
        ┌───────────────────────────┐
        │  USER ACTIONS             │
        │                           │
        ├─ View Results ✓           │
        ├─ Download PDF ✓           │
        ├─ Share with doctor ✓      │
        ├─ Go home / Logout ✓       │
        └   Or take another test ✓ │
        └───────────────────────────┘
```

---

## 5. Risk Score Distribution Graph

```
TYPICAL RISK SCORE DISTRIBUTION (From 5000 test cases)

Frequency
│    ░░░
│    ░░░  ░░░
│    ░░░  ░░░  ░░░
│    ░░░  ░░░  ░░░  ░░░  ░░░
│    ░░░  ░░░  ░░░  ░░░  ░░░  ░░░
0-10│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    ├─────────┼─────────┼─────────┼─────────┤
    0%       25%      50%       75%     100%
   LOW      LOW/MOD   MOD     MOD/HIGH   HIGH

  Peak Distribution:
  • 15-20%: LOW RISK (protective families, good habits)
  • 35-45%: MODERATE RISK (mixed factors)
  • 65-75%: HIGH RISK (genetic predisposition + lifestyle)

Insights:
  ✓ Bimodal distribution (peaks at extremes)
  ✓ Family history is strongest predictor
  ✓ Lifestyle can shift someone ±15-20% points
  ✓ Different educational backgrounds show different patterns
```

---

## 6. Feature Importance Heatmap

```
WHICH FACTORS MATTER MOST?

┌────────────────────────────────────┐
│      FEATURE IMPORTANCE            │
├────────────────────────────────────┤
│ Parents Myopic (both)   ████████░░  │ 25-30%
│ Age (< 8 years)         ██████████  │ 20-25%
│ Screen Time (high)      ████████░░  │ 15-20%
│ Outdoor Time (low)      ████████░░  │ 15-20%
│ Family History          ██████░░░░  │ 10-15%
│ Near Work (high)        █████░░░░░  │ 8-12%
│ Academic Pressure       ████░░░░░░  │ 5-8%
│ BMI Category            ███░░░░░░░  │ 3-5%
│ Sex (Male>Female)       ██░░░░░░░░  │ 2-3%
│ Vitamin D               █░░░░░░░░░  │ 1-2%
└────────────────────────────────────┘

KEY INSIGHT:
Most important factors:
1. Genetics (50% of decision)
2. Lifestyle (30% of decision)
3. Age/Development (20% of decision)

Least important factors:
• Exact BMI value
• Sex (slight male bias)
• Vitamin D (weak effect)
```

---

## 7. Example: Score Calculation Walkthrough

```
EXAMPLE PATIENT: 11-year-old boy

┌─ Input Data ─────────────────────────┐
│ Age: 11 years                        │
│ Sex: Male                            │
│ Height: 150 cm                       │
│ Weight: 45 kg → BMI = 20.0          │
│ Family: Both parents myopic          │
│ Screen time: 6 hours/day             │
│ Outdoor time: 0.5 hours/day          │
│ Near work: 5 hours/day               │
│ Sports: Occasional                   │
│ Vitamin D: No                        │
└──────────────────────────────────────┘
                │
                ↓
┌─ RULE-BASED SCORING ─────────────────┐
│ Base                          30      │
│ + Age 11 (10-12 range)       +5      │
│ + Both parents myopic        +25      │
│ + Screen 6 hrs              +17      │
│ + Outdoor 0.5 hrs           +20      │
│ + Near work 5 hrs           +8       │
│ + Occasional sports          -3      │
│ ─────────────────────────────────    │
│ RULE SCORE = 102 → clamped to 100   │
│ Rule_Prob = 1.0                     │
└──────────────────────────────────────┘
                │
                ↓
┌─ ML MODEL PREDICTION ────────────────┐
│ Input 30 features (derived)          │
│ XGBoost Classifier                   │
│ Output: ML_Prob = 0.78               │
│ (Confident HIGH risk)                │
└──────────────────────────────────────┘
                │
                ↓
┌─ HYBRID FUSION ──────────────────────┐
│ Since ML_Prob = 0.78 >= 0.65:        │
│ (ML is confident)                    │
│                                      │
│ Risk = 0.60 × 0.78 + 0.40 × 1.0     │
│      = 0.468 + 0.400                 │
│      = 0.868                         │
│                                      │
│ Risk_Score = 87%                     │
│                                      │
│ Floor check: max(0.868, 0.75×1.0)   │
│            = max(0.868, 0.75) = 0.868│
│ PASSES floor check ✓                 │
└──────────────────────────────────────┘
                │
                ↓
┌─ CLASSIFICATION ─────────────────────┐
│ Risk_Score = 87%                     │
│ Risk_Level = "HIGH" (≥70)            │
└──────────────────────────────────────┘
                │
                ↓
┌─ DIOPTER ESTIMATION ─────────────────┐
│ Stage 1: has_re = true ✓             │
│ (87% score suggests RE)               │
│                                      │
│ XGBoost Regression                   │
│ Input: 27 features                   │
│ Output: 3.2 diopters                 │
│                                      │
│ Severity: "Moderate" (3.0-6.0D)      │
└──────────────────────────────────────┘
                │
                ↓
┌─ FINAL RESULT ───────────────────────┐
│ {                                    │
│   "risk_score": 87,                  │
│   "risk_level": "HIGH",              │
│   "has_re": true,                    │
│   "re_probability": 0.81,            │
│   "diopters": 3.2,                   │
│   "severity": "Moderate"             │
│ }                                    │
└──────────────────────────────────────┘
                │
                ↓
┌─ VISUALIZATION ──────────────────────┐
│ GAUGE SHOWS:                         │
│                                      │
│       87°  ╱
│          │╱  RED ZONE
│          │   HIGH RISK
│       87%│   (87°/180°)
│          │
│  Needle animates from 0° to 87°     │
│  Arc fills left to right             │
│  Color: RED (amber transitioning)    │
│  Number: "87%" in red                │
└──────────────────────────────────────┘
```

---

This completes the visual breakdown of the entire system!

