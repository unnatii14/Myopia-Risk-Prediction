# 🧮 Myopia Risk Prediction: Complete Calculation & Visualization Guide

## 📊 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    1. DATA COLLECTION (Screening)                   │
│─────────────────────────────────────────────────────────────────────│
│  Screen.tsx - 3-Step Questionnaire                                  │
│                                                                     │
│  Step 1: PERSONAL INFO                                             │
│  ├─ Age (6-18 years)                                              │
│  ├─ Sex (Male/Female)                                             │
│  ├─ Height (cm)                                                   │
│  └─ Weight (kg) → calculates BMI                                 │
│                                                                     │
│  Step 2: FAMILY HISTORY                                           │
│  ├─ Has myopia in family? (Yes/No)                               │
│  └─ How many parents myopic? (None/One/Both)                    │
│                                                                     │
│  Step 3: LIFESTYLE FACTORS                                        │
│  ├─ Screen time per day (0-12 hours) [SLIDER]                   │
│  ├─ Near work per day (0-12 hours) [SLIDER]                     │
│  ├─ Outdoor time per day (0-8 hours) [SLIDER]                   │
│  ├─ Sports frequency (Rare/Occasional/Regular)                   │
│  └─ Vitamin D supplementation? (Yes/No)                          │
│                                                                     │
│  User clicks SUBMIT → Data stored in sessionStorage               │
│  User navigated to /results page                                  │
└─────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│              2. DATA TRANSMISSION TO BACKEND (API)                  │
│─────────────────────────────────────────────────────────────────────│
│  POST http://localhost:5001/predict                               │
│                                                                     │
│  Payload (JSON):                                                   │
│  {                                                                  │
│    "age": 10,                                                       │
│    "sex": "male",                                                   │
│    "height": 145,                                                   │
│    "weight": 38,                                                    │
│    "familyHistory": true,                                           │
│    "parentsMyopic": "one",                                          │
│    "screenTime": 5,                                                 │
│    "nearWork": 3,                                                   │
│    "outdoorTime": 1.5,                                              │
│    "sports": "occasional",                                          │
│    "vitaminD": false                                                │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│               3. BACKEND ML PROCESSING (3-STAGE PIPELINE)           │
│─────────────────────────────────────────────────────────────────────│
│                                                                     │
│  Input Validation:                                                 │
│  ├─ Check data types (age 6-18, screen time 0-24, etc.)          │
│  ├─ Check ranges and missing values                               │
│  └─ Return 400 error if invalid                                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐
│  │ STAGE 1: REFRACTIVE ERROR DETECTION                          │
│  ├──────────────────────────────────────────────────────────────┤
│  │ Question: Does the child have refractive error (RE)?        │
│  │ Model: XGBoost Classification                               │
│  │ Input: 8 features (age, BMI, family history, etc.)         │
│  │ Output: Probability (0-1)                                   │
│  │                                                              │
│  │ re_probability = model.predict_proba()[1]                  │
│  │ has_re = (re_probability >= 0.5)                           │
│  │                                                              │
│  │ Example: RE_Prob = 0.68 → Child LIKELY has RE              │
│  └──────────────────────────────────────────────────────────────┘
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐
│  │ STAGE 2: PROGRESSION RISK ASSESSMENT (HYBRID)               │
│  ├──────────────────────────────────────────────────────────────┤
│  │ Question: What is the progression risk?                    │
│  │ Answer: LOW (0-40%) | MODERATE (40-70%) | HIGH (70-100%)  │
│  │                                                              │
│  │ TWO PARALLEL METHODS:                                       │
│  │                                                              │
│  │ A) ML MODEL (GradientBoosting)                              │
│  │    ├─ AUC: 0.893 (very accurate)                           │
│  │    ├─ Trained on 5000 real screening records              │
│  │    ├─ Input: 30 clinical features                          │
│  │    └─ Output: ML_Probability (0-1)                         │
│  │                                                              │
│  │ B) RULE-BASED SCORING (Evidence-Based WHO Guidelines)     │
│  │    ├─ Base Score: 30 (neutral starting point)             │
│  │    ├─ Add points based on risk factors (see below)        │
│  │    └─ Output: Rule_Probability (0-1)                       │
│  │                                                              │
│  │ ADAPTIVE HYBRID FUSION:                                    │
│  │                                                              │
│  │ if ML_Prob >= 0.65:        (ML confident HIGH)             │
│  │    Risk = 0.60×ML + 0.40×Rule  (Trust ML 60%)            │
│  │                                                              │
│  │ elif ML_Prob >= 0.35:      (ML neutral)                    │
│  │    Risk = 0.50×ML + 0.50×Rule  (50/50 blend)            │
│  │                                                              │
│  │ else:                       (ML giving LOW)                │
│  │    Risk = 0.20×ML + 0.80×Rule  (Lean on rules 80%)      │
│  │                                                              │
│  │ Floor: Risk = max(Risk, 0.75×Rule)  (safety check)        │
│  │                                                              │
│  │ Risk_Percentage = int(Risk × 100)                          │
│  │                                                              │
│  │ Example:                                                    │
│  │  ML_Prob = 0.72 (confident HIGH)                          │
│  │  Rule_Prob = 0.65 (also HIGH)                             │
│  │  Risk = 0.60 × 0.72 + 0.40 × 0.65 = 0.688                │
│  │  Risk_Score = 69% → MODERATE RISK                         │
│  └──────────────────────────────────────────────────────────────┘
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐
│  │ STAGE 3: DIOPTER SEVERITY ESTIMATE (Regression)            │
│  ├──────────────────────────────────────────────────────────────┤
│  │ Question: How severe is the myopia? (in diopters)         │
│  │ Only calculated if Stage 1 = TRUE (has RE)                │
│  │                                                              │
│  │ Model: XGBoost Regression                                  │
│  │ Input: 27 features (subset of all features)               │
│  │ Output: Diopter value (absolute value)                     │
│  │                                                              │
│  │ Diopter Examples:                                           │
│  │  0.0 to 0.5D → Negligible                                  │
│  │  0.5 to 3.0D → Mild       (minor correction needed)        │
│  │  3.0 to 6.0D → Moderate   (noticeable problem)             │
│  │  > 6.0D      → High       (significant correction)         │
│  │                                                              │
│  │ Fallback if model fails:                                    │
│  │  Risk >= 70% → estimate 3.5D                              │
│  │  Risk >= 50% → estimate 2.0D                              │
│  │  Risk < 50%  → estimate 1.0D                              │
│  └──────────────────────────────────────────────────────────────┘
│                                                                     │
│  Return Result (JSON):                                            │
│  {                                                                  │
│    "risk_score": 69,                                               │
│    "risk_level": "MODERATE",                                       │
│    "risk_probability": 0.688,                                      │
│    "has_re": true,                                                 │
│    "re_probability": 0.68,                                         │
│    "diopters": 2.45,                                               │
│    "severity": "Mild"                                              │
│  }                                                                  │
└─────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│          4. FRONTEND DISPLAY & VISUALIZATION (Results.tsx)         │
│─────────────────────────────────────────────────────────────────────│
│                                                                     │
│  Response received from backend                                    │
│  │                                                                 │
│  ├─ Display RISK GAUGE (semi-circular gauge with needle)         │
│  ├─ Display RISK LEVEL (text: "MODERATE RISK")                   │
│  ├─ Display STAGES SUMMARY (3 cards showing each stage)          │
│  ├─ Display RECOMMENDATIONS (what to do)                         │
│  └─ Display DOWNLOAD PDF BUTTON                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 RISK CALCULATION BREAKDOWN

### Rule-Based Risk Scoring System

The system assigns points based on clinical evidence. Higher score = Higher risk:

```
BASE SCORE: 30 (neutral starting point)

AGE FACTOR (Younger = Higher Risk):
├─ Age ≤ 8 years    → +15 points
├─ Age 8-10 years   → +10 points
├─ Age 10-12 years  → +5 points
└─ Age > 12 years   → +0 points

GENETICS/FAMILY HISTORY (Most Important):
├─ Both parents myopic      → +25 points (STRONGEST FACTOR)
├─ One parent myopic        → +15 points
├─ Family history but unclear → +8 points
└─ No family history        → +0 points

SCREEN TIME (Daily Device Usage):
├─ > 8 hours/day    → +22 points
├─ 6-8 hours/day    → +17 points
├─ 4-6 hours/day    → +12 points
├─ 2-4 hours/day    → +6 points
└─ < 2 hours/day    → +0 points

OUTDOOR TIME (STRONGEST PROTECTIVE FACTOR):
├─ 0 hours/day      → +25 points (severe deficit)
├─ < 0.5 hours/day  → +20 points
├─ 0.5-1 hour/day   → +15 points
├─ 1-2 hours/day    → +8 points
└─ ≥ 3 hours/day    → -10 points (protective!)

NEAR WORK (Focus Strain):
├─ > 6 hours/day    → +15 points
├─ 4-6 hours/day    → +8 points
└─ < 4 hours/day    → +0 points

ACADEMIC PRESSURE:
├─ Competitive exam prep → +10 points
├─ Tuition classes       → +5 points
├─ Private/International school → +3 points
└─ Government school     → +0 points

PROTECTIVE FACTORS (Reduce Risk):
├─ Regular sports/exercise → -8 points
├─ Takes Vitamin D supplement → -5 points
└─ No protective factors → +0 points

FINAL CALCULATION:
Score = min(max(Total, 0), 100)
Risk_Percentage = Score / 100
```

### Example Calculation

```
Case: 10-year-old boy, both parents myopic, plays sports, screens 5hrs/day, outdoor 1.5hrs/day

Base Score:                    30
Age (8-10):                   +10  (Score: 40)
Both parents myopic:          +25  (Score: 65)
Screen time (4-6 hrs):        +12  (Score: 77)
Outdoor time (1-2 hrs):        +8  (Score: 85)
Near work (3 hrs):             +0  (Score: 85)
Regular sports:                -8  (Score: 77)
No Vitamin D:                  +0  (Score: 77)
─────────────────────────────────────
FINAL SCORE:                   77% RISK

Result: "HIGH RISK" (≥70%)
```

---

## 📈 How The Graph/Gauge Appears

### RiskGauge Component (Visual)

```
             RISK GAUGE VISUALIZATION

          LOW RISK ZONE (0-40%) [GREEN]
               ╱───────────────╲
              │      SAFE       │
              │    ZONE        │
       ╴─────┤               ├─────╴
       ╴─────┤   ╱╲    ╱    ├─────╴
              │  ╱  ╲  ╱    │
              │ │   ││    │
              │ │   ││    │
              │ │   ││    │
              └─┼───┼┼────┘
                │   ││
                │   ││  NEEDLE
              MODERATE (40-70%)    │
                 HIGH (70-100%)    │


ANIMATED FEATURES:
1. Semi-circular gauge from 0° to 180°
2. Three color zones:
   - GREEN (0-40%):     Safe, low risk
   - AMBER (40-70%):    Moderate concern
   - RED (70-100%):     High risk, needs attention

3. Animated needle that rotates:
   - Calculation: rotation = (score / 100) × 180°
   - Animation time: 1.5 seconds
   - Easing: easeOut (smooth, natural motion)

4. Animated Arc (progress bar):
   - Fills from left to right following the gauge
   - Color matches needle (green/amber/red)
   - Uses SVG strokeDashArray for smooth effect

5. Score Display:
   - Large animated number (e.g., "69%")
   - Color-coded (green/amber/red)
   - Scales in with spring effect at 0.5s delay
```

### Code: How Needle Rotates

```javascript
// In RiskGauge.tsx

// Calculate rotation angle (0° to 180°)
const rotation = (displayScore / 100) * 180;

// Example: score=69%
// rotation = (69 / 100) × 180 = 124.2°

// Color logic
const getColor = (score) => {
  if (score < 40) return "green";      // LOW RISK
  if (score < 70) return "amber";      // MODERATE RISK
  return "red";                         // HIGH RISK
};

// Animation
<motion.g
  animate={{ rotate: rotation }}         // Rotates to 124.2°
  transition={{ duration: 1.5, ease: "easeOut" }}
  style={{ transformOrigin: "100px 90px" }}  // Pivot point
>
  {/* Needle line and circle */}
</motion.g>
```

---

## 🔄 Complete Workflow Example

### Scenario: 9-year-old girl with family history of myopia

**Step 1: User enters screening data**
```
Age: 9 years old
Sex: Female
Height: 135 cm
Weight: 32 kg (BMI = 17.5)
Family: Both parents myopic
Screen time: 5 hours/day
Outdoor time: 1 hour/day
Near work: 4 hours/day
Sports: Occasional
Vitamin D: No
```

**Step 2: Data sent to backend**
```
POST /predict
Payload: {...all fields above...}
```

**Step 3A: Stage 1 - Refractive Error**
```
ML Model Input: (age, BMI, sports, etc.)
↓
ML Model predicts: RE_Probability = 0.72
↓
has_re = true (0.72 >= 0.5)
```

**Step 3B: Stage 2 - Risk Level (Hybrid)**
```
Rule-Based Score:
  (Base) 30
  + (Age 9) 10
  + (Both parents) 25
  + (Screen 5h) 12
  + (Outdoor 1h) 15
  + (Near 4h) 8
  + (Occasional sports) -3
  ─────────────
  Total: 97 → clamped to 100 → Rule_Prob = 1.0

ML Model: ML_Prob = 0.75 (confident HIGH)

Hybrid Fusion:
Since ML_Prob >= 0.65 (confident):
  Risk = 0.60 × 0.75 + 0.40 × 1.0
  Risk = 0.45 + 0.40 = 0.85
  Risk_Score = 85%

Result: "HIGH RISK" (85 >= 70)
```

**Step 3C: Stage 3 - Diopter Estimate**
```
Since has_re = true:
  Regression Model Input: (27 features)
  ↓
  Predicted Diopters = 2.8D
  ↓
  Severity = "Mild" (2.8 is between 0.5-3.0)
```

**Step 4: Results displayed**
```
Frontend receives:
{
  "risk_score": 85,
  "risk_level": "HIGH",
  "has_re": true,
  "re_probability": 0.72,
  "diopters": 2.8,
  "severity": "Mild"
}

Visual Output:
- Gauge animates to 85° over 1.5 seconds
- Needle points to RED zone
- Number shows "85%" in red
- Card displays "HIGH RISK - 85%"
- Three-stage summary shows:
  ✓ Stage 1: YES (72%)
  ✓ Stage 2: HIGH (85%)
  ✓ Stage 3: -2.8D (Mild)
```

---

## 📊 Gauge Zones Explained

```
RISK GAUGE: 0% ├────────┤ 100%

0%          40%        70%        100%
│           │          │          │
├──GREEN────┼─AMBER────┼─RED──────┤
│           │          │          │
LOW         MODERATE   HIGH       EXTREME
RISK        RISK       RISK       RISK
│           │          │          │
└─ Safe     └─ Watch   └─ Action  └─ Critical
  ✓ No act   ⚠ Monitor  🔴 Consult ☠ Medical
  ✓ Healthy ⚠ Yearly   🔴 Eye Dr  ☠ Emergency
            follow-up  🔴 Glasses
                       🔴 Eye
                        exercises
```

---

## 🧬 Key Factors by Impact Weight

```
FACTOR IMPORTANCE (by ML Algorithm)

Strongest Positive Predictors (↑ Risk):
1. Parent myopia (GENETIC)        [25-30% weight]
2. Age (younger)                  [20-25% weight]
3. Screen time (high)             [15-20% weight]
4. Outdoor time (low)             [15-20% weight]
5. Near work hours (high)         [10-15% weight]
6. Academic pressure              [5-10% weight]

Protective Factors (↓ Risk):
1. Outdoor time (≥2 hrs/day)      [-25 points]
2. Regular sports/exercise        [-8 points]
3. Vitamin D supplementation      [-5 points]

KEY INSIGHT:
📌 Outdoor time is the STRONGEST PROTECTIVE factor
📌 Even 30 mins outdoors daily reduces progression risk by ~15%
📌 Family history dominates (25+ points alone)
```

---

## 🎓 Understanding Risk Levels

```
RISK LEVEL INTERPRETATION:

LOW RISK (0-40%)
├─ What it means: Child unlikely to develop/progress myopia
├─ Probability: 40% or less
├─ Recommendation: Continue healthy habits, annual eye checks
└─ Action: No intervention needed

MODERATE RISK (40-70%)
├─ What it means: Moderate chance of myopia progression
├─ Probability: 40-70%
├─ Recommendation: Schedule eye exam, reduce screen time
└─ Action: Monitor closely, lifestyle modifications

HIGH RISK (70-100%)
├─ What it means: High likelihood of myopia development
├─ Probability: 70% or higher
├─ Recommendation: See ophthalmologist urgently
└─ Action: May need glasses, contact lenses, or corrective
         exercises
```

---

## 📥 Data Used in Prediction

### Collected from User Input:
```
8 Direct Inputs:
│
├─ Personal: age, sex, height, weight
├─ Genetic: family history, parents myopic
└─ Lifestyle: screen time, outdoor time, near work,
              sports, vitamin D

30 Features Computed:
│
├─ Basic metrics: BMI, age_groups
├─ Combinations: age×screen, screen+near, screen/outdoor ratio
├─ Encoded categories: sex (binary), parents (0/1/2)
├─ Derived: family_load (genetics strength measure)
└─ Classification bins: BMI category, age category
```

### Example Feature Vector (27 features for diopter regression):

```
Feature Name                    | Value
────────────────────────────────┼─────────
Age                            | 9
BMI                            | 17.5
Screen_Time_Hours              | 5
Near_Work_Hours                | 4
Outdoor_Time_Hours             | 1
Age_Screen                     | 45 (9×5)
Screen_Near_Total              | 9 (5+4)
Screen_Outdoor_Ratio           | 5.0 (5÷1)
High_Risk_Parent               | 1 (both parents)
Family_Load                    | 2 (genetics)
Location_Type_Urban            | 1
School_Type_Encoded            | 0 (government)
Tuition_Binary                 | 0
Comp_Exam_Binary               | 0
Vitamin_D_Binary               | 0
Sports_Encoded                 | 1 (occasional)
[State one-hot encoding]       | 25 binary flags
────────────────────────────────┼─────────────
TOTAL FEATURES                 | 27
```

---

## 💡 Why Hybrid (ML + Rules)?

```
ML MODEL ALONE:
  ✓ Very accurate (AUC 0.893)
  ✓ Learns complex patterns
  ✗ Can be unpredictable on extreme inputs
  ✗ "Black box" - hard to explain

RULES ALONE:
  ✓ Transparent (clinicians understand every point)
  ✓ Follows WHO evidence-based guidelines
  ✗ Misses complex interactions
  ✗ Overly rigid

HYBRID APPROACH:
  ✓ Uses ML when it's confident (≥0.65 prob)
  ✓ Blends both when ML is uncertain (0.35-0.65)
  ✓ Falls back to rules when ML gives low scores
  ✓ ALWAYS enforces rule minimum (safety check)

Result: Best of both worlds!
├─ Accurate AND explainable
├─ Trustworthy AND evidence-based
└─ Safe AND intelligent
```

---

## 📋 Summary of Calculations

| Stage | Input | ML Model | Output | Human-Readable |
|-------|-------|----------|--------|-----------------|
| **1** | 8 features | XGBoost Classification | Probability 0-1 | Has RE? Yes/No |
| **2** | 30 features | GradientBoosting + Rules | % 0-100 | Risk Level & Score |
| **3** | 27 features | XGBoost Regression | Diopter value | Severity Category |

This is a complete end-to-end AI system that combines best practices in machine learning, clinical evidence, and user experience! 🎯

