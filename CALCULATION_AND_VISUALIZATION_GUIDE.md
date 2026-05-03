# ðŸ§® Myopia Risk Prediction: Complete Calculation & Visualization Guide

## ðŸ“Š Complete Data Flow

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚                    1. DATA COLLECTION (Screening)                   â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
â”‚  Screen.tsx - 3-Step Questionnaire                                  â”‚
â”‚                                                                     â”‚
â”‚  Step 1: PERSONAL INFO                                             â”‚
â”‚  â”œâ”€ Age (6-18 years)                                              â”‚
â”‚  â”œâ”€ Sex (Male/Female)                                             â”‚
â”‚  â”œâ”€ Height (cm)                                                   â”‚
â”‚  â””â”€ Weight (kg) â†’ calculates BMI                                 â”‚
â”‚                                                                     â”‚
â”‚  Step 2: FAMILY HISTORY                                           â”‚
â”‚  â”œâ”€ Has myopia in family? (Yes/No)                               â”‚
â”‚  â””â”€ How many parents myopic? (None/One/Both)                    â”‚
â”‚                                                                     â”‚
â”‚  Step 3: LIFESTYLE FACTORS                                        â”‚
â”‚  â”œâ”€ Screen time per day (0-12 hours) [SLIDER]                   â”‚
â”‚  â”œâ”€ Near work per day (0-12 hours) [SLIDER]                     â”‚
â”‚  â”œâ”€ Outdoor time per day (0-8 hours) [SLIDER]                   â”‚
â”‚  â”œâ”€ Sports frequency (Rare/Occasional/Regular)                   â”‚
â”‚  â””â”€ Vitamin D supplementation? (Yes/No)                          â”‚
â”‚                                                                     â”‚
â”‚  User clicks SUBMIT â†’ Data stored in sessionStorage               â”‚
â”‚  User navigated to /results page                                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

                              â†“

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚              2. DATA TRANSMISSION TO BACKEND (API)                  â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
â”‚  POST http://localhost:5001/predict                               â”‚
â”‚                                                                     â”‚
â”‚  Payload (JSON):                                                   â”‚
â”‚  {                                                                  â”‚
â”‚    "age": 10,                                                       â”‚
â”‚    "sex": "male",                                                   â”‚
â”‚    "height": 145,                                                   â”‚
â”‚    "weight": 38,                                                    â”‚
â”‚    "familyHistory": true,                                           â”‚
â”‚    "parentsMyopic": "one",                                          â”‚
â”‚    "screenTime": 5,                                                 â”‚
â”‚    "nearWork": 3,                                                   â”‚
â”‚    "outdoorTime": 1.5,                                              â”‚
â”‚    "sports": "occasional",                                          â”‚
â”‚    "vitaminD": false                                                â”‚
â”‚  }                                                                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

                              â†“

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚               3. BACKEND ML PROCESSING (3-STAGE PIPELINE)           â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
â”‚                                                                     â”‚
â”‚  Input Validation:                                                 â”‚
â”‚  â”œâ”€ Check data types (age 6-18, screen time 0-24, etc.)          â”‚
â”‚  â”œâ”€ Check ranges and missing values                               â”‚
â”‚  â””â”€ Return 400 error if invalid                                   â”‚
â”‚                                                                     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  â”‚ STAGE 1: REFRACTIVE ERROR DETECTION                          â”‚
â”‚  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  â”‚ Question: Does the child have refractive error (RE)?        â”‚
â”‚  â”‚ Model: XGBoost Classification                               â”‚
â”‚  â”‚ Input: 8 features (age, BMI, family history, etc.)         â”‚
â”‚  â”‚ Output: Probability (0-1)                                   â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ re_probability = model.predict_proba()[1]                  â”‚
â”‚  â”‚ has_re = (re_probability >= 0.5)                           â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ Example: RE_Prob = 0.68 â†’ Child LIKELY has RE              â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚                                                                     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  â”‚ STAGE 2: PROGRESSION RISK ASSESSMENT (HYBRID)               â”‚
â”‚  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  â”‚ Question: What is the progression risk?                    â”‚
â”‚  â”‚ Answer: LOW (0-40%) | MODERATE (40-70%) | HIGH (70-100%)  â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ TWO PARALLEL METHODS:                                       â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ A) ML MODEL (GradientBoosting)                              â”‚
â”‚  â”‚    â”œâ”€ AUC: 0.893 (very accurate)                           â”‚
â”‚  â”‚    â”œâ”€ Trained on 5000 real screening records              â”‚
â”‚  â”‚    â”œâ”€ Input: 30 clinical features                          â”‚
â”‚  â”‚    â””â”€ Output: ML_Probability (0-1)                         â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ B) RULE-BASED SCORING (Evidence-Based WHO Guidelines)     â”‚
â”‚  â”‚    â”œâ”€ Base Score: 30 (neutral starting point)             â”‚
â”‚  â”‚    â”œâ”€ Add points based on risk factors (see below)        â”‚
â”‚  â”‚    â””â”€ Output: Rule_Probability (0-1)                       â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ ADAPTIVE HYBRID FUSION:                                    â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ if ML_Prob >= 0.65:        (ML confident HIGH)             â”‚
â”‚  â”‚    Risk = 0.60Ã—ML + 0.40Ã—Rule  (Trust ML 60%)            â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ elif ML_Prob >= 0.35:      (ML neutral)                    â”‚
â”‚  â”‚    Risk = 0.50Ã—ML + 0.50Ã—Rule  (50/50 blend)            â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ else:                       (ML giving LOW)                â”‚
â”‚  â”‚    Risk = 0.20Ã—ML + 0.80Ã—Rule  (Lean on rules 80%)      â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ Floor: Risk = max(Risk, 0.75Ã—Rule)  (safety check)        â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ Risk_Percentage = int(Risk Ã— 100)                          â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ Example:                                                    â”‚
â”‚  â”‚  ML_Prob = 0.72 (confident HIGH)                          â”‚
â”‚  â”‚  Rule_Prob = 0.65 (also HIGH)                             â”‚
â”‚  â”‚  Risk = 0.60 Ã— 0.72 + 0.40 Ã— 0.65 = 0.688                â”‚
â”‚  â”‚  Risk_Score = 69% â†’ MODERATE RISK                         â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚                                                                     â”‚
â”‚  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  â”‚ STAGE 3: DIOPTER SEVERITY ESTIMATE (Regression)            â”‚
â”‚  â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  â”‚ Question: How severe is the myopia? (in diopters)         â”‚
â”‚  â”‚ Only calculated if Stage 1 = TRUE (has RE)                â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ Model: XGBoost Regression                                  â”‚
â”‚  â”‚ Input: 27 features (subset of all features)               â”‚
â”‚  â”‚ Output: Diopter value (absolute value)                     â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ Diopter Examples:                                           â”‚
â”‚  â”‚  0.0 to 0.5D â†’ Negligible                                  â”‚
â”‚  â”‚  0.5 to 3.0D â†’ Mild       (minor correction needed)        â”‚
â”‚  â”‚  3.0 to 6.0D â†’ Moderate   (noticeable problem)             â”‚
â”‚  â”‚  > 6.0D      â†’ High       (significant correction)         â”‚
â”‚  â”‚                                                              â”‚
â”‚  â”‚ Fallback if model fails:                                    â”‚
â”‚  â”‚  Risk >= 70% â†’ estimate 3.5D                              â”‚
â”‚  â”‚  Risk >= 50% â†’ estimate 2.0D                              â”‚
â”‚  â”‚  Risk < 50%  â†’ estimate 1.0D                              â”‚
â”‚  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚                                                                     â”‚
â”‚  Return Result (JSON):                                            â”‚
â”‚  {                                                                  â”‚
â”‚    "risk_score": 69,                                               â”‚
â”‚    "risk_level": "MODERATE",                                       â”‚
â”‚    "risk_probability": 0.688,                                      â”‚
â”‚    "has_re": true,                                                 â”‚
â”‚    "re_probability": 0.68,                                         â”‚
â”‚    "diopters": 2.45,                                               â”‚
â”‚    "severity": "Mild"                                              â”‚
â”‚  }                                                                  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜

                              â†“

â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚          4. FRONTEND DISPLAY & VISUALIZATION (Results.tsx)         â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
â”‚                                                                     â”‚
â”‚  Response received from backend                                    â”‚
â”‚  â”‚                                                                 â”‚
â”‚  â”œâ”€ Display RISK GAUGE (semi-circular gauge with needle)         â”‚
â”‚  â”œâ”€ Display RISK LEVEL (text: "MODERATE RISK")                   â”‚
â”‚  â”œâ”€ Display STAGES SUMMARY (3 cards showing each stage)          â”‚
â”‚  â”œâ”€ Display RECOMMENDATIONS (what to do)                         â”‚
â”‚  â””â”€ Display DOWNLOAD PDF BUTTON                                  â”‚
â”‚                                                                     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

---

## ðŸŽ¯ RISK CALCULATION BREAKDOWN

### Rule-Based Risk Scoring System

The system assigns points based on clinical evidence. Higher score = Higher risk:

```
BASE SCORE: 30 (neutral starting point)

AGE FACTOR (Younger = Higher Risk):
â”œâ”€ Age â‰¤ 8 years    â†’ +15 points
â”œâ”€ Age 8-10 years   â†’ +10 points
â”œâ”€ Age 10-12 years  â†’ +5 points
â””â”€ Age > 12 years   â†’ +0 points

GENETICS/FAMILY HISTORY (Most Important):
â”œâ”€ Both parents myopic      â†’ +25 points (STRONGEST FACTOR)
â”œâ”€ One parent myopic        â†’ +15 points
â”œâ”€ Family history but unclear â†’ +8 points
â””â”€ No family history        â†’ +0 points

SCREEN TIME (Daily Device Usage):
â”œâ”€ > 8 hours/day    â†’ +22 points
â”œâ”€ 6-8 hours/day    â†’ +17 points
â”œâ”€ 4-6 hours/day    â†’ +12 points
â”œâ”€ 2-4 hours/day    â†’ +6 points
â””â”€ < 2 hours/day    â†’ +0 points

OUTDOOR TIME (STRONGEST PROTECTIVE FACTOR):
â”œâ”€ 0 hours/day      â†’ +25 points (severe deficit)
â”œâ”€ < 0.5 hours/day  â†’ +20 points
â”œâ”€ 0.5-1 hour/day   â†’ +15 points
â”œâ”€ 1-2 hours/day    â†’ +8 points
â””â”€ â‰¥ 3 hours/day    â†’ -10 points (protective!)

NEAR WORK (Focus Strain):
â”œâ”€ > 6 hours/day    â†’ +15 points
â”œâ”€ 4-6 hours/day    â†’ +8 points
â””â”€ < 4 hours/day    â†’ +0 points

ACADEMIC PRESSURE:
â”œâ”€ Competitive exam prep â†’ +10 points
â”œâ”€ Tuition classes       â†’ +5 points
â”œâ”€ Private/International school â†’ +3 points
â””â”€ Government school     â†’ +0 points

PROTECTIVE FACTORS (Reduce Risk):
â”œâ”€ Regular sports/exercise â†’ -8 points
â”œâ”€ Takes Vitamin D supplement â†’ -5 points
â””â”€ No protective factors â†’ +0 points

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
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FINAL SCORE:                   77% RISK

Result: "HIGH RISK" (â‰¥70%)
```

---

## ðŸ“ˆ How The Graph/Gauge Appears

### RiskGauge Component (Visual)

```
             RISK GAUGE VISUALIZATION

          LOW RISK ZONE (0-40%) [GREEN]
               â•±â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â•²
              â”‚      SAFE       â”‚
              â”‚    ZONE        â”‚
       â•´â”€â”€â”€â”€â”€â”¤               â”œâ”€â”€â”€â”€â”€â•´
       â•´â”€â”€â”€â”€â”€â”¤   â•±â•²    â•±    â”œâ”€â”€â”€â”€â”€â•´
              â”‚  â•±  â•²  â•±    â”‚
              â”‚ â”‚   â”‚â”‚    â”‚
              â”‚ â”‚   â”‚â”‚    â”‚
              â”‚ â”‚   â”‚â”‚    â”‚
              â””â”€â”¼â”€â”€â”€â”¼â”¼â”€â”€â”€â”€â”˜
                â”‚   â”‚â”‚
                â”‚   â”‚â”‚  NEEDLE
              MODERATE (40-70%)    â”‚
                 HIGH (70-100%)    â”‚


ANIMATED FEATURES:
1. Semi-circular gauge from 0Â° to 180Â°
2. Three color zones:
   - GREEN (0-40%):     Safe, low risk
   - AMBER (40-70%):    Moderate concern
   - RED (70-100%):     High risk, needs attention

3. Animated needle that rotates:
   - Calculation: rotation = (score / 100) Ã— 180Â°
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

// Calculate rotation angle (0Â° to 180Â°)
const rotation = (displayScore / 100) * 180;

// Example: score=69%
// rotation = (69 / 100) Ã— 180 = 124.2Â°

// Color logic
const getColor = (score) => {
  if (score < 40) return "green";      // LOW RISK
  if (score < 70) return "amber";      // MODERATE RISK
  return "red";                         // HIGH RISK
};

// Animation
<motion.g
  animate={{ rotate: rotation }}         // Rotates to 124.2Â°
  transition={{ duration: 1.5, ease: "easeOut" }}
  style={{ transformOrigin: "100px 90px" }}  // Pivot point
>
  {/* Needle line and circle */}
</motion.g>
```

---

## ðŸ”„ Complete Workflow Example

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
â†“
ML Model predicts: RE_Probability = 0.72
â†“
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
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  Total: 97 â†’ clamped to 100 â†’ Rule_Prob = 1.0

ML Model: ML_Prob = 0.75 (confident HIGH)

Hybrid Fusion:
Since ML_Prob >= 0.65 (confident):
  Risk = 0.60 Ã— 0.75 + 0.40 Ã— 1.0
  Risk = 0.45 + 0.40 = 0.85
  Risk_Score = 85%

Result: "HIGH RISK" (85 >= 70)
```

**Step 3C: Stage 3 - Diopter Estimate**
```
Since has_re = true:
  Regression Model Input: (27 features)
  â†“
  Predicted Diopters = 2.8D
  â†“
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
- Gauge animates to 85Â° over 1.5 seconds
- Needle points to RED zone
- Number shows "85%" in red
- Card displays "HIGH RISK - 85%"
- Three-stage summary shows:
  âœ“ Stage 1: YES (72%)
  âœ“ Stage 2: HIGH (85%)
  âœ“ Stage 3: -2.8D (Mild)
```

---

## ðŸ“Š Gauge Zones Explained

```
RISK GAUGE: 0% â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”¤ 100%

0%          40%        70%        100%
â”‚           â”‚          â”‚          â”‚
â”œâ”€â”€GREENâ”€â”€â”€â”€â”¼â”€AMBERâ”€â”€â”€â”€â”¼â”€REDâ”€â”€â”€â”€â”€â”€â”¤
â”‚           â”‚          â”‚          â”‚
LOW         MODERATE   HIGH       EXTREME
RISK        RISK       RISK       RISK
â”‚           â”‚          â”‚          â”‚
â””â”€ Safe     â””â”€ Watch   â””â”€ Action  â””â”€ Critical
  âœ“ No act   âš  Monitor  ðŸ”´ Consult â˜  Medical
  âœ“ Healthy âš  Yearly   ðŸ”´ Eye Dr  â˜  Emergency
            follow-up  ðŸ”´ Glasses
                       ðŸ”´ Eye
                        exercises
```

---

## ðŸ§¬ Key Factors by Impact Weight

```
FACTOR IMPORTANCE (by ML Algorithm)

Strongest Positive Predictors (â†‘ Risk):
1. Parent myopia (GENETIC)        [25-30% weight]
2. Age (younger)                  [20-25% weight]
3. Screen time (high)             [15-20% weight]
4. Outdoor time (low)             [15-20% weight]
5. Near work hours (high)         [10-15% weight]
6. Academic pressure              [5-10% weight]

Protective Factors (â†“ Risk):
1. Outdoor time (â‰¥2 hrs/day)      [-25 points]
2. Regular sports/exercise        [-8 points]
3. Vitamin D supplementation      [-5 points]

KEY INSIGHT:
ðŸ“Œ Outdoor time is the STRONGEST PROTECTIVE factor
ðŸ“Œ Even 30 mins outdoors daily reduces progression risk by ~15%
ðŸ“Œ Family history dominates (25+ points alone)
```

---

## ðŸŽ“ Understanding Risk Levels

```
RISK LEVEL INTERPRETATION:

LOW RISK (0-40%)
â”œâ”€ What it means: Child unlikely to develop/progress myopia
â”œâ”€ Probability: 40% or less
â”œâ”€ Recommendation: Continue healthy habits, annual eye checks
â””â”€ Action: No intervention needed

MODERATE RISK (40-70%)
â”œâ”€ What it means: Moderate chance of myopia progression
â”œâ”€ Probability: 40-70%
â”œâ”€ Recommendation: Schedule eye exam, reduce screen time
â””â”€ Action: Monitor closely, lifestyle modifications

HIGH RISK (70-100%)
â”œâ”€ What it means: High likelihood of myopia development
â”œâ”€ Probability: 70% or higher
â”œâ”€ Recommendation: See ophthalmologist urgently
â””â”€ Action: May need glasses, contact lenses, or corrective
         exercises
```

---

## ðŸ“¥ Data Used in Prediction

### Collected from User Input:
```
8 Direct Inputs:
â”‚
â”œâ”€ Personal: age, sex, height, weight
â”œâ”€ Genetic: family history, parents myopic
â””â”€ Lifestyle: screen time, outdoor time, near work,
              sports, vitamin D

30 Features Computed:
â”‚
â”œâ”€ Basic metrics: BMI, age_groups
â”œâ”€ Combinations: ageÃ—screen, screen+near, screen/outdoor ratio
â”œâ”€ Encoded categories: sex (binary), parents (0/1/2)
â”œâ”€ Derived: family_load (genetics strength measure)
â””â”€ Classification bins: BMI category, age category
```

### Example Feature Vector (27 features for diopter regression):

```
Feature Name                    | Value
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€
Age                            | 9
BMI                            | 17.5
Screen_Time_Hours              | 5
Near_Work_Hours                | 4
Outdoor_Time_Hours             | 1
Age_Screen                     | 45 (9Ã—5)
Screen_Near_Total              | 9 (5+4)
Screen_Outdoor_Ratio           | 5.0 (5Ã·1)
High_Risk_Parent               | 1 (both parents)
Family_Load                    | 2 (genetics)
Location_Type_Urban            | 1
School_Type_Encoded            | 0 (government)
Tuition_Binary                 | 0
Comp_Exam_Binary               | 0
Vitamin_D_Binary               | 0
Sports_Encoded                 | 1 (occasional)
[State one-hot encoding]       | 25 binary flags
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TOTAL FEATURES                 | 27
```

---

## ðŸ’¡ Why Hybrid (ML + Rules)?

```
ML MODEL ALONE:
  âœ“ Very accurate (AUC 0.893)
  âœ“ Learns complex patterns
  âœ— Can be unpredictable on extreme inputs
  âœ— "Black box" - hard to explain

RULES ALONE:
  âœ“ Transparent (clinicians understand every point)
  âœ“ Follows WHO evidence-based guidelines
  âœ— Misses complex interactions
  âœ— Overly rigid

HYBRID APPROACH:
  âœ“ Uses ML when it's confident (â‰¥0.65 prob)
  âœ“ Blends both when ML is uncertain (0.35-0.65)
  âœ“ Falls back to rules when ML gives low scores
  âœ“ ALWAYS enforces rule minimum (safety check)

Result: Best of both worlds!
â”œâ”€ Accurate AND explainable
â”œâ”€ Trustworthy AND evidence-based
â””â”€ Safe AND intelligent
```

---

## ðŸ“‹ Summary of Calculations

| Stage | Input | ML Model | Output | Human-Readable |
|-------|-------|----------|--------|-----------------|
| **1** | 8 features | XGBoost Classification | Probability 0-1 | Has RE? Yes/No |
| **2** | 30 features | GradientBoosting + Rules | % 0-100 | Risk Level & Score |
| **3** | 27 features | XGBoost Regression | Diopter value | Severity Category |

This is a complete end-to-end AI system that combines best practices in machine learning, clinical evidence, and user experience! ðŸŽ¯

