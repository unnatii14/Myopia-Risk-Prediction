# Myopia Risk Prediction System
### AI-Powered Pediatric Myopia Screening for School Children — End-to-End Project Workflow

> **Purpose of this file:** Living project guide. Follow section by section. Tick off items as done.
> **Clinical Advisor:** Doctor (Ophthalmologist / Optometrist)
> **Research Base:** LVPEI · bmjophth-2023 · ijerph-2020 · pone.0250468
> **Project Title:** AI-Based Myopia Risk Prediction Model for School Children (Jalpesh Sir)

---

## Project Goal

Shift from **reactive detection → proactive risk prediction** before refractive changes occur.
Build a web-based screening tool usable by schools, community health workers, and parents — no ophthalmology equipment required. Input: lifestyle + demographic data. Output: risk score + personalized clinical recommendations.

---

## Overall Status Tracker

| Phase | What | Status |
|---|---|---|
| Phase 0 | ML Models (notebook) | ✅ Done |
| Phase 1 | Website Frontend (React) | 🔲 Build Now |
| Phase 2 | FastAPI Backend | 🔲 Build Now |
| Phase 3 | 5-Year Progression Model | 🔲 After data arrives |
| Phase 4 | Clinical Validation + Paper | 🔲 Future |

---

## Phase 0 — ML Work Completed ✅

### Dataset
- 5,000 pediatric records (`Myopia_Dataset_5000.csv`)
- 80/20 train-test split (4,000 train / 1,000 test)
- 35 leakage-free features — post-diagnosis columns removed

### Three-Stage Pipeline

```
Patient lifestyle + demographic input
         │
         ▼
┌──────────────────────────┐
│ Stage 1 — Has RE?        │  XGBoost Classifier   AUC: 0.50 ⚠️
│ Refractive error present?│  (needs visual acuity data to improve)
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ Stage 2 — High Risk?     │  XGBoost Classifier   AUC: 0.88 ✅
│ Progression risk level   │  Accuracy: 81.2%  F1: 0.57
└───────────┬──────────────┘  ← PRODUCTION READY
            ▼ (if RE likely)
┌──────────────────────────┐
│ Stage 3 — Diopter Sev.   │  Gradient Boosting    MAE: 1.75 D ⚠️
│ Estimated diopters       │  (approximate only)
└──────────────────────────┘
```

### Completed Steps in Notebook
- [x] Data exploration, missing value check, class distribution
- [x] Data visualization (RE presence, risk categories, age distribution)
- [x] Feature engineering (35 features, 5 composite features)
- [x] Four models trained: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- [x] GridSearchCV hyperparameter tuning (5-fold CV)
- [x] Data leakage detected and fixed (`Degree_RE_Diopters` removed)
- [x] Stage 1 Has_RE classifier trained
- [x] Stage 3 Diopter regressor trained (on RE-positive children only)
- [x] Permutation importance explainability
- [x] Individual patient waterfall explanation
- [x] Production pipeline function `myopia_full_pipeline()`
- [x] All artifacts saved to `models/`

### Saved Artifacts (`models/`)
```
risk_progression_model.pkl      ← Stage 2 main classifier
has_re_model.pkl                ← Stage 1 screening gate
diopter_regression_model.pkl    ← Stage 3 severity estimator
scaler_classification.pkl       ← StandardScaler for classifiers
scaler_regression.pkl           ← StandardScaler for regressor
feature_columns.json            ← exact input column order for API
model_metadata.json             ← AUC, F1, accuracy, config
```

### Model Performance
| Stage | Model | Metric | Value |
|---|---|---|---|
| Stage 2 — Risk | XGBoost | ROC-AUC | **0.8814** ✅ |
| Stage 2 — Risk | XGBoost | Accuracy | **81.2%** ✅ |
| Stage 2 — Risk | XGBoost | F1-Score | 0.5668 |
| Stage 1 — Has RE | XGBoost | ROC-AUC | 0.5019 ⚠️ |
| Stage 3 — Diopter | GBR | MAE | 1.748 D ⚠️ |
| Stage 3 — Diopter | GBR | R² | -0.146 ⚠️ |

> ⚠️ Stage 1 and Stage 3 are approximate — show disclaimer on website.
> ✅ Stage 2 is the only model safe for production use currently.

---

## Features — Final Decision Table (Doctor-Confirmed)

> ❌ Doctor confirmed: Do NOT use Rural vs Urban — imbalanced data.

### FORM INPUTS (fed into model)

#### Block 1 — Child Info
| Field | Input Type | Model Column |
|---|---|---|
| Age | Slider 5–18 | `Age` |
| Sex | Male / Female | `Sex` (1/0) |
| Height (cm) | Number | `Height_cm` |
| Weight (kg) | Number | `Weight_kg` → auto BMI |
| State | Dropdown 10 states | `State_Encoded` + one-hot |
| ~~Location Type~~ | ~~Rural/Urban~~ | ❌ REMOVED — imbalanced |

#### Block 2 — Family / Genetic
| Field | Input Type | Model Column |
|---|---|---|
| Family history of myopia | Yes / No | `Family_History_Myopia` |
| Parents with myopia | None / One / Both | `Parents_With_Myopia` (0/1/2) |

#### Block 3 — Daily Habits
| Field | Input Type | Model Column |
|---|---|---|
| Screen time (hrs/day) | Slider 0–16 | `Screen_Time_Hours` |
| Near work hrs (hrs/day) | Slider 0–12 | `Near_Work_Hours` |
| Outdoor time (hrs/day) | Slider 0–10 | `Outdoor_Time_Hours` |
| Sports participation | Regular/Occasional/Rare | `Sports_Participation` (2/1/0) |
| Vitamin D supplement | Yes / No | `Vitamin_D_Supplementation` |

#### Block 4 — Academic Pressure
| Field | Input Type | Model Column |
|---|---|---|
| School type | Govt / Private / International | `School_Type` (0/1/2) |
| Tuition / coaching | Yes / No | `Tuition_Classes` |
| Competitive exam prep | Yes / No | `Competitive_Exam_Prep` |

### AUTO-CALCULATED FEATURES (backend only, not shown to user)
| Feature | Formula |
|---|---|
| `Screen_Near_Work` | Screen Time + Near Work Hours |
| `Outdoor_Activity_Role` | Outdoor Time × Sports Participation |
| `Digital_Exposure` | Screen Time ÷ (Outdoor Time + 0.1) |
| `Academic_Stress` | Tuition × Competitive Exam Prep |
| `Risk_Score` | Weighted composite of all major risk factors |

### CORRECTION METHOD (Output/Recommendation only — NOT a model input)
> Doctor's requirement: educate parents about all correction types on results page.

| Correction | Show | What to tell parent |
|---|---|---|
| No correction | ✅ | Baseline — no protection |
| Regular glasses | ✅ | Vision correction only — no myopia control |
| **Special glasses** (myopia control / Ortho-K) | ✅ | Slows progression ~30% |
| Contact lenses (regular soft) | ✅ | Vision correction only |
| **Atropine 0.01% eye drops** | ✅ | Gold standard — slows progression ~50–60% |
| Atropine 0.05% / 1% | ✅ | Higher dose — discuss side effects with doctor |

---

## Phase 1 — Website Frontend (Build Now)

### Tech Stack
| Layer | Technology |
|---|---|
| Framework | React 18 + Vite |
| Styling | TailwindCSS |
| Form | React Hook Form — multi-step wizard |
| Charts | Recharts (risk gauge + bar chart) |
| Routing | React Router v6 |
| HTTP | Axios |

### Pages
```
/           Landing page — what is myopia + call to action
/screen     4-step screening form wizard
/results    Risk report page
/about      Research backing + doctor info + LVPEI reference
/faq        Patient education — causes, prevention, corrections
```

### Form — 4-Step Wizard
```
Step 1 → Child Info     (Age, Sex, Height, Weight, State)
Step 2 → Family History (family myopia, parents with myopia)
Step 3 → Daily Habits   (screen time, outdoor time, sports sliders)
Step 4 → School Info    (school type, tuition, exam prep)
          ↓
     [Get Risk Score]
          ↓
     /results page
```

### Results Page Layout
```
┌──────────────────────────────────────────────────────────┐
│          MYOPIA RISK ASSESSMENT REPORT                   │
│  Child: Age X · [State]                   [Download PDF] │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  RISK SCORE GAUGE                                        │
│  ●●●●●●●●●●○○   87%   [ HIGH RISK ]                     │
│                                                          │
│  Stage 1 — Refractive Error Likely?  YES  (72%)          │
│  Stage 2 — Progression Risk?         HIGH RISK (87%)     │
│  Stage 3 — Estimated Severity?       ~3.2 D (Moderate)  │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  WHY THIS SCORE                                          │
│  🔴 Screen time 9 hrs/day     (limit: 2 hrs/day)        │
│  🔴 Both parents have myopia  (genetic risk elevated)    │
│  🔴 Outdoor time 0.3 hrs/day  (need: ≥ 2 hrs/day)      │
│  🟢 Taking Vitamin D supplement (protective factor)      │
├──────────────────────────────────────────────────────────┤
│  ⚠️  CLINICAL FLAGS                                      │
│  · Age ≤10 + HIGH RISK → eye exam every 3 months        │
│  · No myopia control currently in use                    │
├──────────────────────────────────────────────────────────┤
│  💡 RECOMMENDATIONS                                      │
│  🔴 Refer to ophthalmologist — do not delay              │
│  📱 Limit screen time to <2 hrs/day (20-20-20 rule)     │
│  ☀️  At least 2 hrs outdoor daily                       │
│  💊 Ask doctor about Atropine 0.01% eye drops           │
│  👓 Ask about myopia control spectacle lenses            │
├──────────────────────────────────────────────────────────┤
│  CORRECTION GUIDE (Education)                            │
│  Regular glasses    → vision only, no progression ctrl   │
│  Special glasses    → slows progression ~30%             │
│  Atropine 0.01%     → slows progression ~50–60%  ⭐      │
│  Ortho-K lenses     → night-wear, strong control         │
└──────────────────────────────────────────────────────────┘
```

### Frontend Tasks
- [ ] Scaffold Vite + React + Tailwind in `frontend/`
- [ ] Build landing page with myopia education content
- [ ] Build 4-step form wizard with validation
- [ ] Build results page with risk gauge chart
- [ ] Build correction education section (atropine, special glasses)
- [ ] Connect to API (mock response first, then real)
- [ ] Add print / download PDF button
- [ ] Build /about page (LVPEI reference, research links)
- [ ] Build /faq page (patient education)
- [ ] Mobile responsive pass

---

## Phase 2 — FastAPI Backend (Build Now)

### Folder: `backend/`
```
backend/
├── main.py          ← FastAPI app, CORS, routes
├── schemas.py       ← Pydantic request/response models
├── pipeline.py      ← myopia_full_pipeline() logic
├── requirements.txt
└── Dockerfile
```

### Endpoints
```
POST /predict     → full 3-stage pipeline, returns JSON
GET  /health      → health check
GET  /model-info  → returns model_metadata.json
```

### Request Body (`/predict`)
```json
{
  "age": 10,
  "sex": 1,
  "height_cm": 138,
  "weight_kg": 35,
  "bmi_category": 1,
  "state": "Maharashtra",
  "family_history_myopia": 1,
  "parents_with_myopia": 2,
  "screen_time_hours": 9,
  "near_work_hours": 5,
  "outdoor_time_hours": 0.3,
  "sports_participation": 0,
  "vitamin_d_supplementation": 0,
  "school_type": 2,
  "tuition_classes": 1,
  "competitive_exam_prep": 1
}
```

### Response Body (`/predict`)
```json
{
  "re_probability": 0.72,
  "re_screening": "LIKELY HAS RE",
  "risk_probability": 0.87,
  "risk_percentage": "87.0%",
  "risk_category": "HIGH RISK",
  "diopter_estimate": 3.2,
  "severity": "Moderate (3–6 D)",
  "shap_drivers": ["Screen_Near_Work", "Risk_Score", "Family_History_Myopia"],
  "shap_protectors": ["Vitamin_D_Supplementation"],
  "clinical_flags": ["Age ≤10 with HIGH risk — monitor every 3 months"],
  "recommendations": [
    "Refer to ophthalmologist immediately",
    "Cap screen time to <2 hrs/day",
    "Increase outdoor time to ≥2 hrs/day",
    "Ask doctor about Atropine 0.01% eye drops"
  ]
}
```

### Backend Tasks
- [ ] Set up FastAPI project with Pydantic models
- [ ] Load all `.pkl` artifacts on startup
- [ ] Implement `_encode_patient()` to derive composite features
- [ ] Implement `/predict` endpoint calling `myopia_full_pipeline()`
- [ ] Add CORS middleware (allow React frontend)
- [ ] Input validation (age 5–18, hours 0–24, etc.)
- [ ] Write unit tests
- [ ] Dockerize
- [ ] Deploy to Render / Railway

---

## Phase 3 — 5-Year Progression Model (After Data Arrives)

> Waiting for LVPEI / research longitudinal dataset showing how a child's diopter changes over 5 years.

### What it does
- **Input:** current diopter + age + lifestyle + correction type in use
- **Output:** predicted diopter at Year 1, Year 3, Year 5
- **Algorithm:** SVR + Gaussian Process Regression (per bmjophth-2023 paper — Pearson r = 0.77)

### When to start
- [ ] Obtain longitudinal dataset (LVPEI)
- [ ] Dataset needs: baseline diopter, follow-up diopter at 1/3/5 years, age, correction type
- [ ] Re-read `bmjophth-2023-001298.pdf` sections 2–4 for methodology
- [ ] Train SVR + GPR regression model
- [ ] Add "Progression Simulator" tab on website

### Progression Simulator UI (Phase 3 feature)
```
Without treatment:              ────────────────● -6.5 D by age 18
With Atropine 0.01%:            ─────────────●    -4.1 D by age 18
With Myopia Control Glasses:    ────────────●     -3.8 D by age 18
```

---

## Phase 4 — Clinical Validation & Publication

- [ ] Validate model on real patient data from clinical partner
- [ ] Calculate sensitivity, specificity, PPV, NPV on clinical data
- [ ] Improve Stage 1 (Has_RE) with visual acuity input
- [ ] Write research draft — target: *npj Digital Medicine* or *Ophthalmic Epidemiology*
- [ ] Ethics board submission if using patient data
- [ ] Final production deployment with logging + monitoring

---

## Day-by-Day Work Plan

| Day | Task |
|---|---|
| Day 1 | Scaffold React + Vite + Tailwind in `frontend/` · Build landing page · Build form wizard (mock data) |
| Day 2 | Build results page · Add risk gauge chart · Add correction education section |
| Day 3 | Set up FastAPI in `backend/` · Wire `/predict` to saved models · Connect frontend to real API |
| Day 4 | Polish UI · Mobile responsive · Build /about and /faq pages |
| Day 5 | End-to-end test with demo patients · Deploy backend to Render · Deploy frontend to Vercel |
| Day 6+ | Share with doctor · Collect feedback · Iterate |
| TBD | Obtain 5-year data · Start Phase 3 |

---

## Project Folder Structure (Target)

```
Mayopia/
├── README.md                          ← This file — project workflow
├── material.md                        ← Research links and notes
├── Myopia.ipynb                       ← ML notebook (Phase 0, done)
├── Myopia_Dataset_5000.csv
├── Original Research Data.csv
│
├── models/                            ← Saved ML artifacts
│   ├── risk_progression_model.pkl
│   ├── has_re_model.pkl
│   ├── diopter_regression_model.pkl
│   ├── scaler_classification.pkl
│   ├── scaler_regression.pkl
│   ├── feature_columns.json
│   └── model_metadata.json
│
├── backend/                           ← FastAPI (Phase 2)
│   ├── main.py
│   ├── schemas.py
│   ├── pipeline.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── frontend/                          ← React + Vite (Phase 1)
    ├── src/
    │   ├── pages/
    │   │   ├── Home.jsx
    │   │   ├── Screen.jsx
    │   │   ├── Results.jsx
    │   │   ├── About.jsx
    │   │   └── FAQ.jsx
    │   ├── components/
    │   │   ├── FormWizard/
    │   │   ├── RiskGauge/
    │   │   ├── RecommendationCard/
    │   │   └── CorrectionGuide/
    │   └── api/
    │       └── predict.js
    ├── package.json
    └── tailwind.config.js
```

---

## Rules Confirmed by Doctor

| Rule | Decision |
|---|---|
| Rural vs Urban | ❌ Do NOT include — imbalanced data |
| Correction method as model feature | ❌ NOT an input — show as output recommendation only |
| Atropine — show all doses | ✅ Show 0.01% / 0.05% / 1% with explanation |
| Special glasses vs regular | ✅ Explain clearly in results page |
| Publish model on website only if performs well | ✅ Stage 2 (AUC 0.88) is ready — show disclaimer for Stage 1 & 3 |

---

## Reference Materials (in this workspace)

| File | Content |
|---|---|
| `material.md` | Links: PREMo, LVPEI, MPRAS, BHVI calculator |
| `bmjophth-2023-001298.pdf` | ML for refractive error progression — SVR + GPR, Chinese children, Pearson r=0.77 |
| `ijerph-17-00463.pdf` | ML prediction in adolescents — SVM, feature selection, China |
| `pone.0250468.pdf` | Big-data epidemiology of RE — EMR + spectacle lens records, Ireland |
| `A_1008280620621.pdf` | RELIEFF feature selection algorithm |
| `school prediction.pdf` | School-based myopia prediction reference |
| `Myopia research - Jalpesh sir.docx` | Project proposal — aim, methodology, expected outcomes |

## Key External Links

| Resource | URL | Purpose |
|---|---|---|
| PREMo Score | https://myopiaonset.com/ | Pre-myopia detection — reference and compare |
| LVPEI Mission Myopia | https://missionmyopia.lvpei.org/ | India authority — research backing + link on /about |
| MPRAS Score (Nature) | https://www.nature.com/articles/s41598-023-35696-2 | Another risk score — benchmark against ours |
| BHVI Calculator | https://bhvi.org/myopia-calculator-resources/ | Progression calculator — Phase 3 reference |
| Fundus Dataset | https://www.nature.com/articles/s41597-024-02911-2 | Image data — future fundus/deep learning model |

---

## Key Clinical Facts (for website content)

1. **Family history** — both-parent myopia doubles risk (strongest non-modifiable factor)
2. **Screen time > 2 hrs/day** in children significantly elevates risk
3. **Outdoor time ≥ 2 hrs/day** is the strongest natural protector — sunlight triggers dopamine release that slows axial elongation
4. **Academic pressure** (tuition + competitive exams) compounds near-work load
5. **Age ≤ 10 + high risk** → requires monitoring every 3 months
6. **Atropine 0.01%** — gold standard; slows progression 50–60% with minimal side effects
7. **Early onset** (< age 8) carries highest lifetime risk of high myopia (> –6 D) and associated complications

---

## Website Disclaimer (show on results page)

> This tool provides an AI-generated risk assessment based on lifestyle and demographic factors. It is **not a medical diagnosis**. Results should be reviewed by a qualified ophthalmologist or optometrist. Early professional screening is always recommended for children showing symptoms or identified as high risk.
