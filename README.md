# MyopiaGuard â€” AI-Powered Childhood Myopia Risk Platform

> A full-stack clinical screening tool that predicts myopia risk in children aged 5â€“18 using a 3-stage machine learning pipeline, retinal image analysis, and evidence-based progression calculators. Built for parents, researchers, and eye-care professionals.

ðŸŒ **Live Demo:** [myopiaguard.vercel.app](https://myopiaguard.vercel.app)

---

## What It Does

MyopiaGuard gives parents and clinicians five evidence-based tools in one platform:

| Tool | Description |
|------|-------------|
| **Myopia Risk Screening** | 12-question AI-powered form returns a personalised risk score in under 3 minutes |
| **Image-Based Detection** | Deep learning classifier analyses blue-channel fundus photographs for myopia indicators |
| **Progression Calculator** | Projects diopter change year-by-year to age 18 based on Donovan et al. (2012) |
| **Axial Elongation Tracker** | Models eye growth in millimetres with treatment vs untreated comparison |
| **Onset Predictor** | Estimates when myopia will begin using Zadnik/CLEERE hyperopic reserve norms |

---

## ML Pipeline

The core screening tool uses a **3-stage ML pipeline** trained on 5,000+ Indian school children records:

```
Input (12 features)
      â”‚
      â–¼
Stage 1 â€” Has Refractive Error?     XGBoost          AUC 0.94
      â”‚
      â–¼
Stage 2 â€” Progression Risk Level    GradientBoosting  AUC 0.89
      â”‚
      â–¼
Stage 3 â€” Diopter Severity Est.     Regression        MAE 1.75D
      â”‚
      â–¼
Output: Risk Score + Level + Diopter Estimate + PDF Report
```

### Model Performance

| Stage | Purpose | Algorithm | Metric | Score |
|-------|---------|-----------|--------|-------|
| Stage 1 | Refractive Error Detection | XGBoost | AUC | **0.9431** |
| Stage 2 | Risk Progression Classification | GradientBoosting | AUC | **0.8893** |
| Stage 2 | Risk Progression Classification | GradientBoosting | Accuracy | **81.2%** |
| Stage 3 | Diopter Severity Estimation | Gradient Boosting Regressor | MAE | **1.75 D** |

### Key Features Analysed (30 total)
Age, BMI, screen time, near-work hours, outdoor time, family history, parental myopia, school type, location (urban/rural), sports frequency, Vitamin D, competitive exam pressure, tuition, state (11 Indian states), and 5 engineered interaction features (AgeÃ—Screen, Screen+Near Total, Outdoor Deficit, Screen/Outdoor Ratio, Family Load).

---

## Image Classification

**Dataset:** [Kelly Anderson â€” Myopia Image Dataset](https://www.kaggle.com/datasets/kellysanderson/myopia-image-dataset)
- 124,794 blue-channel fundus photographs
- 63,294 Myopia / 61,500 Normal (near-balanced)
- Binary classification: Myopia vs Normal

**Model:** Keras CNN â†’ converted to **ONNX** (8.5 MB) for lightweight deployment without TensorFlow dependency.

> âš ï¸ This model is trained on **blue-channel fundus images** (medical equipment output). Standard colour phone photos will not give accurate results.

---

## Tech Stack

### Frontend
- **React 18 + TypeScript** â€” type-safe component architecture
- **Vite** â€” fast build tooling
- **Tailwind CSS** â€” utility-first styling
- **motion/react** â€” smooth page and component animations
- **React Router v7** â€” client-side routing with protected routes
- **shadcn/ui** â€” accessible UI primitives
- **jsPDF** â€” client-side PDF report generation

### Backend
- **Python 3.11 + Flask 3.0** â€” REST API
- **XGBoost 3.2 + scikit-learn 1.7** â€” ML inference
- **ONNX Runtime** â€” lightweight image model inference (no TensorFlow in production)
- **SQLite** â€” user accounts and screening history
- **JWT + bcrypt** â€” authentication and password hashing
- **Google OAuth 2.0** â€” social login
- **gunicorn** â€” production WSGI server

### Infrastructure
- **Vercel** â€” frontend hosting with automatic deploys from GitHub
- **Render** â€” backend Docker container hosting
- **Docker** â€” containerised backend (Python 3.11-slim)
- **GitHub Actions** â€” CI/CD pipeline

---

## Key Product Features

### User System
- Email/password signup and login
- Google OAuth one-click login
- JWT-based session management (24-hour expiry)
- Remember Me option (localStorage vs sessionStorage)

### Dashboard
- Live last screening result with risk score and colour-coded badge
- Days since last check
- Trend arrow comparing current vs previous screening
- Tappable history strip â€” tap any past result to see full detail modal
- Empty state with direct CTA for first-time users

### Screening History
- Every completed screening is automatically saved to the user's account
- "Saved to Dashboard" toast confirmation after each save
- History accessible from Dashboard without re-running the screening
- Child name, age, lifestyle inputs, and ML result all stored per record

### Results & Reports
- Animated risk gauge (LOW / MODERATE / HIGH)
- 3-stage pipeline breakdown shown separately
- Accordion with personalised recommendations
- One-click PDF report download with child profile, risk score, and recommendations
- Research references and methodology footnotes in the PDF

### Smart UX
- Screen form pre-fills child name, age, sex, height, weight from last screening
- Logged-in users redirected to Dashboard (not Landing page)
- Logo click goes to Dashboard for logged-in users
- Drag & drop image upload with file preview

---

## Project Structure

```
MyopiaGuard/
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ api.py                        # Main Flask app â€” all ML endpoints
â”‚   â”œâ”€â”€ auth.py                       # Auth blueprint â€” signup, login, Google OAuth
â”‚   â”œâ”€â”€ history.py                    # History blueprint â€” save & retrieve screenings
â”‚   â”œâ”€â”€ config.py                     # Environment-based configuration
â”‚   â”œâ”€â”€ validation.py                 # Input validation with detailed error messages
â”‚   â”œâ”€â”€ logger.py                     # Structured request/response logging
â”‚   â”œâ”€â”€ requirements-docker.txt       # Pinned production dependencies
â”‚   â””â”€â”€ Dockerfile                    # Python 3.11-slim Docker image
â”‚
â”œâ”€â”€ frontend/
â”‚   â””â”€â”€ src/app/
â”‚       â”œâ”€â”€ pages/
â”‚       â”‚   â”œâ”€â”€ Landing.tsx           # Marketing homepage
â”‚       â”‚   â”œâ”€â”€ Dashboard.tsx         # User dashboard with live history
â”‚       â”‚   â”œâ”€â”€ Screen.tsx            # 4-step screening wizard
â”‚       â”‚   â”œâ”€â”€ Results.tsx           # Risk results + PDF download
â”‚       â”‚   â”œâ”€â”€ ImagePredictor.tsx    # Retinal image upload + ONNX inference
â”‚       â”‚   â”œâ”€â”€ Progression.tsx       # Diopter progression calculator
â”‚       â”‚   â”œâ”€â”€ AxialElongation.tsx   # Axial length growth tracker
â”‚       â”‚   â”œâ”€â”€ OnsetPredictor.tsx    # Myopia onset age predictor
â”‚       â”‚   â”œâ”€â”€ About.tsx             # Research methodology + references
â”‚       â”‚   â””â”€â”€ FAQ.tsx               # Frequently asked questions
â”‚       â”œâ”€â”€ components/
â”‚       â”‚   â”œâ”€â”€ Navbar.tsx            # Responsive navbar with tool dropdown
â”‚       â”‚   â”œâ”€â”€ HomeRedirect.tsx      # Smart redirect â€” guestâ†’Landing, userâ†’Dashboard
â”‚       â”‚   â”œâ”€â”€ PrivateRoute.tsx      # Auth-protected route wrapper
â”‚       â”‚   â””â”€â”€ GoogleLoginButton.tsx # Google OAuth button
â”‚       â”œâ”€â”€ context/
â”‚       â”‚   â””â”€â”€ AuthContext.tsx       # Global auth state (localStorage + sessionStorage)
â”‚       â””â”€â”€ lib/
â”‚           â”œâ”€â”€ apiConfig.ts          # API URL resolution (dev vs production)
â”‚           â”œâ”€â”€ historyApi.ts         # Screening history API calls
â”‚           â””â”€â”€ imageApi.ts           # Image prediction API calls
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ has_re_model_improved.pkl     # Stage 1 â€” XGBoost refractive error detector
â”‚   â”œâ”€â”€ risk_progression_model.pkl    # Stage 2 â€” GradientBoosting risk classifier
â”‚   â”œâ”€â”€ diopter_regression_model.pkl  # Stage 3 â€” Diopter severity estimator
â”‚   â”œâ”€â”€ myopia_classifier.onnx        # Image classifier (8.5MB, no TF needed)
â”‚   â”œâ”€â”€ scaler_classification.pkl     # Feature scaler for Stage 2
â”‚   â”œâ”€â”€ scaler_regression.pkl         # Feature scaler for Stage 3
â”‚   â”œâ”€â”€ has_re_scaler.pkl             # Feature scaler for Stage 1
â”‚   â”œâ”€â”€ has_re_features.json          # Stage 1 feature column order
â”‚   â””â”€â”€ feature_columns.json          # Stage 2/3 feature column order
â”‚
â”œâ”€â”€ vercel.json                        # Vercel build config + COOP headers + rewrites
â”œâ”€â”€ docker-compose.yml                 # Local full-stack orchestration
â””â”€â”€ .github/workflows/ci.yml          # GitHub Actions CI pipeline
```

---

## API Reference

### `GET /health`
```json
{
  "status": "ok",
  "features": 30,
  "image_model_loaded": true,
  "image_model_error": null
}
```

### `POST /predict`
**Body:**
```json
{
  "age": 12, "sex": "male", "height": 150, "weight": 40,
  "screenTime": 5, "nearWork": 3, "outdoorTime": 1,
  "sports": "occasional", "familyHistory": true,
  "parentsMyopic": "both", "vitaminD": false,
  "locationType": "urban", "schoolType": "private",
  "state": "Maharashtra"
}
```
**Response:**
```json
{
  "risk_score": 78,
  "risk_level": "HIGH",
  "risk_probability": 0.782,
  "has_re": true,
  "re_probability": 0.841,
  "diopters": 3.2,
  "severity": "Moderate"
}
```

### `POST /predict-image`
Multipart form-data with field `image` (PNG/JPG).
```json
{
  "label": "MYOPIA",
  "myopia_probability": 0.873,
  "normal_probability": 0.127,
  "threshold": 0.5,
  "model_input_size": [224, 224],
  "duration_ms": 142.5
}
```

### `POST /auth/signup` Â· `POST /auth/login` Â· `POST /auth/google`
Standard JWT auth. Returns `{ token, name, email }`.

### `POST /history/save` Â· `GET /history` Â· `GET /history/latest`
JWT-protected. Saves and retrieves screening records per user.

---

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # fill in your values
python api.py
# API running at http://localhost:5001
```

### Frontend
```bash
cd frontend
npm install
cp .env.example .env   # add VITE_API_URL and VITE_GOOGLE_CLIENT_ID
npm run dev
# App running at http://localhost:5173
```

### Environment Variables

**frontend/.env**
```env
VITE_API_URL=http://localhost:5001
VITE_GOOGLE_CLIENT_ID=your_google_client_id
```

**backend/.env**
```env
FLASK_ENV=development
JWT_SECRET=your_jwt_secret
SECRET_KEY=your_secret_key
GOOGLE_CLIENT_ID=your_google_client_id
CORS_ORIGINS=http://localhost:5173
```

---

## Deployment

### Frontend â†’ Vercel
1. Connect GitHub repo to Vercel
2. Set environment variables: `VITE_API_URL`, `VITE_GOOGLE_CLIENT_ID`
3. Vercel auto-deploys on every push to `main`

### Backend â†’ Render (Docker)
1. Create a new Web Service on Render â†’ select Docker runtime
2. Set environment variables:

```env
FLASK_ENV=production
JWT_SECRET=<strong-random-string>
SECRET_KEY=<strong-random-string>
GOOGLE_CLIENT_ID=<your-google-client-id>
CORS_ORIGINS=https://myopiaguard.vercel.app
```

3. Deploy from the `main` branch

---

## Research References

- **Donovan et al. (2012)** â€” Age-specific myopia progression rates. *Optometry and Vision Science.*
- **Zadnik / CLEERE Study** â€” Hyperopic reserve norms for onset prediction. *Invest. Ophthalmol. Vis. Sci.*
- **BHVI / IMI Guidelines** â€” Myopia management treatment effect benchmarks.
- **MPRAS (Nature, 2023)** â€” Myopia Prediction Risk Assessment Score. *Scientific Reports.*
- **LVPEI Mission Myopia** â€” Clinical context for Indian school children screening.

---

## License

MIT License â€” free to use, modify, and distribute with attribution.

---

## Contributors

| | Name | GitHub |
|--|------|--------|
| ðŸ‘©â€ðŸ’» | **Nency Pansuria** | [@Nency02](https://github.com/Nency02) |
| ðŸ‘©â€ðŸ’» | **Unnati Tank** | [@unnatii14](https://github.com/unnatii14) |

---

## Contact

ðŸ“§ [GitHub Issues](https://github.com/unnatii14/Myopia-Risk-Prediction/issues) for questions or collaboration.
