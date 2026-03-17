# Myopia Risk Prediction System

AI-powered pediatric myopia screening system for proactive risk assessment in school children aged 5-18. Predicts myopia risk based on lifestyle and demographic factors without requiring ophthalmology equipment.

## Features

- **3-Stage ML Pipeline**: Refractive error detection (AUC 0.94), risk progression classification (AUC 0.88), diopter severity estimation
- **Input Validation**: Age constraints (5-18 years), time limits, required field enforcement
- **Structured Logging**: Request/response tracking with file rotation
- **Production Ready**: Docker deployment, CI/CD pipeline, comprehensive testing
- **Interactive Frontend**: React-based screening form with risk visualization

## Technology Stack

**Backend**
- Python 3.13, Flask 3.0
- XGBoost 3.2, scikit-learn 1.7
- Structured logging with rotation

**Frontend**
- React 18, TypeScript, Vite
- TailwindCSS, shadcn/ui components
- React Hook Form, Recharts

**Deployment**
- Docker multi-stage builds
- Nginx reverse proxy
- GitHub Actions CI/CD

## Quick Start

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python api.py
```

API runs on `http://localhost:5001`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`

### Docker Deployment

```bash
docker-compose up --build
```

Access at `http://localhost` (frontend) and `http://localhost:5001` (API)

## Deploy (Recommended: Vercel + Render)

This repo is configured for the fastest free setup:

- Frontend: Vercel (`frontend/vercel.json`)
- Backend API: Render (`render.yaml`)

### 1) Deploy Backend on Render (10-15 min)

1. Push this repository to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Connect the GitHub repo and select the branch.
4. Render will detect `render.yaml` and create `mayopia-backend`.
5. In service environment variables, set:

```env
CORS_ORIGINS=https://your-frontend-domain.vercel.app
```

6. Deploy and confirm health endpoint works:

```text
https://your-render-service.onrender.com/health
```

### 2) Deploy Frontend on Vercel (5-10 min)

1. In Vercel, choose **Add New...** -> **Project**.
2. Import the same GitHub repository.
3. Set **Root Directory** to `frontend`.
4. Add environment variable:

```env
VITE_API_URL=https://your-render-service.onrender.com
```

5. Deploy.
6. Open the deployed site and test signup/login + prediction flow.

### 3) First-Run Checks (5-10 min)

1. Confirm browser network calls hit Render URL (not localhost).
2. Confirm CORS has no errors.
3. Check `/health` and one `/predict` request.
4. Note: Render free tier can sleep; first call after idle may be slow.

## API Endpoints

### `GET /health`
Health check endpoint

**Response:** `200 OK`
```json
{
  "status": "ok",
  "features": 30
}
```

### `POST /predict`
Predict myopia risk for a child

**Request Body:**
```json
{
  "age": 12,
  "sex": "male",
  "height": 150,
  "weight": 40,
  "screenTime": 5,
  "nearWork": 3,
  "outdoorTime": 1.5,
  "sports": "occasional",
  "familyHistory": true,
  "parentsMyopic": "both",
  "vitaminD": false
}
```

**Response:** `200 OK`
```json
{
  "risk_score": 87,
  "risk_level": "HIGH",
  "risk_probability": 0.874,
  "has_re": true,
  "re_probability": 0.721,
  "diopters": 3.2,
  "severity": "Moderate"
}
```

**Validation Rules:**
- Age: 5-18 years
- Screen time, near work, outdoor time: 0-24 hours
- Total daily time ≤ 24 hours
- Required fields: age, sex, screenTime, nearWork, outdoorTime, sports

## Model Performance

| Stage | Purpose | Model | Metric | Value |
|-------|---------|-------|--------|-------|
| Stage 1 | Refractive Error Detection | XGBoost | AUC | **0.9431** |
| Stage 2 | Risk Progression | XGBoost | AUC | **0.8842** |
| Stage 2 | Risk Progression | XGBoost | Accuracy | **81.2%** |
| Stage 3 | Diopter Estimation | Gradient Boosting | MAE | 1.75 D |

## Project Structure

```
Mayopia/
├── backend/
│   ├── api.py                 # Flask application
│   ├── validation.py          # Input validation
│   ├── logger.py              # Structured logging
│   ├── config.py              # Configuration management
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Backend container
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── pages/         # Landing, Screen, Results, FAQ
│   │   │   └── components/    # UI components
│   │   └── main.tsx           # Entry point
│   ├── package.json           # Node dependencies
│   ├── Dockerfile             # Frontend container
│   └── nginx.conf             # Nginx configuration
├── models/
│   ├── has_re_model_improved.pkl        # Stage 1 model
│   ├── risk_progression_model.pkl       # Stage 2 model
│   ├── diopter_regression_model.pkl     # Stage 3 model
│   └── *.pkl, *.json          # Scalers and metadata
├── docker-compose.yml         # Full stack orchestration
└── .github/workflows/ci.yml   # CI/CD pipeline
```

## Environment Variables

Create `backend/.env`:

```env
FLASK_ENV=development
FLASK_DEBUG=True
LOG_LEVEL=INFO
```

Production (Render) uses dashboard environment variables. Recommended:

```env
FLASK_ENV=production
LOG_LEVEL=INFO
CORS_ORIGINS=https://your-frontend-domain.vercel.app
```

## License

MIT License

## Contact

For questions or collaboration, please open an issue on GitHub.
