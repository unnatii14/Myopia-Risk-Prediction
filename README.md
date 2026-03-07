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

## API Endpoints

### `GET /health`
Health check endpoint

**Response:** `200 OK`
```json
{
  "status": "healthy",
  "timestamp": "2026-03-07T10:30:00Z"
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
  "status": "success",
  "risk_level": "HIGH",
  "risk_percentage": 87.5,
  "has_refractive_error": true,
  "has_re_confidence": 0.72,
  "estimated_diopters": -3.2,
  "recommendations": [
    "Limit screen time to <2 hours/day",
    "Increase outdoor time to ≥2 hours/day",
    "Schedule eye examination with ophthalmologist"
  ]
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

## License

MIT License

## Contact

For questions or collaboration, please open an issue on GitHub.
