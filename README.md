# Myopia Risk Prediction System

AI-powered pediatric myopia screening platform for children aged 5-18. It predicts risk from lifestyle and demographic inputs without requiring ophthalmology equipment.

## Live Links

- Frontend: https://mayopia-frontend.vercel.app
- Backend: https://mayopia-backend.onrender.com
- Health: https://mayopia-backend.onrender.com/health

## What This Project Includes

- Three-stage ML pipeline (risk + refractive error + diopter severity)
- Multi-step screening UI with strict validation
- Auth flow with signup and login
- Production deployment on Vercel + Render

## Stack

- Backend: Python, Flask, Gunicorn, scikit-learn, XGBoost
- Frontend: React, TypeScript, Vite, Tailwind
- Infra: Docker, Render, Vercel

## Key Paths

- API app: backend/api.py
- Auth: backend/auth.py
- Validation: backend/validation.py
- Frontend app: frontend/src/app
- Render blueprint: render.yaml
- Vercel config: vercel.json

## Local Setup

### Backend

```bash
cd backend
pip install -r requirements.txt
python api.py
```

API URL: http://localhost:5001

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: http://localhost:5173

### Docker (Full Stack)

```bash
docker compose up --build
```

Local URLs:

- Frontend: http://localhost
- Backend: http://localhost:5001

## Deployment Guide (Brief)

### 1) Deploy Backend on Render

1. Create a Blueprint service from this repository.
2. Ensure Render uses render.yaml from repository root.
3. Set environment variable:

```env
CORS_ORIGINS=https://mayopia-frontend.vercel.app
```

4. Deploy latest commit and verify /health.

### 2) Deploy Frontend on Vercel

1. Import the same repository.
2. Set Root Directory to frontend.
3. Set environment variable:

```env
VITE_API_URL=https://mayopia-backend.onrender.com
```

4. Deploy and use the production Vercel URL.

## API Reference

### GET /health

```json
{
  "status": "ok",
  "features": 30
}
```

### GET /

Returns API metadata and available routes.

### POST /predict

Example request:

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

Example response:

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

### Auth Endpoints

- POST /auth/signup
- POST /auth/login

## Validation Rules

- Age range: 5 to 18
- Screen/near/outdoor hours: 0 to 24
- Total daily hours must be <= 24
- Required fields are enforced before prediction

## Troubleshooting

- "Could not reach server": verify VITE_API_URL in Vercel.
- Browser CORS error: verify CORS_ORIGINS in Render.
- Slow first request: expected on free Render cold start.
- 401 on preview URL: use production Vercel domain.

## Security Notes

- Restrict CORS to trusted frontend domains.
- Keep production secrets out of code.
- Add uptime monitoring for /health.

## License

MIT
