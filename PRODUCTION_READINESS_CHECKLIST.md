# Myopia Project Critical Launch Checklist

Use this only for must-have checks before demo/release. If any item fails, do not launch.

## 1) Security and Secrets (Blocker)
- [ ] Rotate JWT_SECRET and Google OAuth credentials if they were ever exposed.
- [ ] Keep secrets only in env files or deployment secrets manager.
- [ ] Ensure no real secrets exist in tracked files.
- [ ] Set FLASK_ENV=production in production.

## 2) Core Environment Configuration (Blocker)
- [ ] backend/.env exists and has valid values for:
  - [ ] SECRET_KEY
  - [ ] JWT_SECRET
  - [ ] GOOGLE_CLIENT_ID
  - [ ] CORS_ORIGINS
- [ ] frontend/.env exists and has valid values for:
  - [ ] VITE_API_URL
  - [ ] VITE_GOOGLE_CLIENT_ID
- [ ] Frontend and backend use the same Google client ID.

## 3) OAuth and CORS (Blocker)
- [ ] Google Cloud OAuth Authorized JavaScript origins include exact deployed frontend URL.
- [ ] CORS_ORIGINS includes only trusted frontend origins.
- [ ] Google login works on deployed domain (not only localhost).

## 4) Backend Runtime Health (Blocker)
- [ ] Backend starts without model-loading errors.
- [ ] GET /health returns 200.
- [ ] POST /predict returns 200 for a valid sample payload.
- [ ] POST /predict returns 400 for invalid payload.
- [ ] API never returns raw internal exception details to client.

## 5) Auth Flows (Blocker)
- [ ] Email signup works.
- [ ] Email login works.
- [ ] Duplicate signup returns proper conflict response.
- [ ] Invalid password login returns unauthorized response.
- [ ] Google login returns app token successfully.

## 6) CI/CD Quality Gates (Blocker)
- [ ] Frontend typecheck passes.
- [ ] Frontend build passes.
- [ ] Backend tests pass.
- [ ] Docker image builds pass.
- [ ] CI has no ignore-on-failure for required checks.

## 7) Data Persistence and Recovery (Blocker)
- [ ] Chosen storage strategy is explicit (SQLite for demo or managed DB for production).
- [ ] If SQLite is used, file persistence is guaranteed in deployment.
- [ ] Backup and restore procedure is documented and tested once.

## 8) User Safety and Clinical Messaging (Blocker)
- [ ] Results page clearly states this is risk screening, not medical diagnosis.
- [ ] PDF/report includes non-diagnostic disclaimer.
- [ ] Guidance tells users to consult an eye specialist for diagnosis/treatment.

## 9) Launch Smoke Test (Final Gate)
- [ ] Open app home page on deployed URL.
- [ ] Complete one full screening flow to results.
- [ ] Generate PDF report successfully.
- [ ] Verify logs show request and response entries without sensitive leakage.

## Definition of Ready to Launch
- [ ] Every blocker above is checked.
- [ ] Release owner sign-off completed.

Owner: ____________________
Date: _____________________
Release tag/version: _____________________
