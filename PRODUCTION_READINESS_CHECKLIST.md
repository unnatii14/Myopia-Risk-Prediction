# Production Readiness Checklist

Use this checklist before public demo, external sharing, or production release.

## Must Have (Blocker)

### 1) Secrets and Credentials
- [ ] Rotate JWT secret and Google OAuth client secret if they were ever exposed.
- [ ] Keep secrets only in environment variables (never in tracked files).
- [ ] Verify backend and frontend use the same Google client ID.

### 2) Environment Configuration
- [ ] Create real env files from examples:
  - [ ] backend/.env
  - [ ] frontend/.env
- [ ] Confirm frontend API points to backend:
  - [ ] VITE_API_URL is set correctly for deployment.
- [ ] Confirm backend CORS config matches deployed frontend origin(s).

### 3) OAuth and CORS Safety
- [ ] In Google Cloud Console, set exact Authorized JavaScript Origins for deployed frontend URL(s).
- [ ] Remove unnecessary localhost origins in production settings.
- [ ] Confirm OAuth login works end-to-end on deployed domain.

### 4) Runtime Health and Core Paths
- [ ] Backend starts successfully with no model-loading errors.
- [ ] GET /health returns 200.
- [ ] POST /predict works with a known-good sample payload.
- [ ] Login/signup and Google auth flows succeed.

### 5) CI and Build Gates
- [ ] Frontend build passes.
- [ ] Backend validation job passes.
- [ ] Docker image builds pass.
- [ ] No CI rule silently ignores failures for required checks.

### 6) Storage and Backups
- [ ] Decide persistent storage approach for user/auth data.
- [ ] If SQLite is used, ensure persistent volume and backup policy are defined.
- [ ] Define restore procedure and test at least once.

### 7) Transport and Error Safety
- [ ] Enable HTTPS for deployed frontend and backend.
- [ ] Do not expose raw backend exception traces to end users.
- [ ] Confirm security headers at reverse proxy/web server.

### 8) Clinical/Legal Messaging
- [ ] Show non-diagnostic disclaimer on results/report pages.
- [ ] Ensure recommendation language is supportive, not definitive diagnosis.
- [ ] Include contact/escalation guidance for clinical follow-up.

## Nice to Have (Post-Launch)

### 1) Stronger Testing
- [ ] Add backend integration tests for auth and predict endpoints.
- [ ] Add frontend lint and typecheck scripts and enforce in CI.
- [ ] Add API contract tests for request/response shape.

### 2) Security Hardening
- [ ] Add rate limiting and basic brute-force protection.
- [ ] Add dependency vulnerability scanning in CI.
- [ ] Add session/token revocation strategy.

### 3) Observability
- [ ] Add structured metrics for auth success/failure and predict latency.
- [ ] Configure alerting for health failures and high error rates.
- [ ] Define log retention and privacy-safe logging policy.

### 4) Model Operations
- [ ] Verify model version metadata at app startup.
- [ ] Add input drift monitoring and periodic model quality checks.
- [ ] Document retraining and rollback process.

## 45-Minute Go-Live Order

1. [ ] Rotate secrets and update env values.
2. [ ] Confirm OAuth origins + CORS for deployed domain.
3. [ ] Run local verification (health, predict, auth).
4. [ ] Push and ensure CI is fully green.
5. [ ] Deploy and run smoke checks on live URL.

## Sign-off

- Owner: ____________________
- Date: _____________________
- Release tag/version: _____________________
- Notes: __________________________________
