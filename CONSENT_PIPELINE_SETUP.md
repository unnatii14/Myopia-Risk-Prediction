# Image Consent & Contribution Pipeline — Setup

This document explains the image-donation pipeline that was added, how it works,
and the one manual step you must do (provision a database).

## What was built

A logged-in user who runs an image prediction can now **optionally donate** that
retinal image to help improve the model. The design deliberately avoids the naive
"auto-train on every upload" approach, because that would (a) train the model on
its own guesses — a feedback loop that degrades accuracy, and (b) train on junk or
mislabelled images.

Instead, every donation:

1. Requires **explicit, opt-in consent** (checkbox is unchecked by default).
2. Is **validated** as a real fundus image before it is stored.
3. Is saved with status `pending` — **nothing is auto-added to the training set**.
   A human must review and label it before it can ever be used.
4. Can be **listed and withdrawn** by the user at any time (a data-privacy right).

### New / changed files

| File | Purpose |
|------|---------|
| `backend/db.py` | Tiny DB layer: PostgreSQL in production (via `DATABASE_URL`), SQLite locally. |
| `backend/contributions.py` | Blueprint: `POST /contribute-image`, `GET /contribute/mine`, `DELETE /contribute/<id>`. |
| `backend/image_validation.py` | Shared fundus-image validator (used by predict **and** contribute). |
| `backend/api.py` | Registers the new blueprint; imports the shared validator. |
| `backend/requirements.txt` | Adds `psycopg2-binary` (Postgres driver). |
| `frontend/.../lib/imageApi.ts` | `contributeImage()` API helper. |
| `frontend/.../pages/ImagePredictor.tsx` | Consent card shown after a prediction. |

### The `contributions` table

```
id, email, image_b64, image_mime,
model_prediction, model_confidence,   -- what the model guessed
reported_label,                       -- what the user says (myopia/normal/unknown)
consent_version,                      -- which consent wording they agreed to
review_status,                        -- 'pending' until a human approves
reviewer_label,                       -- the human-confirmed label (filled at review)
created_at
```

## The one manual step: provision the database

The code runs on **SQLite automatically for local development** — no setup needed.
For **production on Render**, add a free Postgres database so donations survive
redeploys (Render's disk is ephemeral, so file storage would be wiped each deploy).

1. Render dashboard → **New → PostgreSQL** → Free plan → create.
2. Open the new database → copy its **Internal Database URL**.
3. Go to your `mayopia-backend` **web service → Environment** → add:
   - Key: `DATABASE_URL`
   - Value: *(the Internal Database URL you copied)*
4. Save. Render redeploys. On boot the `contributions` table is created automatically.

That's it — when `DATABASE_URL` is present the app uses Postgres; when it's absent
it uses local SQLite. No code change needed to switch.

> Tip: the same `DATABASE_URL` approach can later replace the ephemeral `users.db`
> so real signups also survive redeploys. That migration is a good follow-up.

## What is NOT built yet (planned next steps)

- **Admin review queue UI** — a protected page to view pending donations, confirm
  the label, and approve or reject. (The data is already captured with a
  `review_status` field ready for this.)
- **Export for retraining** — a script to export approved+labelled images so your
  teammate can fold them into the training set.
- **Model promotion gate** — only ship a retrained model if it beats the current
  one on a held-out test set.

## Consent wording (current)

> "I consent to donate this retinal image to help improve the model, and I confirm
> I have the right to share it."

Consent version string: `2026-07-04.v1` (bump this in `contributions.py` whenever
the wording changes).
