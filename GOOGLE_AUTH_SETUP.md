# Google OAuth Setup Guide

This guide explains how to set up Google OAuth authentication for the Myopia Risk Prediction application.

## Prerequisites

- Google Cloud Project (free)
- Google Client ID for web applications

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click the project dropdown at the top
3. Click "NEW PROJECT"
4. Enter a project name (e.g., "Myopia Guard")
5. Click "CREATE"

## Step 2: Enable Google+ API

1. In the Cloud Console, go to "APIs & Services" > "Library"
2. Search for "Google+ API"
3. Click on it and then click "ENABLE"

## Step 3: Create OAuth 2.0 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client ID"
3. If prompted, select "Configure OAuth Consent Screen":
   - Choose "External" user type
   - Fill in app name: "MyopiaGuard"
   - Add your email as support email and developer contact
   - For scopes, add: `email`, `profile`, `openid`
   - Save and continue
4. Back to Create OAuth 2.0 Client ID:
   - Select Application type: "Web application"
   - Add Authorized JavaScript origins:
     - `http://localhost:5173` (local development)
     - `http://localhost:3000` (if using port 3000)
     - Your production domain (e.g., `https://myopiapredict.com`)
   - Add Authorized redirect URIs:
     - `http://localhost:5173/` (local development)
     - Your production domain URI
   - Click "CREATE"

5. Copy your **Client ID** from the modal that appears

## Step 4: Configure Frontend

### Create `.env` file in frontend directory:

```bash
cd Myopia-Risk-Prediction/frontend
cp .env.example .env
```

### Edit `.env` file:

```
VITE_GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID_HERE
```

Replace `YOUR_GOOGLE_CLIENT_ID_HERE` with your actual Client ID from Step 3.

## Step 5: Configure Backend

### Create `.env` file in backend directory:

```bash
cd ../backend
touch .env
```

### Edit `.env` file:

```
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID_HERE
```

Replace `YOUR_GOOGLE_CLIENT_ID_HERE` with the same Client ID.

### Install dependencies:

```bash
pip install -r requirements.txt
```

The new `google-auth>=2.25.0` dependency has been added to verify Google tokens.

## Step 6: Run the Application

### Terminal 1 - Backend:
```bash
cd Myopia-Risk-Prediction/backend
python api.py
```

### Terminal 2 - Frontend:
```bash
cd Myopia-Risk-Prediction/frontend
npm run dev
```

Or run both together:
```bash
npm start
```

## Step 7: Test Google Login

1. Open http://localhost:5173 in your browser
2. Go to Login or Sign Up page
3. You should see a "Sign in with Google" button
4. Click it and authenticate with your Google account
5. You'll be logged in automatically!

## How It Works

### Frontend Flow:
1. User clicks "Sign in with Google"
2. Google OAuth dialog opens
3. User authenticates with Google
4. Google returns a JWT ID token
5. Frontend sends token to backend: `POST /api/auth/google`

### Backend Flow:
1. Backend receives Google JWT token
2. Verifies token signature using Google's public keys
3. Extracts user info (name, email) from verified token
4. Checks if user exists in SQLite database
5. If new user, creates account automatically
6. Returns app's own JWT token to frontend
7. Frontend stores token in localStorage and logs user in

## Troubleshooting

### "Google Client ID not configured" error
- Check that `GOOGLE_CLIENT_ID` is set in backend `.env`
- Ensure the backend is running

### "Invalid Google token" error
- Verify the Client ID in frontend matches backend
- Check that JavaScript origins are configured correctly in Google Cloud Console
- Clear browser cookies/cache and try again

### CORS errors
- CORS is already enabled on the backend (`flask-cors`)
- No additional configuration needed

### Token verification fails
- Make sure `google-auth` >= 2.25.0 is installed
- Run `pip install -r requirements.txt` in backend directory

## Environment Variables Reference

### Frontend (`Myopia-Risk-Prediction/frontend/.env`)
```
VITE_GOOGLE_CLIENT_ID=your_client_id_here
```

### Backend (`Myopia-Risk-Prediction/backend/.env`)
```
GOOGLE_CLIENT_ID=your_client_id_here
```

## API Endpoints

- **POST** `/api/auth/signup` - Traditional email/password signup
- **POST** `/api/auth/login` - Traditional email/password login
- **POST** `/api/auth/google` - Google OAuth login (NEW)

## Production Deployment

When deploying to production:

1. Add your production domain to Google Cloud Console:
   - Authorized JavaScript origins
   - Authorized redirect URIs

2. Set environment variables on your production server:
   - `GOOGLE_CLIENT_ID`
   - Other existing backend variables

3. Update frontend environment if using a different domain

## Security Notes

- The Google Client ID is safe to expose in frontend code (it's designed to be public)
- Backend uses `google-auth` library to cryptographically verify tokens
- Tokens are verified against Google's public keys (not stored)
- User data is only extracted after successful token verification
- Users who login via Google get auto-created accounts

## Support

For issues with:
- Google Cloud Console: See [Google Cloud Documentation](https://cloud.google.com/docs)
- OAuth flow: See [Google Identity Platform](https://developers.google.com/identity)
- Application: Check logs and error messages in console
