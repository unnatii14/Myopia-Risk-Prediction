/** Full-page OAuth (authorization code). Use when the GIS iframe returns 403 / origin errors. */

const AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth";

export function getGoogleRedirectUri(): string {
  return `${window.location.origin}/auth/google/callback`;
}

export function startGoogleRedirectOAuth(): void {
  const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID?.trim();
  if (!clientId) return;

  const state = crypto.randomUUID();
  sessionStorage.setItem("google_oauth_state", state);

  const params = new URLSearchParams({
    client_id: clientId,
    redirect_uri: getGoogleRedirectUri(),
    response_type: "code",
    scope: "openid email profile",
    state,
    prompt: "select_account",
  });

  window.location.href = `${AUTH_URL}?${params.toString()}`;
}
