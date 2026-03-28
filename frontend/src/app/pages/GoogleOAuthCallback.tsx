import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { Loader2 } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { getGoogleRedirectUri } from "../lib/googleRedirectOAuth";

const API_URL = "http://localhost:5001";

export default function GoogleOAuthCallback() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [message, setMessage] = useState("Completing sign-in…");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get("code");
    const state = params.get("state");
    const err = params.get("error");
    const saved = sessionStorage.getItem("google_oauth_state");
    sessionStorage.removeItem("google_oauth_state");

    if (err) {
      setMessage(params.get("error_description") || err || "Google sign-in was cancelled.");
      return;
    }

    if (!code) {
      setMessage("Missing authorization code. Go back to Login and try again.");
      return;
    }

    if (!state || !saved || state !== saved) {
      setMessage("Security check failed (state). Close this tab and try signing in again from Login.");
      return;
    }

    const redirect_uri = getGoogleRedirectUri();

    (async () => {
      try {
        const res = await fetch(`${API_URL}/auth/google/exchange`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code, redirect_uri }),
        });
        const data = await res.json();
        if (!res.ok) {
          setMessage(data.error || "Could not complete sign-in.");
          return;
        }
        login(data.name, data.email, data.token, data.childName || undefined, true);
        navigate("/", { replace: true });
      } catch {
        setMessage("Could not reach the server. Is the API running on port 5001?");
      }
    })();
  }, [login, navigate]);

  return (
    <div className="min-h-[calc(100vh-80px)] flex flex-col items-center justify-center gap-4 px-4">
      <Loader2 className="w-10 h-10 text-[var(--primary-green)] animate-spin" />
      <p className="text-sm text-[var(--text-muted)] text-center max-w-md">{message}</p>
    </div>
  );
}
