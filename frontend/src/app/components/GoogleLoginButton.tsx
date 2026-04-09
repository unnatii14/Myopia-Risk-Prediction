import { GoogleLogin, CredentialResponse } from "@react-oauth/google";
import { useNavigate, useLocation } from "react-router";
import { useAuth } from "../context/AuthContext";
import { useState } from "react";
import { API_URL } from "../lib/apiConfig";

interface GoogleLoginButtonProps {
  onError?: (message: string) => void;
}

export default function GoogleLoginButton({ onError }: GoogleLoginButtonProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();
  const [loading, setLoading] = useState(false);
  const authBaseUrl = ((import.meta.env.VITE_AUTH_API_URL as string | undefined) || API_URL)
    .trim()
    .replace(/\/+$/, "");

  const handleSuccess = async (credentialResponse: CredentialResponse) => {
    if (!credentialResponse.credential) {
      onError?.("Failed to get Google credentials");
      return;
    }

    setLoading(true);
    try {
      if (!authBaseUrl) {
        onError?.("Auth server URL is missing. Set VITE_AUTH_API_URL in deployment settings.");
        return;
      }

      const candidates = [
        `${authBaseUrl}/auth/google`,
        `${authBaseUrl}/api/auth/google`,
      ];

      let errorMessage = "Google login failed. Please try again.";
      let loggedIn = false;

      for (const endpoint of candidates) {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token: credentialResponse.credential }),
        });

        let data: any = {};
        try {
          data = await res.json();
        } catch {
          data = {};
        }

        if (res.ok) {
          login(data.name, data.email, data.token, undefined, true);
          const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? "/";
          navigate(from, { replace: true });
          loggedIn = true;
          break;
        }

        if (res.status === 404) {
          continue;
        }

        errorMessage = data.error || errorMessage;
        break;
      }

      if (!loggedIn) {
        onError?.(errorMessage);
      }
    } catch (err) {
      onError?.("Could not reach server. Please try again.");
      console.error("Google login error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleError = () => {
    onError?.("Login Failed");
  };

  return (
    <div className="w-full rounded-2xl border border-[var(--border)] bg-white p-1 shadow-sm">
      <div className="overflow-hidden rounded-xl">
        <GoogleLogin
          onSuccess={handleSuccess}
          onError={handleError}
          useOneTap
          theme="outline"
          size="large"
          text="continue_with"
          shape="rectangular"
          width="330"
        />
      </div>
    </div>
  );
}
