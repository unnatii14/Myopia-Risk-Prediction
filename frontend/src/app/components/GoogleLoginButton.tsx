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

  const handleSuccess = async (credentialResponse: CredentialResponse) => {
    if (!credentialResponse.credential) {
      onError?.("Failed to get Google credentials");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/auth/google`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: credentialResponse.credential }),
      });

      const data = await res.json();
      if (!res.ok) {
        onError?.(data.error || "Google login failed. Please try again.");
        return;
      }

      login(data.name, data.email, data.token, undefined, true);
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? "/";
      navigate(from, { replace: true });
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
          width="100%"
        />
      </div>
    </div>
  );
}
