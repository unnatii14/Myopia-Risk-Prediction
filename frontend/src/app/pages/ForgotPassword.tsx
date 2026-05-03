import { useState } from "react";
import { useNavigate } from "react-router";
import { Mail, Key } from "lucide-react";
import BokehBackground from "../components/BokehBackground";
import { API_URL } from "../lib/apiConfig";

export default function ForgotPassword() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [token, setToken] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [stage, setStage] = useState<"request" | "confirm">("request");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const authBaseUrl = ((import.meta.env.VITE_AUTH_API_URL as string | undefined) || API_URL)
    .trim()
    .replace(/\/+$/, "");

  const requestReset = async () => {
    setMessage("");
    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setMessage("Enter a valid email address.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${authBaseUrl}/auth/request-password-reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (res.ok) {
        setStage("confirm");
        setMessage("We sent a reset link/code to your email. It expires in 1 hour.");
      } else {
        const data = await res.json().catch(() => ({}));
        setMessage(data.error || "Could not send reset email.");
      }
    } catch {
      setMessage("Unable to reach server. Try again later.");
    } finally {
      setLoading(false);
    }
  };

  const submitReset = async () => {
    setMessage("");
    if (!token) { setMessage("Enter the code or link token sent to your email."); return; }
    if (!newPassword || newPassword.length < 8) { setMessage("Password must be at least 8 characters."); return; }
    if (newPassword !== confirmPassword) { setMessage("Passwords do not match."); return; }

    setLoading(true);
    try {
      const res = await fetch(`${authBaseUrl}/auth/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, token, newPassword }),
      });
      if (res.ok) {
        navigate("/login", { replace: true });
      } else {
        const data = await res.json().catch(() => ({}));
        setMessage(data.error || "Reset failed. Check the token and try again.");
      }
    } catch {
      setMessage("Unable to reach server. Try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <BokehBackground />
      <div className="relative z-10 w-full max-w-md bg-white rounded-3xl shadow-2xl border border-[var(--border)] px-8 py-10">
        <h2 className="text-xl font-bold mb-2">Reset your password</h2>
        <p className="text-sm text-[var(--text-muted)] mb-4">Enter your account email to receive a reset link or code.</p>

        {message && <div className="mb-3 text-sm text-[var(--text-muted)]">{message}</div>}

        {stage === 'request' ? (
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Email</label>
              <div className="relative mt-1">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
                <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" className="w-full pl-10 pr-3 py-2 rounded-xl border bg-[var(--background-mint)]" />
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={requestReset} className="flex-1 py-2 rounded-full bg-[var(--primary-green)] text-white">Send reset email</button>
              <button onClick={() => navigate('/login')} className="py-2 px-3 rounded-xl border">Cancel</button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div>
              <label className="text-sm font-medium">Token / code</label>
              <input value={token} onChange={(e) => setToken(e.target.value)} placeholder="paste token from email" className="w-full mt-1 pl-3 pr-3 py-2 rounded-xl border bg-[var(--background-mint)]" />
            </div>
            <div>
              <label className="text-sm font-medium">New password</label>
              <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="New password" className="w-full mt-1 pl-3 pr-3 py-2 rounded-xl border bg-[var(--background-mint)]" />
            </div>
            <div>
              <label className="text-sm font-medium">Confirm password</label>
              <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} placeholder="Confirm password" className="w-full mt-1 pl-3 pr-3 py-2 rounded-xl border bg-[var(--background-mint)]" />
            </div>
            <div className="flex gap-2">
              <button onClick={submitReset} className="flex-1 py-2 rounded-full bg-[var(--primary-green)] text-white">Set new password</button>
              <button onClick={() => { setStage('request'); setToken(''); }} className="py-2 px-3 rounded-xl border">Back</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
