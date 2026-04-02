import { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router";
import { motion } from "motion/react";
import { Eye, EyeOff, Mail, Lock, LogIn } from "lucide-react";
import BokehBackground from "../components/BokehBackground";
import GoogleLoginButton from "../components/GoogleLoginButton";
import { useAuth } from "../context/AuthContext";
import { API_URL } from "../lib/apiConfig";

export default function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [form, setForm] = useState({ email: "", password: "" });
  const [errors, setErrors] = useState<{ email?: string; password?: string; general?: string }>({});
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState<"initial" | "verifying" | "finalizing">("initial");

  const validate = () => {
    const errs: typeof errors = {};
    if (!form.email) errs.email = "Email is required.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      errs.email = "Enter a valid email address.";
    if (!form.password) errs.password = "Password is required.";
    else if (form.password.length < 6)
      errs.password = "Password must be at least 6 characters.";
    return errs;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length) {
      setErrors(errs);
      return;
    }
    setErrors({});
    setLoading(true);
    setLoadingStage("initial");

    try {
      // Stage 1: Initial
      await new Promise(resolve => setTimeout(resolve, 800));
      setLoadingStage("verifying");

      // Stage 2: Verifying credentials
      await new Promise(resolve => setTimeout(resolve, 1200));
      setLoadingStage("finalizing");

      // Make API call
      const res = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: form.email, password: form.password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrors({ general: data.error || "Login failed. Please try again." });
        return;
      }
      login(data.name, data.email, data.token, undefined, rememberMe);
      const from = (location.state as { from?: { pathname: string } })?.from?.pathname ?? "/";
      navigate(from, { replace: true });
    } catch {
      setErrors({ general: "Could not reach server. Please try again." });
    } finally {
      setLoading(false);
      setLoadingStage("initial");
    }
  };

  return (
    <div className="relative min-h-[calc(100vh-80px)] flex items-center justify-center overflow-hidden px-4 py-12">
      <BokehBackground />

      {/* Subtle grid overlay */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage:
            "linear-gradient(var(--primary-green) 1px, transparent 1px), linear-gradient(90deg, var(--primary-green) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
        }}
      />

      <motion.div
        className="relative z-10 w-full max-w-md"
        initial={{ opacity: 0, y: 32 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      >
        {/* Card */}
        <div className="bg-white rounded-3xl shadow-2xl border border-[var(--border)] px-8 py-10">
          {/* Header */}
          <div className="flex flex-col items-center mb-8">
            <div className="w-14 h-14 rounded-full bg-[var(--primary-green)] flex items-center justify-center mb-4 shadow-lg">
              <LogIn className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-[var(--text-dark)]" style={{ fontFamily: "var(--font-heading)" }}>
              Welcome back
            </h1>
            <p className="text-[var(--text-muted)] text-sm mt-1 text-center">
              Sign in to your MyopiaGuard account
            </p>
          </div>

          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            {errors.general && (
              <div className="px-4 py-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-600">
                {errors.general}
              </div>
            )}
            {/* Email */}
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-semibold text-[var(--text-dark)] mb-1.5"
              >
                Email address
              </label>
              <div className="relative">
                <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
                <input
                  id="email"
                  type="email"
                  autoComplete="email"
                  value={form.email}
                  onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))}
                  placeholder="you@example.com"
                  className={`w-full pl-10 pr-4 py-3 rounded-xl border text-[var(--text-dark)] text-sm bg-[var(--background-mint)] placeholder-[var(--text-muted)] outline-none transition-all focus:ring-2 focus:ring-[var(--secondary-green)] focus:border-[var(--secondary-green)] ${
                    errors.email ? "border-[var(--warning-coral)]" : "border-[var(--border)]"
                  }`}
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-xs text-[var(--warning-coral)]">{errors.email}</p>
              )}
            </div>

            {/* Password */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label
                  htmlFor="password"
                  className="block text-sm font-semibold text-[var(--text-dark)]"
                >
                  Password
                </label>
                <Link
                  to="/forgot-password"
                  className="text-xs text-[var(--secondary-green)] hover:text-[var(--primary-green)] transition-colors font-medium"
                >
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  value={form.password}
                  onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
                  placeholder="••••••••"
                  className={`w-full pl-10 pr-11 py-3 rounded-xl border text-[var(--text-dark)] text-sm bg-[var(--background-mint)] placeholder-[var(--text-muted)] outline-none transition-all focus:ring-2 focus:ring-[var(--secondary-green)] focus:border-[var(--secondary-green)] ${
                    errors.password ? "border-[var(--warning-coral)]" : "border-[var(--border)]"
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--primary-green)] transition-colors"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              {errors.password && (
                <p className="mt-1 text-xs text-[var(--warning-coral)]">{errors.password}</p>
              )}
            </div>

            {/* Remember Me */}
            <div className="flex items-center gap-2">
              <input
                id="remember-me"
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="w-4 h-4 rounded border-[var(--border)] cursor-pointer accent-[var(--primary-green)]"
              />
              <label
                htmlFor="remember-me"
                className="text-sm text-[var(--text-muted)] cursor-pointer hover:text-[var(--primary-green)] transition-colors"
              >
                Remember me for 24 hours
              </label>
            </div>

            {/* Submit */}
            <motion.button
              type="submit"
              disabled={loading}
              whileTap={{ scale: 0.97 }}
              className="w-full py-3 rounded-full bg-[var(--primary-green)] hover:bg-[var(--secondary-green)] text-white font-semibold text-sm transition-colors mt-2 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                  {loadingStage === "initial" && "Signing in…"}
                  {loadingStage === "verifying" && "Verifying credentials…"}
                  {loadingStage === "finalizing" && "Almost there…"}
                </>
              ) : (
                "Sign In"
              )}
            </motion.button>
          </form>

          {/* Divider */}
          <div className="flex items-center gap-3 my-6">
            <div className="flex-1 h-px bg-[var(--border)]" />
            <span className="text-xs text-[var(--text-muted)]">or</span>
            <div className="flex-1 h-px bg-[var(--border)]" />
          </div>

          {/* Google Login */}
          <GoogleLoginButton onError={(msg) => setErrors({ general: msg })} />

          {/* Sign up link */}
          <p className="text-center text-sm text-[var(--text-muted)]">
            Don't have an account?{" "}
            <Link
              to="/signup"
              className="text-[var(--primary-green)] hover:text-[var(--secondary-green)] font-semibold transition-colors"
            >
              Create one
            </Link>
          </p>
        </div>

        {/* Trust note */}
        <p className="text-center text-xs text-[var(--text-muted)] mt-4">
          Your data is never shared. Protected by MyopiaGuard's privacy policy.
        </p>
      </motion.div>
    </div>
  );
}
