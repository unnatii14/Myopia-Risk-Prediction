import { useState } from "react";
import { Link, useNavigate } from "react-router";
import { motion } from "motion/react";
import { Eye, EyeOff, Mail, Lock, User, UserPlus } from "lucide-react";
import BokehBackground from "../components/BokehBackground";
import GoogleLoginButton from "../components/GoogleLoginButton";
import { useAuth } from "../context/AuthContext";
import { API_URL } from "../lib/apiConfig";

export default function Signup() {
  const navigate = useNavigate();
  const { login } = useAuth();
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [form, setForm] = useState({
    name: "",
    childName: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<{
    name?: string;
    childName?: string;
    email?: string;
    password?: string;
    confirmPassword?: string;
    general?: string;
  }>({});
  const [loading, setLoading] = useState(false);

  const validate = () => {
    const errs: typeof errors = {};
    if (!form.name.trim()) errs.name = "Full name is required.";
    if (!form.childName.trim()) errs.childName = "Child name is required.";
    if (!form.email) errs.email = "Email is required.";
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(form.email))
      errs.email = "Enter a valid email address.";
    if (!form.password) errs.password = "Password is required.";
    else if (form.password.length < 8)
      errs.password = "Password must be at least 8 characters.";
    if (!form.confirmPassword)
      errs.confirmPassword = "Please confirm your password.";
    else if (form.password !== form.confirmPassword)
      errs.confirmPassword = "Passwords do not match.";
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
    try {
      const res = await fetch(`${API_URL}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: form.name, childName: form.childName, email: form.email, password: form.password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrors({ general: data.error || "Signup failed. Please try again." });
        return;
      }
      login(data.name, data.email, data.token, form.childName);
      navigate("/");
    } catch {
      setErrors({ general: "Could not reach server. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const strength = (pw: string) => {
    if (!pw) return 0;
    let score = 0;
    if (pw.length >= 8) score++;
    if (/[A-Z]/.test(pw)) score++;
    if (/[0-9]/.test(pw)) score++;
    if (/[^A-Za-z0-9]/.test(pw)) score++;
    return score;
  };

  const pwStrength = strength(form.password);
  const strengthLabel = ["", "Weak", "Fair", "Good", "Strong"][pwStrength];
  const strengthColor = [
    "",
    "var(--warning-coral)",
    "var(--moderate-risk)",
    "var(--accent-blue)",
    "var(--low-risk)",
  ][pwStrength];

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
              <UserPlus className="w-7 h-7 text-white" />
            </div>
            <h1
              className="text-2xl font-bold text-[var(--text-dark)]"
              style={{ fontFamily: "var(--font-heading)" }}
            >
              Create your account
            </h1>
            <p className="text-[var(--text-muted)] text-sm mt-1 text-center">
              Start protecting your child's vision today
            </p>
          </div>

          <form onSubmit={handleSubmit} noValidate className="space-y-5">
            {errors.general && (
              <div className="px-4 py-3 bg-red-50 border border-red-200 rounded-xl text-sm text-red-600">
                {errors.general}
              </div>
            )}
            {/* Full Name */}
            <div>
              <label
                htmlFor="name"
                className="block text-sm font-semibold text-[var(--text-dark)] mb-1.5"
              >
                Full name
              </label>
              <div className="relative">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
                <input
                  id="name"
                  type="text"
                  autoComplete="name"
                  value={form.name}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, name: e.target.value }))
                  }
                  placeholder="Jane Smith"
                  className={`w-full pl-10 pr-4 py-3 rounded-xl border text-[var(--text-dark)] text-sm bg-[var(--background-mint)] placeholder-[var(--text-muted)] outline-none transition-all focus:ring-2 focus:ring-[var(--secondary-green)] focus:border-[var(--secondary-green)] ${
                    errors.name
                      ? "border-[var(--warning-coral)]"
                      : "border-[var(--border)]"
                  }`}
                />
              </div>
              {errors.name && (
                <p className="mt-1 text-xs text-[var(--warning-coral)]">
                  {errors.name}
                </p>
              )}
            </div>

            {/* Child Name */}
            <div>
              <label
                htmlFor="childName"
                className="block text-sm font-semibold text-[var(--text-dark)] mb-1.5"
              >
                Child's name
              </label>
              <div className="relative">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
                <input
                  id="childName"
                  type="text"
                  autoComplete="off"
                  value={form.childName}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, childName: e.target.value }))
                  }
                  placeholder="John Doe"
                  className={`w-full pl-10 pr-4 py-3 rounded-xl border text-[var(--text-dark)] text-sm bg-[var(--background-mint)] placeholder-[var(--text-muted)] outline-none transition-all focus:ring-2 focus:ring-[var(--secondary-green)] focus:border-[var(--secondary-green)] ${
                    errors.childName
                      ? "border-[var(--warning-coral)]"
                      : "border-[var(--border)]"
                  }`}
                />
              </div>
              {errors.childName && (
                <p className="mt-1 text-xs text-[var(--warning-coral)]">
                  {errors.childName}
                </p>
              )}
            </div>

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
                  onChange={(e) =>
                    setForm((f) => ({ ...f, email: e.target.value }))
                  }
                  placeholder="you@example.com"
                  className={`w-full pl-10 pr-4 py-3 rounded-xl border text-[var(--text-dark)] text-sm bg-[var(--background-mint)] placeholder-[var(--text-muted)] outline-none transition-all focus:ring-2 focus:ring-[var(--secondary-green)] focus:border-[var(--secondary-green)] ${
                    errors.email
                      ? "border-[var(--warning-coral)]"
                      : "border-[var(--border)]"
                  }`}
                />
              </div>
              {errors.email && (
                <p className="mt-1 text-xs text-[var(--warning-coral)]">
                  {errors.email}
                </p>
              )}
            </div>

            {/* Password */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-semibold text-[var(--text-dark)] mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="new-password"
                  value={form.password}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, password: e.target.value }))
                  }
                  placeholder="Min. 8 characters"
                  className={`w-full pl-10 pr-11 py-3 rounded-xl border text-[var(--text-dark)] text-sm bg-[var(--background-mint)] placeholder-[var(--text-muted)] outline-none transition-all focus:ring-2 focus:ring-[var(--secondary-green)] focus:border-[var(--secondary-green)] ${
                    errors.password
                      ? "border-[var(--warning-coral)]"
                      : "border-[var(--border)]"
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--primary-green)] transition-colors"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  {showPassword ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>

              {/* Strength bar */}
              {form.password && (
                <div className="mt-2">
                  <div className="flex gap-1 mb-1">
                    {[1, 2, 3, 4].map((i) => (
                      <div
                        key={i}
                        className="h-1 flex-1 rounded-full transition-all duration-300"
                        style={{
                          backgroundColor:
                            i <= pwStrength ? strengthColor : "var(--border)",
                        }}
                      />
                    ))}
                  </div>
                  <p className="text-xs" style={{ color: strengthColor }}>
                    {strengthLabel}
                  </p>
                </div>
              )}

              {errors.password && (
                <p className="mt-1 text-xs text-[var(--warning-coral)]">
                  {errors.password}
                </p>
              )}
            </div>

            {/* Confirm Password */}
            <div>
              <label
                htmlFor="confirmPassword"
                className="block text-sm font-semibold text-[var(--text-dark)] mb-1.5"
              >
                Confirm password
              </label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
                <input
                  id="confirmPassword"
                  type={showConfirm ? "text" : "password"}
                  autoComplete="new-password"
                  value={form.confirmPassword}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      confirmPassword: e.target.value,
                    }))
                  }
                  placeholder="Re-enter password"
                  className={`w-full pl-10 pr-11 py-3 rounded-xl border text-[var(--text-dark)] text-sm bg-[var(--background-mint)] placeholder-[var(--text-muted)] outline-none transition-all focus:ring-2 focus:ring-[var(--secondary-green)] focus:border-[var(--secondary-green)] ${
                    errors.confirmPassword
                      ? "border-[var(--warning-coral)]"
                      : "border-[var(--border)]"
                  }`}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--primary-green)] transition-colors"
                  aria-label={
                    showConfirm ? "Hide password" : "Show password"
                  }
                >
                  {showConfirm ? (
                    <EyeOff className="w-4 h-4" />
                  ) : (
                    <Eye className="w-4 h-4" />
                  )}
                </button>
              </div>
              {errors.confirmPassword && (
                <p className="mt-1 text-xs text-[var(--warning-coral)]">
                  {errors.confirmPassword}
                </p>
              )}
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
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8z"
                    />
                  </svg>
                  Creating account…
                </>
              ) : (
                "Create Account"
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

          {/* Log in link */}
          <p className="text-center text-sm text-[var(--text-muted)]">
            Already have an account?{" "}
            <Link
              to="/login"
              className="text-[var(--primary-green)] hover:text-[var(--secondary-green)] font-semibold transition-colors"
            >
              Sign in
            </Link>
          </p>
        </div>

        {/* Trust note */}
        <p className="text-center text-xs text-[var(--text-muted)] mt-4">
          By creating an account, you agree to our{" "}
          <span className="text-[var(--primary-green)] font-medium cursor-pointer">
            Privacy Policy
          </span>
          . Your data is never sold.
        </p>
      </motion.div>
    </div>
  );
}
