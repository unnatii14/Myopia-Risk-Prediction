import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { useAuth } from "../context/AuthContext";
import { fetchLatestScreening, fetchHistory, type ScreeningRecord } from "../lib/historyApi";
import {
  ClipboardList, TrendingUp, Ruler, AlertCircle,
  Camera, ChevronRight, Activity, BookOpen, HelpCircle,
  Clock, ArrowUpRight, ArrowDownRight, Minus, CalendarDays,
  ShieldCheck, ShieldAlert, ShieldX, History, X,
  User, Eye, Smartphone, Sun, Users, Dna,
} from "lucide-react";

// ── Tool definitions ──────────────────────────────────────────
const TOOLS = [
  {
    id: "screen", title: "Myopia Risk Screening",
    description: "Assess your child's myopia risk in under 3 minutes",
    longDescription: "Answer 12 questions about demographics, family history, and lifestyle. Our XGBoost model (AUC 0.94) returns a personalised risk score.",
    icon: <ClipboardList className="h-6 w-6" />, href: "/screen",
    color: "var(--primary-green)", bgColor: "rgba(45,106,79,0.08)", badge: "Most Used",
  },
  {
    id: "image", title: "Image-Based Detection",
    description: "Upload a retinal image for AI-powered analysis",
    longDescription: "Deep-learning classifier analyses fundus photographs and returns a myopia probability score. Requires medical-grade retinal images.",
    icon: <Camera className="h-6 w-6" />, href: "/image-predictor",
    color: "#7C3AED", bgColor: "rgba(124,58,237,0.08)", badge: "New",
  },
  {
    id: "progression", title: "Progression Calculator",
    description: "Model future diopter progression over time",
    longDescription: "Based on Donovan et al. (2012) age-specific rates. Forecast diopter change year by year with treatment overlays.",
    icon: <TrendingUp className="h-6 w-6" />, href: "/progression",
    color: "#0891B2", bgColor: "rgba(8,145,178,0.08)",
  },
  {
    id: "axial", title: "Axial Elongation",
    description: "Track eye growth in millimetres",
    longDescription: "Uses SE-to-AL conversion (0.35 mm/D) to model axial length growth. Compare untreated vs treated trajectories.",
    icon: <Ruler className="h-6 w-6" />, href: "/axial",
    color: "#B45309", bgColor: "rgba(180,83,9,0.08)",
  },
  {
    id: "onset", title: "Onset Predictor",
    description: "Predict when myopia might begin",
    longDescription: "Compares spherical equivalent to age-matched hyperopic reserve norms (Zadnik/CLEERE). Returns estimated onset age and risk.",
    icon: <AlertCircle className="h-6 w-6" />, href: "/onset",
    color: "#DC2626", bgColor: "rgba(220,38,38,0.08)",
  },
];

const RISK_CONFIG = {
  LOW:      { icon: <ShieldCheck className="h-5 w-5" />, color: "#16a34a", bg: "#f0fdf4", border: "#bbf7d0", label: "Low Risk" },
  MODERATE: { icon: <ShieldAlert className="h-5 w-5" />, color: "#d97706", bg: "#fffbeb", border: "#fde68a", label: "Moderate Risk" },
  HIGH:     { icon: <ShieldX    className="h-5 w-5" />, color: "#dc2626", bg: "#fef2f2", border: "#fecaca", label: "High Risk" },
};

function daysSince(dateStr: string): number {
  const then = new Date(dateStr);
  const now  = new Date();
  return Math.floor((now.getTime() - then.getTime()) / (1000 * 60 * 60 * 24));
}

function TrendIcon({ current, prev }: { current: number; prev: number }) {
  const diff = current - prev;
  if (Math.abs(diff) < 3) return <Minus className="h-4 w-4 text-gray-400" />;
  if (diff > 0) return <ArrowUpRight className="h-4 w-4 text-red-500" />;
  return <ArrowDownRight className="h-4 w-4 text-green-600" />;
}

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [latest, setLatest]       = useState<ScreeningRecord | null | undefined>(undefined);
  const [history, setHistory]     = useState<ScreeningRecord[]>([]);
  const [selected, setSelected]   = useState<ScreeningRecord | null>(null);

  const firstName = user?.name?.split(" ")[0] ?? "there";
  const hour      = new Date().getHours();
  const greeting  = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  useEffect(() => {
    if (!user?.token) { setLatest(null); return; }
    fetchLatestScreening(user.token).then(setLatest).catch(() => setLatest(null));
    fetchHistory(user.token).then(setHistory).catch(() => setHistory([]));
  }, [user?.token]);

  const riskCfg   = latest ? RISK_CONFIG[latest.risk_level] : null;
  const prevRecord = history.length >= 2 ? history[1] : null;
  const days       = latest ? daysSince(latest.screened_at) : null;

  return (
    <div className="min-h-screen bg-[var(--background-mint)] px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">

        {/* ── Header ── */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }} className="mb-8">
          <p className="text-sm font-medium text-[var(--text-muted)]">{greeting},</p>
          <h1 className="mt-0.5 text-3xl font-bold text-[var(--text-dark)]">{firstName} 👋</h1>
          <p className="mt-2 max-w-xl text-[var(--text-muted)]">
            {latest
              ? "Here's your child's latest screening summary and tools."
              : "Welcome to MyopiaGuard. Run your first screening to get started."}
          </p>
        </motion.div>

        {/* ── Last Screening card ── */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.05 }} className="mb-8">
          <AnimatePresence mode="wait">
            {latest === undefined ? (
              // Loading skeleton
              <div key="skeleton" className="h-36 animate-pulse rounded-2xl border border-[var(--border)] bg-white" />
            ) : latest === null ? (
              // No history yet
              <motion.div
                key="empty"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-[var(--border)] bg-white py-10 text-center"
              >
                <div className="flex h-14 w-14 items-center justify-center rounded-full bg-[rgba(45,106,79,0.08)]">
                  <ClipboardList className="h-6 w-6" style={{ color: "var(--primary-green)" }} />
                </div>
                <div>
                  <p className="font-semibold text-[var(--text-dark)]">No screenings yet</p>
                  <p className="mt-1 text-sm text-[var(--text-muted)]">Run your first risk screening to see results here</p>
                </div>
                <button
                  onClick={() => navigate("/screen")}
                  className="mt-1 rounded-xl px-5 py-2.5 text-sm font-semibold text-white"
                  style={{ background: "var(--primary-green)" }}
                >
                  Start Screening
                </button>
              </motion.div>
            ) : (
              // Latest result card
              <motion.div
                key="result"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="rounded-2xl border bg-white shadow-sm overflow-hidden"
                style={{ borderColor: riskCfg?.border }}
              >
                <div className="flex items-center justify-between px-5 py-3" style={{ background: riskCfg?.bg }}>
                  <div className="flex items-center gap-2" style={{ color: riskCfg?.color }}>
                    {riskCfg?.icon}
                    <span className="font-bold text-sm uppercase tracking-wide">{riskCfg?.label}</span>
                  </div>
                  <span className="text-xs text-[var(--text-muted)] flex items-center gap-1">
                    <CalendarDays className="h-3.5 w-3.5" />
                    {days === 0 ? "Today" : days === 1 ? "Yesterday" : `${days} days ago`}
                  </span>
                </div>

                <div className="grid grid-cols-2 gap-4 p-5 sm:grid-cols-4">
                  {/* Risk score */}
                  <div>
                    <p className="text-xs text-[var(--text-muted)]">Risk Score</p>
                    <div className="flex items-end gap-1.5 mt-1">
                      <span className="text-3xl font-bold" style={{ color: riskCfg?.color }}>{latest.risk_score}%</span>
                      {prevRecord && <TrendIcon current={latest.risk_score} prev={prevRecord.risk_score} />}
                    </div>
                    {prevRecord && (
                      <p className="text-xs text-[var(--text-muted)] mt-0.5">
                        was {prevRecord.risk_score}% before
                      </p>
                    )}
                  </div>

                  {/* Child */}
                  <div>
                    <p className="text-xs text-[var(--text-muted)]">Child</p>
                    <p className="mt-1 font-semibold text-[var(--text-dark)]">{latest.child_name || "—"}</p>
                    <p className="text-xs text-[var(--text-muted)] mt-0.5">
                      {latest.input_data?.age ? `Age ${latest.input_data.age}` : ""}
                    </p>
                  </div>

                  {/* Refractive error */}
                  <div>
                    <p className="text-xs text-[var(--text-muted)]">Refractive Error</p>
                    <p className="mt-1 font-semibold" style={{ color: latest.has_re ? "#dc2626" : "#16a34a" }}>
                      {latest.has_re ? "Detected" : "Unlikely"}
                    </p>
                    {latest.diopters && (
                      <p className="text-xs text-[var(--text-muted)] mt-0.5">~-{latest.diopters}D · {latest.severity}</p>
                    )}
                  </div>

                  {/* CTA */}
                  <div className="flex items-center justify-end">
                    <button
                      onClick={() => navigate("/screen")}
                      className="flex items-center gap-1.5 rounded-xl border px-4 py-2 text-sm font-semibold transition-colors hover:bg-[var(--background-mint)]"
                      style={{ borderColor: riskCfg?.border, color: riskCfg?.color }}
                    >
                      <Clock className="h-3.5 w-3.5" />
                      Re-screen
                    </button>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* ── History strip (if multiple records) ── */}
        {history.length > 1 && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.1 }} className="mb-8">
            <div className="flex items-center gap-2 mb-3">
              <History className="h-4 w-4" style={{ color: "var(--primary-green)" }} />
              <h2 className="text-sm font-bold text-[var(--text-dark)]">Screening History</h2>
            </div>
            <div className="flex gap-3 overflow-x-auto pb-1">
              {history.slice(0, 6).map((rec, i) => {
                const cfg = RISK_CONFIG[rec.risk_level];
                const d   = daysSince(rec.screened_at);
                return (
                  <button
                    key={rec.id}
                    onClick={() => setSelected(rec)}
                    className="flex-shrink-0 rounded-xl border bg-white px-4 py-3 text-center min-w-[90px] transition-shadow hover:shadow-md hover:-translate-y-0.5 transition-transform"
                    style={{ borderColor: cfg.border }}
                  >
                    <p className="text-xs text-[var(--text-muted)]">{i === 0 ? "Latest" : `${d}d ago`}</p>
                    <p className="text-lg font-bold mt-0.5" style={{ color: cfg.color }}>{rec.risk_score}%</p>
                    <p className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: cfg.color }}>{rec.risk_level}</p>
                    <p className="text-[10px] text-[var(--text-muted)] mt-0.5">tap to view</p>
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}

        {/* ── Tools grid ── */}
        <div className="mb-4 flex items-center gap-2">
          <Activity className="h-5 w-5" style={{ color: "var(--primary-green)" }} />
          <h2 className="text-lg font-bold text-[var(--text-dark)]">Your Tools</h2>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {TOOLS.map((tool, i) => (
            <motion.button
              key={tool.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.12 + 0.06 * i }}
              whileHover={{ y: -3, transition: { duration: 0.15 } }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate(tool.href)}
              className="relative flex flex-col items-start rounded-2xl border border-[var(--border)] bg-white p-5 text-left shadow-sm transition-shadow hover:shadow-md"
            >
              {tool.badge && (
                <span className="absolute right-4 top-4 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide"
                  style={{ background: tool.bgColor, color: tool.color }}>
                  {tool.badge}
                </span>
              )}
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl" style={{ background: tool.bgColor, color: tool.color }}>
                {tool.icon}
              </div>
              <h3 className="mb-1 font-bold text-[var(--text-dark)]">{tool.title}</h3>
              <p className="mb-3 text-sm leading-relaxed text-[var(--text-muted)]">{tool.description}</p>
              <p className="mb-4 text-xs leading-relaxed text-[var(--text-muted)] opacity-70">{tool.longDescription}</p>
              <div className="mt-auto flex items-center gap-1 text-sm font-semibold" style={{ color: tool.color }}>
                Open tool <ChevronRight className="h-4 w-4" />
              </div>
            </motion.button>
          ))}
        </div>

        {/* ── Quick links ── */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4, delay: 0.6 }} className="mt-10 flex flex-wrap gap-3">
          {[
            { label: "Research & Methodology", href: "/about",  icon: <BookOpen className="h-4 w-4" /> },
            { label: "Frequently Asked Questions", href: "/faq", icon: <HelpCircle className="h-4 w-4" /> },
          ].map((link) => (
            <button
              key={link.href}
              onClick={() => navigate(link.href)}
              className="flex items-center gap-2 rounded-full border border-[var(--border)] bg-white px-4 py-2 text-sm font-medium text-[var(--text-muted)] transition-colors hover:border-[var(--primary-green)] hover:text-[var(--primary-green)]"
            >
              {link.icon}
              {link.label}
            </button>
          ))}
        </motion.div>

      </div>

      {/* ── History detail modal ── */}
      <AnimatePresence>
        {selected && (() => {
          const cfg = RISK_CONFIG[selected.risk_level];
          const d   = selected.input_data;
          const days = daysSince(selected.screened_at);
          const dateStr = new Date(selected.screened_at).toLocaleDateString("en-IN", {
            day: "numeric", month: "long", year: "numeric",
          });
          return (
            <motion.div
              key="modal-backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/40 px-4 pb-4 sm:pb-0"
              onClick={() => setSelected(null)}
            >
              <motion.div
                key="modal-panel"
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 40 }}
                transition={{ type: "spring", stiffness: 320, damping: 32 }}
                className="w-full max-w-md rounded-3xl bg-white shadow-2xl overflow-hidden"
                onClick={(e) => e.stopPropagation()}
              >
                {/* Header */}
                <div className="flex items-center justify-between px-5 py-4" style={{ background: cfg.bg }}>
                  <div className="flex items-center gap-2" style={{ color: cfg.color }}>
                    {cfg.icon}
                    <span className="font-bold">{cfg.label} — {selected.risk_score}%</span>
                  </div>
                  <button onClick={() => setSelected(null)} className="rounded-full p-1.5 hover:bg-black/10 transition-colors">
                    <X className="h-4 w-4 text-[var(--text-muted)]" />
                  </button>
                </div>

                <div className="px-5 py-4 space-y-4">
                  {/* Date */}
                  <p className="text-xs text-[var(--text-muted)] flex items-center gap-1.5">
                    <CalendarDays className="h-3.5 w-3.5" />
                    {dateStr} &nbsp;·&nbsp; {days === 0 ? "Today" : days === 1 ? "Yesterday" : `${days} days ago`}
                  </p>

                  {/* Child info row */}
                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { icon: <User className="h-3.5 w-3.5"/>, label: "Child", value: selected.child_name || "—" },
                      { icon: <CalendarDays className="h-3.5 w-3.5"/>, label: "Age", value: d?.age ? `${d.age} yrs` : "—" },
                      { icon: <Users className="h-3.5 w-3.5"/>, label: "Sex", value: d?.sex ? String(d.sex) : "—" },
                      { icon: <Dna className="h-3.5 w-3.5"/>, label: "Family history", value: d?.familyHistory ? "Yes" : "No" },
                    ].map((item) => (
                      <div key={item.label} className="rounded-xl bg-[var(--background-mint)] px-3 py-2.5">
                        <div className="flex items-center gap-1 text-[var(--text-muted)] mb-1">{item.icon}<p className="text-[10px]">{item.label}</p></div>
                        <p className="text-sm font-semibold text-[var(--text-dark)]">{item.value}</p>
                      </div>
                    ))}
                  </div>

                  {/* Lifestyle row */}
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { icon: <Smartphone className="h-3.5 w-3.5"/>, label: "Screen time", value: d?.screenTime ? `${d.screenTime}h/day` : "—" },
                      { icon: <Sun className="h-3.5 w-3.5"/>, label: "Outdoor time", value: d?.outdoorTime ? `${d.outdoorTime}h/day` : "—" },
                      { icon: <Eye className="h-3.5 w-3.5"/>, label: "Near work", value: d?.nearWork ? `${d.nearWork}h/day` : "—" },
                    ].map((item) => (
                      <div key={item.label} className="rounded-xl bg-[var(--background-mint)] px-3 py-2.5 text-center">
                        <div className="flex items-center justify-center gap-1 text-[var(--text-muted)] mb-1">{item.icon}</div>
                        <p className="text-sm font-bold text-[var(--text-dark)]">{item.value}</p>
                        <p className="text-[10px] text-[var(--text-muted)]">{item.label}</p>
                      </div>
                    ))}
                  </div>

                  {/* ML result row */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="rounded-xl border px-3 py-2.5" style={{ borderColor: cfg.border, background: cfg.bg }}>
                      <p className="text-[10px] text-[var(--text-muted)]">Refractive Error</p>
                      <p className="font-bold text-sm mt-0.5" style={{ color: selected.has_re ? "#dc2626" : "#16a34a" }}>
                        {selected.has_re ? "Detected" : "Unlikely"}
                      </p>
                    </div>
                    <div className="rounded-xl border px-3 py-2.5" style={{ borderColor: cfg.border, background: cfg.bg }}>
                      <p className="text-[10px] text-[var(--text-muted)]">Est. Severity</p>
                      <p className="font-bold text-sm mt-0.5" style={{ color: cfg.color }}>
                        {selected.diopters ? `-${selected.diopters}D · ${selected.severity}` : selected.has_re ? "—" : "None"}
                      </p>
                    </div>
                  </div>

                  {/* Action button */}
                  <button
                    onClick={() => { setSelected(null); navigate("/screen"); }}
                    className="w-full rounded-2xl py-3 text-sm font-bold text-white transition-opacity hover:opacity-90"
                    style={{ background: "var(--primary-green)" }}
                  >
                    Run New Screening
                  </button>
                </div>
              </motion.div>
            </motion.div>
          );
        })()}
      </AnimatePresence>

    </div>
  );
}
