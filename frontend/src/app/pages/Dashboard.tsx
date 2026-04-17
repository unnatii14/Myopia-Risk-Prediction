import { useNavigate } from "react-router";
import { motion } from "motion/react";
import { useAuth } from "../context/AuthContext";
import {
  ClipboardList, TrendingUp, Ruler, AlertCircle,
  Camera, ChevronRight, Activity, BookOpen, HelpCircle,
} from "lucide-react";

interface Tool {
  id: string;
  title: string;
  description: string;
  longDescription: string;
  icon: React.ReactNode;
  href: string;
  color: string;
  bgColor: string;
  badge?: string;
}

const TOOLS: Tool[] = [
  {
    id: "screen",
    title: "Myopia Risk Screening",
    description: "Assess your child's myopia risk in under 3 minutes",
    longDescription:
      "Answer 12 questions about demographics, family history, and lifestyle. Our XGBoost model (AUC 0.94) returns a personalised risk score with actionable recommendations.",
    icon: <ClipboardList className="h-6 w-6" />,
    href: "/screen",
    color: "var(--primary-green)",
    bgColor: "rgba(45,106,79,0.08)",
    badge: "Most Used",
  },
  {
    id: "image",
    title: "Image-Based Detection",
    description: "Upload a retinal or eye image for AI-powered analysis",
    longDescription:
      "Our deep-learning image classifier (224 × 224 input) analyses retinal photographs and returns a myopia probability score with confidence level.",
    icon: <Camera className="h-6 w-6" />,
    href: "/image-predictor",
    color: "#7C3AED",
    bgColor: "rgba(124,58,237,0.08)",
    badge: "New",
  },
  {
    id: "progression",
    title: "Progression Calculator",
    description: "Model future diopter progression over time",
    longDescription:
      "Based on Donovan et al. (2012) age-specific rates. Adjust for ethnicity, sex, and treatment method to forecast diopter change year by year.",
    icon: <TrendingUp className="h-6 w-6" />,
    href: "/progression",
    color: "#0891B2",
    bgColor: "rgba(8,145,178,0.08)",
  },
  {
    id: "axial",
    title: "Axial Elongation",
    description: "Track eye growth in millimetres over time",
    longDescription:
      "Uses the SE-to-AL conversion (0.35 mm/D) to model axial length growth. Compare untreated vs treated trajectories with intervention overlays.",
    icon: <Ruler className="h-6 w-6" />,
    href: "/axial",
    color: "#B45309",
    bgColor: "rgba(180,83,9,0.08)",
  },
  {
    id: "onset",
    title: "Onset Predictor",
    description: "Predict when myopia might begin for pre-myopic children",
    longDescription:
      "Compares current spherical equivalent to age-matched hyperopic reserve norms (Zadnik/CLEERE). Returns estimated onset age and risk classification.",
    icon: <AlertCircle className="h-6 w-6" />,
    href: "/onset",
    color: "#DC2626",
    bgColor: "rgba(220,38,38,0.08)",
  },
];

const QUICK_LINKS = [
  { label: "Research & Methodology", href: "/about", icon: <BookOpen className="h-4 w-4" /> },
  { label: "Frequently Asked Questions", href: "/faq", icon: <HelpCircle className="h-4 w-4" /> },
];

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const firstName = user?.name?.split(" ")[0] ?? "there";
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div className="min-h-screen bg-[var(--background-mint)] px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-5xl">

        {/* ── Header ── */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-10"
        >
          <p className="text-sm font-medium text-[var(--text-muted)]">{greeting},</p>
          <h1 className="mt-0.5 text-3xl font-bold text-[var(--text-dark)]">
            {firstName} 👋
          </h1>
          <p className="mt-2 max-w-xl text-[var(--text-muted)]">
            MyopiaGuard gives you five evidence-based tools to screen, track, and manage childhood myopia.
            Choose a tool below to get started.
          </p>
        </motion.div>

        {/* ── Stats bar ── */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.05 }}
          className="mb-10 grid grid-cols-3 gap-4 rounded-2xl border border-[var(--border)] bg-white px-6 py-5 shadow-sm"
        >
          {[
            { label: "Screening Model AUC", value: "0.94" },
            { label: "Risk Features Analysed", value: "30" },
            { label: "Evidence-Based Tools", value: "5" },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <p className="text-2xl font-bold" style={{ color: "var(--primary-green)" }}>
                {stat.value}
              </p>
              <p className="mt-0.5 text-xs text-[var(--text-muted)]">{stat.label}</p>
            </div>
          ))}
        </motion.div>

        {/* ── Tools grid ── */}
        <div className="mb-6 flex items-center gap-2">
          <Activity className="h-5 w-5" style={{ color: "var(--primary-green)" }} />
          <h2 className="text-lg font-bold text-[var(--text-dark)]">Your Tools</h2>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {TOOLS.map((tool, i) => (
            <motion.button
              key={tool.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.08 * i }}
              whileHover={{ y: -3, transition: { duration: 0.15 } }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate(tool.href)}
              className="relative flex flex-col items-start rounded-2xl border border-[var(--border)] bg-white p-5 text-left shadow-sm transition-shadow hover:shadow-md"
            >
              {/* Badge */}
              {tool.badge && (
                <span
                  className="absolute right-4 top-4 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide"
                  style={{ background: tool.bgColor, color: tool.color }}
                >
                  {tool.badge}
                </span>
              )}

              {/* Icon */}
              <div
                className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl"
                style={{ background: tool.bgColor, color: tool.color }}
              >
                {tool.icon}
              </div>

              {/* Text */}
              <h3 className="mb-1 font-bold text-[var(--text-dark)]">{tool.title}</h3>
              <p className="mb-3 text-sm leading-relaxed text-[var(--text-muted)]">
                {tool.description}
              </p>
              <p className="mb-4 text-xs leading-relaxed text-[var(--text-muted)] opacity-70">
                {tool.longDescription}
              </p>

              {/* CTA */}
              <div
                className="mt-auto flex items-center gap-1 text-sm font-semibold"
                style={{ color: tool.color }}
              >
                Open tool <ChevronRight className="h-4 w-4" />
              </div>
            </motion.button>
          ))}
        </div>

        {/* ── Quick links ── */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.4, delay: 0.5 }}
          className="mt-10 flex flex-wrap gap-3"
        >
          {QUICK_LINKS.map((link) => (
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
    </div>
  );
}
