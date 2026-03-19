import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { useNavigate } from "react-router";
import { jsPDF } from "jspdf";
import {
  Download, AlertTriangle, CheckCircle, Sun,
  Smartphone, Users, Calendar, ExternalLink,
  Eye, Loader2
} from "lucide-react";
import RiskGauge from "../components/RiskGauge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../components/ui/accordion";
import { useAuth } from "../context/AuthContext";

const API_URL  = "http://localhost:5001";
const NODE_URL = "http://localhost:5000";

interface ScreeningData {
  age: number;
  sex: string;
  height: number;
  weight: number;
  familyHistory: boolean | null;
  parentsMyopic: string;
  screenTime: number;
  nearWork: number;
  outdoorTime: number;
  sports: string;
  vitaminD: boolean | null;
}

interface PredictionResult {
  risk_score: number;
  risk_level: "LOW" | "MODERATE" | "HIGH";
  risk_probability: number;
  has_re: boolean;
  re_probability: number;
  diopters: number | null;
  severity: string | null;
}

export default function Results() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [data, setData] = useState<ScreeningData | null>(null);
  const [riskScore, setRiskScore] = useState(0);
  const [riskLevel, setRiskLevel] = useState<"LOW" | "MODERATE" | "HIGH">("LOW");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState<string | null>(null);

  // Save screening result to MongoDB (only when user is logged in)
  const saveRecord = (screeningData: ScreeningData, pred: PredictionResult) => {
    if (!user?.token) return;
    fetch(`${NODE_URL}/api/myopia/save`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${user.token}`,
      },
      body: JSON.stringify({ screeningData, prediction: pred }),
    }).catch(() => {/* silently ignore — saving is best-effort */});
  };

  const downloadPdf = () => {
    if (!data) return;
    const doc = new jsPDF({ unit: "mm", format: "a4" });
    const pageW = doc.internal.pageSize.getWidth();
    const margin = 18;
    const col = pageW - margin * 2;

    // ── Header bar ──
    doc.setFillColor(42, 120, 90);
    doc.rect(0, 0, pageW, 28, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(18);
    doc.setFont("helvetica", "bold");
    doc.text("MyopiaGuard - Risk Assessment Report", margin, 18);

    // ── Date / meta ──
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.text(`Generated: ${new Date().toLocaleDateString("en-IN")}   |   Powered by GradientBoosting ML - AUC 0.893`, margin, 24);

    let y = 38;
    doc.setTextColor(30, 30, 30);

    // ── Child profile ──
    doc.setFillColor(237, 247, 241);
    doc.roundedRect(margin, y, col, 26, 3, 3, "F");
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("Child Profile", margin + 4, y + 8);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    if (user?.childName) doc.text(`Child Name: ${user.childName}`, margin + 4, y + 16);
    doc.text(`Age: ${data.age} years`, margin + 4, user?.childName ? y + 24 : y + 16);
    doc.text(`Sex: ${data.sex === "male" ? "Male" : "Female"}`, margin + 50, user?.childName ? y + 24 : y + 16);
    if (data.height > 0) doc.text(`Height: ${data.height} cm`, margin + 100, user?.childName ? y + 24 : y + 16);
    if (data.weight > 0) doc.text(`Weight: ${data.weight} kg`, margin + 145, user?.childName ? y + 24 : y + 16);
    y += user?.childName ? 42 : 34;

    // ── Risk result box ──
    const riskColors: Record<string, [number, number, number]> = {
      HIGH: [231, 111, 81],
      MODERATE: [244, 162, 97],
      LOW: [82, 183, 136],
    };
    const [r, g, b] = riskColors[riskLevel] ?? [100, 150, 100];
    doc.setFillColor(r, g, b);
    doc.roundedRect(margin, y, col, 28, 3, 3, "F");
    doc.setTextColor(255, 255, 255);
    doc.setFontSize(22);
    doc.setFont("helvetica", "bold");
    doc.text(`${riskLevel} RISK - ${riskScore}%`, margin + 4, y + 18);
    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.text("Myopia Progression Risk Score", margin + 4, y + 25);
    y += 36;
    doc.setTextColor(30, 30, 30);

    // ── Three-stage summary ──
    const stages = [
      ["Stage 1 - Refractive Error",
        prediction
          ? `${prediction.has_re ? "YES" : "NO"} | ${Math.round(prediction.re_probability * 100)}%`
          : `${riskScore > 60 ? "POSSIBLE" : "UNLIKELY"} | ${Math.round(riskScore * 0.8)}%`
      ],
      ["Stage 2 - Progression Risk", `${riskLevel} | ${riskScore}%`],
      ["Stage 3 - Est. Severity",
        prediction?.diopters != null
          ? `-${prediction.diopters}D (${prediction.severity})`
          : prediction && !prediction.has_re
          ? "No RE detected"
          : riskLevel === "HIGH" ? "-3.2D (Moderate)" : riskLevel === "MODERATE" ? "-1.5D (Mild)" : "-0.5D (Very Mild)"
      ],
    ];
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("Clinical Stage Summary", margin, y); y += 6;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    stages.forEach(([label, val]) => {
      doc.setFillColor(248, 250, 248);
      doc.roundedRect(margin, y, col, 10, 2, 2, "F");
      doc.text(label, margin + 3, y + 7);
      doc.text(val, margin + col - 3, y + 7, { align: "right" });
      y += 13;
    });
    y += 4;

    // ── Input factors ──
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("Reported Risk Factors", margin, y); y += 6;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    const factors: [string, string][] = [
      ["Screen time", `${data.screenTime} hrs/day (recommended <2)`],
      ["Outdoor time", `${data.outdoorTime} hrs/day (recommended 2+ hrs)`],
      ["Near work", `${data.nearWork} hrs/day`],
      ["Parents myopic", data.parentsMyopic === "both" ? "Both parents" : data.parentsMyopic === "one" ? "One parent" : "None"],
      ["Family history", data.familyHistory ? "Yes" : "No"],
      ["Sports participation", data.sports || "Not specified"],
      ["Vitamin D", data.vitaminD ? "Taking supplement" : "Not taking"],
    ];
    factors.forEach(([k, v]) => {
      doc.setFillColor(252, 252, 252);
      doc.roundedRect(margin, y, col, 9, 2, 2, "F");
      doc.setFont("helvetica", "bold");
      doc.text(k, margin + 3, y + 6);
      doc.setFont("helvetica", "normal");
      doc.text(v, margin + 70, y + 6);
      y += 11;
    });
    y += 6;

    // ── Recommendations ──
    if (y > 230) { doc.addPage(); y = 20; }
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.text("Personalised Recommendations", margin, y); y += 6;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    const recs: string[] = [];
    if (data.outdoorTime < 2) recs.push("Increase outdoor time to at least 2 hours per day (daylight exposure is the #1 protector).");
    if (data.screenTime > 2) recs.push("Reduce recreational screen time below 2 hours/day. Apply the 20-20-20 rule.");
    recs.push(riskLevel === "HIGH" ? "Book an eye exam immediately - recommended every 3-6 months for high-risk children." : "Schedule an annual comprehensive eye examination.");
    recs.push("Ask your eye doctor about myopia control options: atropine 0.01%, ortho-K lenses, or myopia-control spectacles.");
    if (data.parentsMyopic === "both") recs.push("Both parents are myopic - inform your doctor. Monitor closely.");
    recs.forEach((rec, i) => {
      const lines = doc.splitTextToSize(`${i + 1}. ${rec}`, col - 4);
      doc.text(lines, margin + 3, y);
      y += lines.length * 5.5 + 3;
    });
    y += 4;

    // ── Disclaimer ──
    if (y > 260) { doc.addPage(); y = 20; }
    doc.setFillColor(255, 251, 230);
    doc.roundedRect(margin, y, col, 18, 2, 2, "F");
    doc.setFontSize(8);
    doc.setTextColor(120, 90, 20);
    const disclaimer = "DISCLAIMER: This AI assessment is not a medical diagnosis. It provides a risk estimate based on lifestyle and family history. Please consult a qualified ophthalmologist for proper examination and diagnosis.";
    doc.text(doc.splitTextToSize(disclaimer, col - 6), margin + 3, y + 6);

    doc.save(`MyopiaGuard_Report_${data.age}yr_${new Date().toISOString().slice(0,10)}.pdf`);
  };

  useEffect(() => {
    const stored = sessionStorage.getItem("screeningData");
    if (!stored) {
      navigate("/screen");
      return;
    }

    const parsedData: ScreeningData = JSON.parse(stored);
    setData(parsedData);

    // Call the ML backend
    fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(parsedData),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`API error ${res.status}`);
        return res.json();
      })
      .then((result: PredictionResult) => {
        setPrediction(result);
        setRiskScore(result.risk_score);
        setRiskLevel(result.risk_level);
        setLoading(false);
        saveRecord(parsedData, { ...result, source: "ml" } as PredictionResult & { source: string });
      })
      .catch((err) => {
        console.error("API call failed:", err);
        setApiError("Could not reach prediction server. Showing rule-based estimate.");
        // Fallback to rule-based scoring
        const score = fallbackRiskScore(parsedData);
        const level: "LOW" | "MODERATE" | "HIGH" = score < 40 ? "LOW" : score < 70 ? "MODERATE" : "HIGH";
        setRiskScore(score);
        setRiskLevel(level);
        setLoading(false);
        saveRecord(parsedData, {
          risk_score: score, risk_level: level, risk_probability: score / 100,
          has_re: score > 60, re_probability: score * 0.8 / 100,
          diopters: null, severity: null, source: "rule-based",
        } as PredictionResult & { source: string });
      });
  }, [navigate]);

  const fallbackRiskScore = (d: ScreeningData): number => {
    let s = 30;
    if (d.age <= 8) s += 15; else if (d.age <= 10) s += 10; else if (d.age <= 12) s += 5;
    if (d.parentsMyopic === "both") s += 25;
    else if (d.parentsMyopic === "one") s += 15;
    else if (d.familyHistory) s += 10;
    if (d.screenTime > 6) s += 20; else if (d.screenTime > 4) s += 15; else if (d.screenTime > 2) s += 8;
    if (d.outdoorTime < 1) s += 20; else if (d.outdoorTime < 2) s += 10; else if (d.outdoorTime >= 3) s -= 10;
    if (d.nearWork > 6) s += 15; else if (d.nearWork > 4) s += 8;
    if (d.vitaminD) s -= 5;
    if (d.sports === "regular") s -= 5;
    return Math.min(Math.max(s, 0), 100);
  };

  if (!data || loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-[var(--background-mint)] to-white flex flex-col items-center justify-center gap-4">
        <Loader2 className="w-12 h-12 text-[var(--primary-green)] animate-spin" />
        <p className="text-lg font-medium text-[var(--text-muted)]">
          {loading ? "Analysing with AI model…" : "Loading…"}
        </p>
      </div>
    );
  }

  const topRiskFactors = [
    ...(data.screenTime > 2 ? [{
      icon: Smartphone,
      title: "High Screen Time",
      value: `${data.screenTime} hrs/day`,
      recommendation: "Recommended: <2 hrs",
      type: "risk" as const,
    }] : []),
    ...(data.outdoorTime < 2 ? [{
      icon: Sun,
      title: "Low Outdoor Time",
      value: `${data.outdoorTime} hrs/day`,
      recommendation: "Need: ≥2 hrs/day",
      type: "risk" as const,
    }] : []),
    ...(data.parentsMyopic === "both" ? [{
      icon: Users,
      title: "Both Parents Myopic",
      value: "Genetic risk elevated",
      recommendation: "6x higher risk",
      type: "risk" as const,
    }] : []),
    ...(data.vitaminD ? [{
      icon: CheckCircle,
      title: "Vitamin D Supplement",
      value: "Taking supplement",
      recommendation: "Protective ✅",
      type: "protective" as const,
    }] : []),
    ...(data.outdoorTime >= 2 ? [{
      icon: Sun,
      title: "Good Outdoor Time",
      value: `${data.outdoorTime} hrs/day`,
      recommendation: "Protective ✅",
      type: "protective" as const,
    }] : []),
  ];

  const recommendations = [
    ...(data.outdoorTime < 2 ? [{
      icon: Sun,
      title: "Increase Outdoor Time",
      description: "Aim for at least 2 hours of outdoor daylight exposure daily. This is the single most effective natural intervention.",
    }] : []),
    ...(data.screenTime > 2 ? [{
      icon: Smartphone,
      title: "Reduce Screen Time",
      description: "Limit recreational screen time to under 2 hours per day. Take 20-second breaks every 20 minutes (20-20-20 rule).",
    }] : []),
    {
      icon: Calendar,
      title: "Schedule Eye Exam",
      description: riskLevel === "HIGH" 
        ? "Book an eye exam immediately. High-risk children need screening every 3-6 months."
        : "Get an annual comprehensive eye examination from a qualified optometrist or ophthalmologist.",
    },
    {
      icon: Eye,
      title: "Consider Myopia Control",
      description: "Discuss myopia control options with your eye doctor: atropine eye drops (0.01%), myopia control glasses, or ortho-K lenses.",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[var(--background-mint)] to-white py-12 px-4">
      <div className="max-w-5xl mx-auto">
        {/* HERO RESULT CARD */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-3xl p-8 shadow-2xl mb-8"
        >
          <div className="flex flex-col lg:flex-row gap-8">
            {/* Left - Summary */}
            <div className="lg:w-1/3">
              <h2 className="text-2xl font-bold text-[var(--text-dark)] mb-4">
                Risk Assessment Complete
              </h2>
              <div className="space-y-2 text-sm text-[var(--text-muted)]">
                <p>
                  <strong>Age:</strong> {data.age} years
                </p>
                <p>
                  <strong>Sex:</strong> {data.sex === "male" ? "Male" : "Female"}
                </p>
                <p className="text-xs pt-2">
                  <strong>Date:</strong> {new Date().toLocaleDateString("en-IN")}
                </p>
              </div>

              <button
                onClick={downloadPdf}
                className="mt-6 flex items-center gap-2 px-4 py-2 border-2 border-[var(--primary-green)] text-[var(--primary-green)] rounded-full hover:bg-[var(--primary-green)] hover:text-white transition-all"
              >
                <Download className="w-4 h-4" />
                Download PDF
              </button>
            </div>

            {/* Center - Risk Gauge */}
            <div className="lg:w-1/3 flex flex-col items-center justify-center">
              <div className={`rounded-full p-3 transition-all duration-700 ${
                riskLevel === "HIGH"
                  ? "shadow-[0_0_70px_rgba(231,111,81,0.35)]"
                  : riskLevel === "MODERATE"
                  ? "shadow-[0_0_70px_rgba(244,162,97,0.35)]"
                  : "shadow-[0_0_70px_rgba(82,183,136,0.35)]"
              }`}>
                <RiskGauge score={riskScore} />
              </div>
              <motion.div
                animate={riskLevel === "HIGH" ? { scale: [1, 1.07, 1] } : {}}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="text-center mt-4"
              >
                <div
                  className={`inline-block px-6 py-2 rounded-full font-bold text-lg ${
                    riskLevel === "LOW"
                      ? "bg-[var(--low-risk)]/20 text-[var(--low-risk)]"
                      : riskLevel === "MODERATE"
                      ? "bg-[var(--moderate-risk)]/20 text-[var(--moderate-risk)]"
                      : "bg-[var(--high-risk)]/20 text-[var(--high-risk)]"
                  }`}
                >
                  {riskLevel} RISK
                </div>
              </motion.div>
            </div>

            {/* Right - Stages */}
            <div className="lg:w-1/3 space-y-3">
              <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl">
                <p className="text-xs text-[var(--text-muted)] mb-1">Stage 1</p>
                <p className="font-bold text-[var(--text-dark)]">Refractive Error Likely</p>
                <p className="text-sm">
                  <span className={prediction?.has_re ? "text-[var(--warning-coral)]" : "text-[var(--secondary-green)]"}>
                    {prediction ? (prediction.has_re ? "YES" : "NO") : (riskScore > 60 ? "POSSIBLE" : "UNLIKELY")}
                  </span>
                  {" · "}
                  <span className="text-[var(--text-muted)]">
                    {prediction ? `${Math.round(prediction.re_probability * 100)}%` : `${Math.round(riskScore * 0.8)}%`}
                  </span>
                </p>
              </div>

              <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl">
                <p className="text-xs text-[var(--text-muted)] mb-1">Stage 2</p>
                <p className="font-bold text-[var(--text-dark)]">Progression Risk</p>
                <p className="text-sm">
                  <span className={riskLevel === "HIGH" ? "text-[var(--warning-coral)]" : riskLevel === "MODERATE" ? "text-[var(--moderate-risk)]" : "text-[var(--secondary-green)]"}>
                    {riskLevel}
                  </span>
                  {" · "}
                  <span className="text-[var(--text-muted)]">{riskScore}%</span>
                </p>
              </div>

              <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl">
                <p className="text-xs text-[var(--text-muted)] mb-1">Stage 3</p>
                <p className="font-bold text-[var(--text-dark)]">Est. Severity</p>
                <p className="text-sm text-[var(--text-muted)]">
                  {prediction?.diopters != null
                    ? `~-${prediction.diopters}D · ${prediction.severity ?? ""}`
                    : prediction && !prediction.has_re
                    ? "No RE detected"
                    : riskLevel === "HIGH" ? "~-3.2D · Moderate" : riskLevel === "MODERATE" ? "~-1.5D · Mild" : "~-0.5D · Very Mild"}
                </p>
              </div>
            </div>
          </div>

          {apiError && (
            <div className="mt-4 px-4 py-2 bg-amber-50 border border-amber-200 rounded-xl text-xs text-amber-700">
              ⚠️ {apiError}
            </div>
          )}
          <div className="mt-6 pt-6 border-t border-gray-200 text-xs text-[var(--text-muted)] text-center">
            {prediction
              ? "Powered by GradientBoosting ML Model · AUC 0.893 · Trained on 5,000+ Indian children · Live ML Prediction"
              : "Rule-based estimate (ML server offline) · For real predictions, start the backend"}
          </div>
        </motion.div>

        {/* TOP RISK FACTORS */}
        {topRiskFactors.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-8"
          >
            <h3 className="text-2xl font-bold text-[var(--text-dark)] mb-4">
              Key Risk & Protective Factors
            </h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {topRiskFactors.map((factor, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className={`p-4 rounded-2xl border border-l-4 ${
                    factor.type === "risk"
                      ? "bg-[var(--warning-coral)]/5 border-[var(--warning-coral)]/20 border-l-[var(--warning-coral)]"
                      : "bg-[var(--secondary-green)]/5 border-[var(--secondary-green)]/20 border-l-[var(--secondary-green)]"
                  }`}
                >
                  <factor.icon
                    className={`w-8 h-8 mb-3 ${
                      factor.type === "risk" ? "text-[var(--warning-coral)]" : "text-[var(--secondary-green)]"
                    }`}
                  />
                  <h4 className="font-bold text-[var(--text-dark)] mb-1">
                    {factor.title}
                  </h4>
                  <p className="text-sm text-[var(--text-muted)] mb-1">
                    {factor.value}
                  </p>
                  <p className="text-xs font-medium" style={{ 
                    color: factor.type === "risk" ? "var(--warning-coral)" : "var(--secondary-green)" 
                  }}>
                    {factor.recommendation}
                  </p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* CLINICAL FLAGS */}
        {riskLevel === "HIGH" && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mb-8 p-6 bg-amber-50 border-2 border-amber-300 rounded-2xl"
          >
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-6 h-6 text-amber-600 flex-shrink-0 mt-1" />
              <div>
                <h4 className="font-bold text-amber-900 mb-2">Clinical Flags</h4>
                <ul className="space-y-2 text-sm text-amber-800">
                  {data.age <= 10 && (
                    <li>• Age ≤10 with HIGH RISK — eye exam every 3-6 months recommended</li>
                  )}
                  <li>• No myopia control currently in use — discuss options with eye doctor</li>
                  {data.parentsMyopic === "both" && (
                    <li>• Both parents myopic — 6x increased risk, close monitoring needed</li>
                  )}
                </ul>
              </div>
            </div>
          </motion.div>
        )}

        {/* RECOMMENDATIONS */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mb-8 p-6 bg-gradient-to-br from-[var(--secondary-green)]/10 to-white rounded-2xl border border-[var(--secondary-green)]/30"
        >
          <h3 className="text-2xl font-bold text-[var(--text-dark)] mb-6">
            Personalized Recommendations
          </h3>
          <div className="space-y-4">
            {recommendations.map((rec, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + index * 0.1 }}
                className="flex gap-4 p-4 bg-white rounded-xl"
              >
                <rec.icon className="w-6 h-6 text-[var(--primary-green)] flex-shrink-0 mt-1" />
                <div>
                  <h4 className="font-bold text-[var(--text-dark)] mb-1">
                    {rec.title}
                  </h4>
                  <p className="text-sm text-[var(--text-muted)]">
                    {rec.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* CORRECTION GUIDE */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="mb-8"
        >
          <h3 className="text-2xl font-bold text-[var(--text-dark)] mb-4">
            Understanding Treatment Options
          </h3>
          <Accordion type="single" collapsible className="bg-white rounded-2xl overflow-hidden shadow-lg">
            <AccordionItem value="atropine">
              <AccordionTrigger className="px-6 py-4 hover:bg-[var(--background-mint)]">
                <div className="flex items-center gap-3">
                  <div className="flex gap-0.5">
                    {[1, 2].map((i) => (
                      <span key={i} className="text-[var(--moderate-risk)]">★</span>
                    ))}
                  </div>
                  <span className="font-bold">Atropine 0.01% Eye Drops</span>
                  <span className="text-sm text-[var(--text-muted)]">(Gold Standard)</span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-6 pb-4">
                <p className="text-[var(--text-muted)] mb-2">
                  Low-dose atropine eye drops (0.01%) have been proven to slow myopia progression by 50-60% with minimal side effects. Applied once daily at bedtime.
                </p>
                <a href="#" className="text-[var(--primary-green)] text-sm font-medium flex items-center gap-1 hover:underline">
                  Ask your doctor about this <ExternalLink className="w-3 h-3" />
                </a>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="control-glasses">
              <AccordionTrigger className="px-6 py-4 hover:bg-[var(--background-mint)]">
                <div className="flex items-center gap-3">
                  <span className="text-[var(--moderate-risk)]">★</span>
                  <span className="font-bold">Myopia Control Glasses</span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-6 pb-4">
                <p className="text-[var(--text-muted)] mb-2">
                  Specially designed spectacle lenses with peripheral defocus technology. Slows progression by approximately 30%. Good option for children who can't use drops or contact lenses.
                </p>
                <a href="#" className="text-[var(--primary-green)] text-sm font-medium flex items-center gap-1 hover:underline">
                  Ask your doctor about this <ExternalLink className="w-3 h-3" />
                </a>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="ortho-k">
              <AccordionTrigger className="px-6 py-4 hover:bg-[var(--background-mint)]">
                <div className="flex items-center gap-3">
                  <div className="flex gap-0.5">
                    {[1, 2].map((i) => (
                      <span key={i} className="text-[var(--moderate-risk)]">★</span>
                    ))}
                  </div>
                  <span className="font-bold">Orthokeratology (Ortho-K)</span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-6 pb-4">
                <p className="text-[var(--text-muted)] mb-2">
                  Special rigid contact lenses worn overnight to temporarily reshape the cornea. Provides clear vision during the day without glasses and slows myopia progression by 40-50%.
                </p>
                <a href="#" className="text-[var(--primary-green)] text-sm font-medium flex items-center gap-1 hover:underline">
                  Ask your doctor about this <ExternalLink className="w-3 h-3" />
                </a>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="regular">
              <AccordionTrigger className="px-6 py-4 hover:bg-[var(--background-mint)]">
                <span className="font-bold">Regular Glasses</span>
              </AccordionTrigger>
              <AccordionContent className="px-6 pb-4">
                <p className="text-[var(--text-muted)]">
                  Traditional spectacles provide vision correction only but do not slow myopia progression. If your child already has myopia, consider upgrading to myopia control options.
                </p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </motion.div>

        {/* WHAT TO DO NEXT */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="bg-gradient-to-br from-[var(--primary-green)] to-[var(--secondary-green)] text-white rounded-3xl p-8"
        >
          <h3 className="text-2xl font-bold mb-6">What To Do Next</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center flex-shrink-0">
                1
              </div>
              <div>
                <h4 className="font-bold mb-1">Download This Report</h4>
                <p className="text-sm text-white/90">Share with your eye doctor</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center flex-shrink-0">
                2
              </div>
              <div>
                <h4 className="font-bold mb-1">Book Eye Exam</h4>
                <p className="text-sm text-white/90">
                  Find LVPEI centre near you →
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center flex-shrink-0">
                3
              </div>
              <div>
                <h4 className="font-bold mb-1">Start Lifestyle Changes</h4>
                <p className="text-sm text-white/90">Begin today, not tomorrow</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* DISCLAIMER */}
        <div className="mt-8 p-4 bg-yellow-50 border border-yellow-200 rounded-2xl text-sm text-[var(--text-dark)]">
          <strong>Important:</strong> This AI assessment is not a medical diagnosis. It provides a risk estimate based on lifestyle and family history factors. Please consult a qualified ophthalmologist for comprehensive eye examination and proper diagnosis.
        </div>
      </div>
    </div>
  );
}
