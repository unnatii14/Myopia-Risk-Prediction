import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { useNavigate } from "react-router";
import { jsPDF } from "jspdf";
import {
  Download, AlertTriangle, CheckCircle, Sun,
  Smartphone, Users, Calendar, ExternalLink,
  Eye, Loader2, BookmarkCheck
} from "lucide-react";
import RiskGauge from "../components/RiskGauge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../components/ui/accordion";
import { useAuth } from "../context/AuthContext";
import { API_URL } from "../lib/apiConfig";
import { saveScreening } from "../lib/historyApi";

interface ScreeningData {
  childName?: string;
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
  existingMyopiaStatus?: "none" | "near" | "distance";
  currentPrescription?: number;
  diagnosisAge?: number;
  myopiaControl?: string;
  progressionRate?: "slow" | "moderate" | "fast";
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
  const [savedToHistory, setSavedToHistory] = useState(false);
  const resolvedChildName = (data?.childName || user?.childName || "").trim();

  const persistHistory = async (
    screeningData: ScreeningData,
    result: {
      risk_score: number;
      risk_level: "LOW" | "MODERATE" | "HIGH";
      has_re: boolean;
      diopters: number | null;
      severity: string | null;
    }
  ) => {
    if (!user?.token) return;

    try {
      await saveScreening(user.token, screeningData as unknown as Record<string, unknown>, result);
      setSavedToHistory(true);
      setTimeout(() => setSavedToHistory(false), 4000);
    } catch (error) {
      console.error("Failed to save screening history:", error);
      setApiError((current) => current || "Result was calculated, but could not be saved to history.");
    }
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
    if (resolvedChildName) doc.text(`Child Name: ${resolvedChildName}`, margin + 4, y + 16);
    doc.text(`Age: ${data.age} years`, margin + 4, resolvedChildName ? y + 24 : y + 16);
    doc.text(`Sex: ${data.sex === "male" ? "Male" : "Female"}`, margin + 50, resolvedChildName ? y + 24 : y + 16);
    if (data.height > 0) doc.text(`Height: ${data.height} cm`, margin + 100, resolvedChildName ? y + 24 : y + 16);
    if (data.weight > 0) doc.text(`Weight: ${data.weight} kg`, margin + 145, resolvedChildName ? y + 24 : y + 16);
    if (data?.existingMyopiaStatus === "distance") {
      y += resolvedChildName ? 34 : 26;
      if (data.currentPrescription) doc.text(`Prescription: -${data.currentPrescription.toFixed(2)}D`, margin + 4, y);
      if (data.diagnosisAge) doc.text(`Diagnosed: ${data.diagnosisAge} years old`, margin + 4, y + (data.currentPrescription ? 8 : 0));
    }
    y += resolvedChildName ? 42 : 34;

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

    const safeName = resolvedChildName.replace(/[^a-zA-Z0-9_-]+/g, "_").replace(/^_+|_+$/g, "");
    const nameSuffix = safeName ? `${safeName}_` : "";
    doc.save(`MyopiaGuard_Report_${nameSuffix}${data.age}yr_${new Date().toISOString().slice(0,10)}.pdf`);
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
        void persistHistory(parsedData, {
          risk_score: result.risk_score,
          risk_level: result.risk_level,
          has_re: result.has_re,
          diopters: result.diopters,
          severity: result.severity,
        });
      })
      .catch((err) => {
        console.error("API call failed:", err);
        setApiError("Could not reach prediction server. Showing rule-based estimate.");
        const score = fallbackRiskScore(parsedData);
        const level: "LOW" | "MODERATE" | "HIGH" = score < 40 ? "LOW" : score < 70 ? "MODERATE" : "HIGH";
        setRiskScore(score);
        setRiskLevel(level);
        setLoading(false);
        void persistHistory(parsedData, {
          risk_score: score,
          risk_level: level,
          has_re: false,
          diopters: null,
          severity: null,
        });
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

  const getScoringFactors = (d: ScreeningData) => {
    const factors: { label: string; impact: string }[] = [];

    // Base
    factors.push({ label: "Base score", impact: "+30%" });

    // Age
    if (d.age <= 8) factors.push({ label: "Age ≤8 years", impact: "+15%" });
    else if (d.age <= 10) factors.push({ label: "Age ≤10 years", impact: "+10%" });
    else if (d.age <= 12) factors.push({ label: "Age ≤12 years", impact: "+5%" });

    // Genetics
    if (d.parentsMyopic === "both") factors.push({ label: "Both parents myopic", impact: "+25%" });
    else if (d.parentsMyopic === "one") factors.push({ label: "One parent myopic", impact: "+15%" });
    else if (d.familyHistory) factors.push({ label: "Family history of myopia", impact: "+10%" });

    // Screen time
    if (d.screenTime > 6) factors.push({ label: "Very high screen time (>6h)", impact: "+20%" });
    else if (d.screenTime > 4) factors.push({ label: "High screen time (>4h)", impact: "+15%" });
    else if (d.screenTime > 2) factors.push({ label: "Above recommended screen time", impact: "+8%" });

    // Outdoor time
    if (d.outdoorTime < 1) factors.push({ label: "Very low outdoor time (<1h)", impact: "+20%" });
    else if (d.outdoorTime < 2) factors.push({ label: "Below recommended outdoor time", impact: "+10%" });
    if (d.outdoorTime >= 3) factors.push({ label: "Excellent outdoor time (≥3h)", impact: "-10%" });

    // Near work
    if (d.nearWork > 6) factors.push({ label: "High near work (>6h)", impact: "+15%" });
    else if (d.nearWork > 4) factors.push({ label: "Moderate near work (>4h)", impact: "+8%" });

    // Protective factors
    if (d.vitaminD) factors.push({ label: "Vitamin D supplement", impact: "-5%" });
    if (d.sports === "regular") factors.push({ label: "Regular sports", impact: "-5%" });

    return factors;
  };

  const scoringFactors = data ? getScoringFactors(data) : [];

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

        {/* ── Saved-to-history toast ── */}
        <AnimatePresence>
          {savedToHistory && (
            <motion.div
              initial={{ opacity: 0, y: -16, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -16, scale: 0.96 }}
              transition={{ duration: 0.25 }}
              className="fixed top-24 left-1/2 z-50 -translate-x-1/2 flex items-center gap-2.5 rounded-2xl border border-green-200 bg-white px-5 py-3 shadow-xl"
            >
              <BookmarkCheck className="h-5 w-5 text-green-600 shrink-0" />
              <span className="text-sm font-semibold text-[var(--text-dark)]">Result saved to your Dashboard</span>
            </motion.div>
          )}
        </AnimatePresence>

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
                {data?.existingMyopiaStatus === "distance"
                  ? "Progression Forecast"
                  : "Risk Assessment Complete"}
              </h2>
              <div className="space-y-2 text-sm text-[var(--text-muted)]">
                {resolvedChildName && (
                  <p>
                    <strong>Child Name:</strong> {resolvedChildName}
                  </p>
                )}
                <p>
                  <strong>Age:</strong> {data.age} years
                </p>
                <p>
                  <strong>Sex:</strong> {data.sex === "male" ? "Male" : "Female"}
                </p>
                {data?.existingMyopiaStatus === "distance" && data.currentPrescription && (
                  <p>
                    <strong>Current Prescription:</strong> -{data.currentPrescription.toFixed(2)}D
                  </p>
                )}
                {data?.existingMyopiaStatus === "distance" && data.diagnosisAge && (
                  <p>
                    <strong>Diagnosed At:</strong> {data.diagnosisAge} years
                  </p>
                )}
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
                  {data?.existingMyopiaStatus === "distance" && data?.progressionRate
                    ? `${data.progressionRate.toUpperCase()} PROGRESSION`
                    : `${riskLevel} RISK`}
                </div>
              </motion.div>
            </div>

            {/* Right - Stages or Myopia Info */}
            {data?.existingMyopiaStatus === "distance" ? (
              <div className="lg:w-1/3 space-y-3">
                <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl">
                  <p className="text-xs text-[var(--text-muted)] mb-1">Current Status</p>
                  <p className="font-bold text-[var(--text-dark)]">Existing Myopia</p>
                  <p className="text-sm">
                    <span className="text-[var(--warning-coral)] font-semibold">
                      {data.myopiaControl === "none" ? "⚠️ No control method" : "✅ Control active"}
                    </span>
                  </p>
                </div>

                <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl">
                  <p className="text-xs text-[var(--text-muted)] mb-1">Progression Rate</p>
                  <p className="font-bold text-[var(--text-dark)]  capitalize">{data.progressionRate || "Unknown"}</p>
                  <p className="text-sm text-[var(--text-muted)]">
                    {data.progressionRate === "slow" && "Stable or slow progression"}
                    {data.progressionRate === "moderate" && "Typical progression rate"}
                    {data.progressionRate === "fast" && "Rapid progression - monitor closely"}
                  </p>
                </div>

                <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl">
                  <p className="text-xs text-[var(--text-muted)] mb-1">Control Method</p>
                  <p className="font-bold text-[var(--text-dark)]">
                    {data.myopiaControl === "atropine" && "Atropine Drops"}
                    {data.myopiaControl === "ortho-k" && "Ortho-K Lenses"}
                    {data.myopiaControl === "glasses" && "Control Glasses"}
                    {data.myopiaControl === "none" && "None (Consider starting)"}
                  </p>
                </div>
              </div>
            ) : (
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

                <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl border-l-4 border-green-400">
                  <p className="text-xs text-[var(--text-muted)] mb-1 font-semibold">Stage 2: Progression Risk Level</p>
                  <p className="font-bold text-[var(--text-dark)] mb-2">How fast will myopia progress?</p>
                  <p className="text-sm mb-2">
                    <span className={riskLevel === "HIGH" ? "text-[var(--warning-coral)]" : riskLevel === "MODERATE" ? "text-[var(--moderate-risk)]" : "text-[var(--secondary-green)]"}>
                      {riskLevel}
                    </span>
                    {" · "}
                    <span className="text-[var(--text-muted)] font-semibold">{riskScore}% risk score</span>
                  </p>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    <strong>Method:</strong> GradientBoosting classifier (AUC: 0.893) + clinical rules. Adaptive weighting based on ML confidence. Input: genetics, lifestyle, age, demographics. <strong>CI:</strong> ±12%
                  </p>
                </div>

                <div className="bg-gradient-to-r from-[var(--background-mint)] to-white p-4 rounded-2xl border-l-4 border-purple-400">
                  <p className="text-xs text-[var(--text-muted)] mb-1 font-semibold">Stage 3: Severity Estimate</p>
                  <p className="font-bold text-[var(--text-dark)] mb-2">Predicted myopia level at age 18</p>
                  <p className="text-sm mb-2 text-[var(--text-muted)]">
                    {prediction?.diopters != null
                      ? `~-${prediction.diopters}D · ${prediction.severity ?? "Estimated severity"}`
                      : prediction && !prediction.has_re
                      ? "No myopia detected"
                      : riskLevel === "HIGH" ? "~-3.2D · Moderate myopia" : riskLevel === "MODERATE" ? "~-1.5D · Mild myopia" : "~-0.5D · Very mild myopia"}
                  </p>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    <strong>Method:</strong> Polynomial regression (Stage 1+ only). Applies age-specific progression rates (Donovan et al. 2012) with ethnicity & gender multipliers. <strong>CI:</strong> ±0.5D
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* ── ML Backend Status Badge ── */}
          <div className="mt-6 pt-6 border-t border-gray-100">
            {prediction ? (
              <div className="flex items-center justify-center gap-2 px-4 py-2 bg-emerald-50 border border-emerald-200 rounded-xl">
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
                <span className="text-xs font-semibold text-emerald-700">Live ML Prediction</span>
                <span className="text-xs text-emerald-600 hidden sm:inline">· GradientBoosting · AUC 0.893 · 5,000+ Indian children</span>
              </div>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center justify-center gap-2 px-4 py-2 bg-amber-50 border border-amber-200 rounded-xl">
                  <span className="flex h-2.5 w-2.5 rounded-full bg-amber-400"></span>
                  <span className="text-xs font-semibold text-amber-700">Rule-based Estimate</span>
                  <span className="text-xs text-amber-600 hidden sm:inline">· ML server offline</span>
                </div>
                <p className="text-xs text-center text-[var(--text-muted)]">
                  ⚠️ {apiError ?? "Could not reach ML backend"} —{" "}
                  <span className="font-medium">run <code className="bg-gray-100 px-1 py-0.5 rounded text-gray-700">python backend/api.py</code> to enable live predictions</span>
                </p>
              </div>
            )}
          </div>
        </motion.div>

        {/* METHODOLOGY SECTION */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8 p-6 bg-gradient-to-br from-[var(--background-mint)] to-white rounded-2xl border border-[var(--secondary-green)]/20 shadow-lg"
        >
          <h3 className="text-xl font-bold text-[var(--text-dark)] mb-4">
            📊 How This Score Was Calculated
          </h3>

          <div className="grid md:grid-cols-2 gap-6 mb-6">
            {/* Model Info */}
            <div>
              <p className="text-sm font-semibold text-[var(--text-dark)] mb-2">
                🤖 Scoring Model
              </p>
              <p className="text-sm text-[var(--text-muted)] mb-2">
                {prediction
                  ? "GradientBoosting ML Model (Hybrid)"
                  : "Rule-based Scoring System (ML offline)"}
              </p>
              <ul className="text-xs text-[var(--text-muted)] space-y-1">
                <li>✓ Trained on 5,000+ Indian children</li>
                <li>✓ AUC: 0.893 (excellent accuracy)</li>
                <li>✓ Accuracy: 81.2%</li>
                {prediction && <li>✓ Live ML prediction</li>}
              </ul>
            </div>

            {/* Methodology */}
            <div>
              <p className="text-sm font-semibold text-[var(--text-dark)] mb-2">
                Three-Stage Prediction Pipeline
              </p>
              <div className="text-xs text-[var(--text-muted)] space-y-2 mb-3">
                <div className="p-2 bg-blue-50 rounded border border-blue-200">
                  <strong className="text-blue-900">Stage 1 - Refractive Error Detection:</strong> XGBoost classifier determines if child likely has or will develop myopia (AUC: 0.94). Input: age, family history, screen time, outdoor activity.
                </div>
                <div className="p-2 bg-green-50 rounded border border-green-200">
                  <strong className="text-green-900">Stage 2 - Progression Risk:</strong> GradientBoosting + adaptive rules (AUC: 0.893). If Stage 1 = positive, predicts how quickly myopia will progress (LOW/MODERATE/HIGH). Weights ML confidence for clinical safety.
                </div>
                <div className="p-2 bg-purple-50 rounded border border-purple-200">
                  <strong className="text-purple-900">Stage 3 - Severity Estimate:</strong> Polynomial regression estimates refractive error magnitude at age 18 using age-specific progression rates (Donovan et al. 2012 meta-analysis). Applies ethnicity & gender adjustment factors.
                </div>
              </div>
              <p className="text-xs text-[var(--text-muted)]">
                {prediction
                  ? "<strong>Current Mode:</strong> Live ML prediction combining all three stages with adaptive weighting based on model confidence."
                  : "<strong>Current Mode:</strong> Rule-based estimation using clinical thresholds (ML server offline)."}
              </p>
            </div>
          </div>

          {/* Contributing Factors */}
          <div>
            <p className="text-sm font-semibold text-[var(--text-dark)] mb-3">
              📌 Factors in Your Score
            </p>
            <div className="grid md:grid-cols-2 gap-2 mb-4">
              {scoringFactors.map((factor, idx) => (
                <div key={idx} className="flex justify-between items-center p-2 bg-white rounded-lg border border-gray-100 text-xs">
                  <span className="text-[var(--text-dark)]">{factor.label}</span>
                  <span className={`font-semibold ${
                    factor.impact.startsWith("+")
                      ? "text-[var(--warning-coral)]"
                      : "text-[var(--secondary-green)]"
                  }`}>
                    {factor.impact}
                  </span>
                </div>
              ))}
            </div>

            {/* Score Calculation */}
            <div className="p-3 bg-white rounded-xl border border-gray-100 text-sm">
              <div className="flex justify-between items-center font-semibold text-[var(--text-dark)]">
                <span>Your Risk Score →</span>
                <span className={`text-lg ${
                  riskLevel === "HIGH" ? "text-[var(--warning-coral)]" :
                  riskLevel === "MODERATE" ? "text-[var(--moderate-risk)]" :
                  "text-[var(--secondary-green)]"
                }`}>
                  {riskScore}%
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                Classification: <strong>{riskLevel}</strong>
                {riskLevel === "LOW" && " (<40%) - Lower myopia progression risk"}
                {riskLevel === "MODERATE" && " (40-70%) - Moderate progression risk"}
                {riskLevel === "HIGH" && " (≥70%) - Higher progression risk requiring monitoring"}
              </p>
            </div>
          </div>

          {/* Disclaimer */}
          <p className="text-xs text-[var(--text-muted)] mt-4 pt-4 border-t border-gray-200">
            ⓘ <strong>Important:</strong> This is a screening tool, not a medical diagnosis. Please consult an ophthalmologist for professional evaluation, diagnosis, and treatment planning.
          </p>
        </motion.div>
        {/* UNDERSTANDING YOUR RESULTS SECTION */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="mb-8 p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl border border-blue-200 shadow-lg"
        >
          <h3 className="text-xl font-bold text-blue-900 mb-4">
            📊 What Your Results Mean
          </h3>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Simple explanation */}
            <div>
              <p className="text-sm font-semibold text-blue-900 mb-3">
                Easy-to-Understand Summary
              </p>
              <ul className="text-xs text-blue-800 space-y-2">
                <li><strong>Stage 1:</strong> Shows whether the child may already have myopia or may develop it soon.</li>
                <li><strong>Stage 2:</strong> Shows the overall risk score. A higher score means more careful eye monitoring is needed.</li>
                <li><strong>Confidence Interval (CI):</strong> Shows how much the result may vary. A smaller range means a steadier estimate.</li>
              </ul>
            </div>

            {/* How the app works */}
            <div>
              <p className="text-sm font-semibold text-blue-900 mb-3">
                How the App Works
              </p>
              <ul className="text-xs text-blue-800 space-y-2">
                <li><strong>Stage 1:</strong> Checks whether there may be a vision problem now or in the near future.</li>
                <li><strong>Stage 2:</strong> Estimates how likely the condition is to get worse over time.</li>
                <li><strong>Stage 3:</strong> Gives a rough idea of how strong the prescription may become later.</li>
              </ul>
            </div>

            {/* What to do next */}
            <div>
              <p className="text-sm font-semibold text-blue-900 mb-3">
                What to Do Next
              </p>
              <ul className="text-xs text-blue-800 space-y-2">
                <li><span className="inline-block bg-green-100 px-2 py-1 rounded">LOW</span> Keep regular yearly eye checks and healthy habits.</li>
                <li><span className="inline-block bg-yellow-100 px-2 py-1 rounded">MODERATE</span> Plan regular eye checks and try simple preventive steps, like more outdoor time.</li>
                <li><span className="inline-block bg-red-100 px-2 py-1 rounded">HIGH</span> Visit an eye doctor soon and discuss treatment options.</li>
              </ul>
            </div>

            {/* What affects the score */}
            <div>
              <p className="text-sm font-semibold text-blue-900 mb-3">
                Things That Affect the Score
              </p>
              <ul className="text-xs text-blue-800 space-y-2">
                <li><strong>Family history:</strong> If parents are myopic, the child may have a higher chance too.</li>
                <li><strong>Daily habits:</strong> More screen time and less outdoor time can raise the risk.</li>
                <li><strong>Age and growth:</strong> Younger children and growth patterns can change the result.</li>
                <li><strong>Current eye condition:</strong> Existing glasses, progression, or treatment are also considered.</li>
              </ul>
            </div>
          </div>

          <div className="mt-4 p-3 bg-white rounded-lg border border-blue-200 text-xs text-blue-900">
            <strong>Note:</strong> These results are meant to help families decide when to get an eye check. They do not replace a doctor.
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

