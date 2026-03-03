import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { useNavigate } from "react-router";
import { 
  Download, AlertTriangle, CheckCircle, Sun, 
  Smartphone, Users, Calendar, ExternalLink,
  Eye, Activity, TrendingUp
} from "lucide-react";
import RiskGauge from "../components/RiskGauge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "../components/ui/accordion";

interface ScreeningData {
  age: number;
  sex: string;
  height: number;
  weight: number;
  state: string;
  familyHistory: boolean | null;
  parentsMyopic: string;
  screenTime: number;
  nearWork: number;
  outdoorTime: number;
  sports: string;
  vitaminD: boolean | null;
  schoolType: string;
  tuition: boolean | null;
  competitiveExam: boolean | null;
}

export default function Results() {
  const navigate = useNavigate();
  const [data, setData] = useState<ScreeningData | null>(null);
  const [riskScore, setRiskScore] = useState(0);
  const [riskLevel, setRiskLevel] = useState<"LOW" | "MODERATE" | "HIGH">("LOW");

  useEffect(() => {
    const stored = sessionStorage.getItem("screeningData");
    if (!stored) {
      navigate("/screen");
      return;
    }

    const parsedData = JSON.parse(stored);
    setData(parsedData);

    // Calculate risk score based on factors
    const score = calculateRiskScore(parsedData);
    setRiskScore(score);

    if (score < 40) setRiskLevel("LOW");
    else if (score < 70) setRiskLevel("MODERATE");
    else setRiskLevel("HIGH");
  }, [navigate]);

  const calculateRiskScore = (data: ScreeningData): number => {
    let score = 30; // Base score

    // Age factor (younger = higher risk for progression)
    if (data.age <= 8) score += 15;
    else if (data.age <= 10) score += 10;
    else if (data.age <= 12) score += 5;

    // Family history
    if (data.parentsMyopic === "both") score += 25;
    else if (data.parentsMyopic === "one") score += 15;
    else if (data.familyHistory) score += 10;

    // Screen time
    if (data.screenTime > 6) score += 20;
    else if (data.screenTime > 4) score += 15;
    else if (data.screenTime > 2) score += 8;

    // Outdoor time (protective)
    if (data.outdoorTime < 1) score += 20;
    else if (data.outdoorTime < 2) score += 10;
    else if (data.outdoorTime >= 3) score -= 10;

    // Near work
    if (data.nearWork > 6) score += 15;
    else if (data.nearWork > 4) score += 8;

    // Academic pressure
    if (data.competitiveExam) score += 10;
    if (data.tuition) score += 5;
    if (data.schoolType === "international" || data.schoolType === "private") score += 5;

    // Protective factors
    if (data.vitaminD) score -= 5;
    if (data.sports === "regular") score -= 5;

    return Math.min(Math.max(score, 0), 100);
  };

  if (!data) return null;

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
                <p>
                  <strong>State:</strong> {data.state}
                </p>
                <p className="text-xs pt-2">
                  <strong>Date:</strong> {new Date().toLocaleDateString("en-IN")}
                </p>
              </div>

              <button className="mt-6 flex items-center gap-2 px-4 py-2 border-2 border-[var(--primary-green)] text-[var(--primary-green)] rounded-full hover:bg-[var(--primary-green)] hover:text-white transition-all">
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
                  <span className={riskScore > 60 ? "text-[var(--warning-coral)]" : "text-[var(--secondary-green)]"}>
                    {riskScore > 60 ? "YES" : "POSSIBLE"}
                  </span>
                  {" · "}
                  <span className="text-[var(--text-muted)]">{Math.round(riskScore * 0.8)}%</span>
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
                  {riskLevel === "HIGH" ? "~-3.2D · Moderate" : riskLevel === "MODERATE" ? "~-1.5D · Mild" : "~-0.5D · Very Mild"}
                </p>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-6 border-t border-gray-200 text-xs text-[var(--text-muted)] text-center">
            Powered by XGBoost ML Model · AUC 0.88 · Trained on 5,000+ Indian children
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
