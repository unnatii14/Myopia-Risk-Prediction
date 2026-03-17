import { useState } from "react";
import { motion } from "motion/react";
import { Clock, AlertTriangle, CheckCircle, ShieldAlert, Info } from "lucide-react";

// ── Evidence base ─────────────────────────────────────────────
// Age-specific hyperopic reserve norms (cycloplegic SE, diopters)
// Source: Zadnik et al. / CLEERE Study, Invest Ophthalmol Vis Sci
const NORM: Record<number, number> = {
  5: 1.38, 6: 1.16, 7: 0.90, 8: 0.68, 9: 0.46,
  10: 0.24, 11: 0.02, 12: -0.21, 13: -0.44, 14: -0.67,
};

// Pre-myopia drift rate: average annual change toward myopia before onset
const DRIFT = 0.25; // D / year

type RiskLevel = "myopic" | "high" | "moderate" | "low";

interface Result {
  level: RiskLevel;
  probability: number;
  onsetAge: number | null;
  yearsToOnset: number | null;
  message: string;
}

function analyze(age: number, se: number, parents: string): Result {
  const norm = NORM[Math.max(5, Math.min(14, age))] ?? -0.67;
  const parentMult = parents === "both" ? 1.40 : parents === "one" ? 1.18 : 1.0;

  // Already myopic
  if (se <= -0.50) {
    return {
      level: "myopic", probability: 100, onsetAge: null, yearsToOnset: null,
      message: "Myopia is already present. Focus on slowing progression.",
    };
  }

  // Years until SE reaches −0.50 D at current drift rate
  const yearsRaw     = Math.max(0.5, (se - (-0.50)) / DRIFT);
  const predictedAge = Math.round(Math.min(20, age + yearsRaw));

  // Deficit vs age norm (positive = child is below age expectation = higher risk)
  const deficit = norm - se;

  let level: RiskLevel;
  let baseProb: number;
  let message: string;

  if (se <= 0.50 || deficit >= 0.75) {
    level    = "high";
    baseProb = 78;
    message  = "High likelihood of myopia onset within 1–3 years. Start preventive measures now.";
  } else if (deficit >= 0.30) {
    level    = "moderate";
    baseProb = 48;
    message  = "Moderate risk. Below expected hyperopic reserve for this age. Monitor every 6 months.";
  } else {
    level    = "low";
    baseProb = 16;
    message  = "Low risk at this time. Maintain healthy habits and annual eye exams.";
  }

  const probability = Math.min(94, Math.round(baseProb * parentMult));

  return {
    level, probability,
    onsetAge:     predictedAge,
    yearsToOnset: Math.round(yearsRaw * 10) / 10,
    message,
  };
}

const LEVEL_CONFIG = {
  myopic:   { label: "Already Myopic",  bg: "bg-red-50",                   border: "border-red-300",          text: "var(--warning-coral)", Icon: AlertTriangle },
  high:     { label: "High Risk",       bg: "bg-orange-50",                border: "border-orange-300",       text: "var(--warning-coral)", Icon: ShieldAlert },
  moderate: { label: "Moderate Risk",  bg: "bg-amber-50",                 border: "border-amber-300",        text: "var(--moderate-risk)", Icon: AlertTriangle },
  low:      { label: "Low Risk",        bg: "bg-[var(--background-mint)]", border: "border-[var(--border)]",  text: "var(--low-risk)",      Icon: CheckCircle },
};

const RECOMMENDATIONS: Record<RiskLevel, string[]> = {
  myopic: [
    "Discuss myopia control options with your doctor (atropine drops, ortho-K, or control lenses).",
    "Aim for ≥2 hours of outdoor time daily — this is the single strongest protective factor.",
    "Limit recreational screen time to <2 hours/day and apply the 20-20-20 rule.",
    "Schedule eye exams every 3–6 months to track progression.",
  ],
  high: [
    "Book a cycloplegic refraction with an eye care professional immediately.",
    "Increase outdoor time to at least 2 hours per day in natural daylight.",
    "Limit near-work screen time and take regular breaks (20-20-20 rule).",
    "Discuss pre-myopia interventions with your doctor (low-dose atropine is being studied for this).",
    "With both parents myopic, genetics increase risk significantly — monitor closely.",
  ],
  moderate: [
    "Schedule a comprehensive eye exam including cycloplegic refraction.",
    "Prioritise 2+ hours of outdoor time daily — proven to delay myopia onset.",
    "Reduce prolonged near-work sessions; maintain proper reading distance (>30 cm).",
    "Re-check annually, or every 6 months if refraction is close to the pre-myopia zone.",
  ],
  low: [
    "Continue annual eye examinations to track refractive development.",
    "Maintain outdoor habits (2+ hrs/day) as a lifelong protective factor.",
    "Keep screen time and near-work within recommended limits.",
  ],
};

export default function OnsetPredictor() {
  const [age,     setAge]     = useState(8);
  const [se,      setSE]      = useState(0.75);
  const [parents, setParents] = useState("none");
  const [result,  setResult]  = useState<Result | null>(null);

  const handleCalc = () => setResult(analyze(age, se, parents));

  const cfg = result ? LEVEL_CONFIG[result.level] : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-[var(--background-mint)] to-white py-12 px-4">
      <div className="max-w-4xl mx-auto">

        {/* ── Header card ──────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-3xl p-8 shadow-2xl mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-full bg-[var(--primary-green)] flex items-center justify-center flex-shrink-0">
              <Clock className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[var(--text-dark)]">Myopia Onset Predictor</h1>
              <p className="text-sm text-[var(--text-muted)]">
                Estimate the likelihood and age of myopia onset based on current refraction and family history
              </p>
            </div>
          </div>

          <div className="mt-6 p-3 bg-[var(--background-mint)] rounded-xl flex gap-2 text-xs text-[var(--text-muted)]">
            <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>Based on CLEERE Study norms (Zadnik et al.) and IMI pre-myopia guidelines. Use cycloplegic refraction values for best accuracy.</span>
          </div>

          {/* ── Inputs ─────────────────────────────────────────── */}
          <div className="grid md:grid-cols-3 gap-6 mt-6">
            {/* Age */}
            <div>
              <label className="block text-sm font-semibold text-[var(--text-dark)] mb-2">
                Current Age: <span className="text-[var(--primary-green)]">{age} yrs</span>
              </label>
              <input
                type="range" min={5} max={14} step={1} value={age}
                onChange={e => { setAge(+e.target.value); setResult(null); }}
                className="w-full h-2 rounded-full appearance-none cursor-pointer"
                style={{ accentColor: "var(--primary-green)" }}
              />
              <div className="flex justify-between text-xs text-[var(--text-muted)] mt-1">
                <span>5</span><span>14</span>
              </div>
            </div>

            {/* Current refraction */}
            <div>
              <label className="block text-sm font-semibold text-[var(--text-dark)] mb-2">
                Current Refraction (SE)
              </label>
              <select
                value={se}
                onChange={e => { setSE(+e.target.value); setResult(null); }}
                className="w-full px-3 py-2.5 rounded-xl border border-[var(--border)] bg-[var(--background-mint)] text-[var(--text-dark)] text-sm outline-none focus:ring-2 focus:ring-[var(--secondary-green)]"
              >
                {[3.00,2.75,2.50,2.25,2.00,1.75,1.50,1.25,1.00,0.75,0.50,0.25,0.00,-0.25,-0.49].map(v => (
                  <option key={v} value={v}>
                    {v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2)} D
                    {v >= 0.50 && v <= 1.50 ? " (hyperopic)" : ""}
                    {v > -0.50 && v < 0.50 ? " (pre-myopia zone)" : ""}
                    {v <= -0.50 ? " (myopic)" : ""}
                  </option>
                ))}
              </select>
              <p className="text-xs text-[var(--text-muted)] mt-1">Use cycloplegic values if available</p>
            </div>

            {/* Parental myopia */}
            <div>
              <label className="block text-sm font-semibold text-[var(--text-dark)] mb-3">Parental Myopia</label>
              <div className="space-y-2">
                {[
                  { value: "none", label: "Neither parent" },
                  { value: "one",  label: "One parent myopic" },
                  { value: "both", label: "Both parents myopic" },
                ].map(opt => (
                  <label key={opt.value} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio" name="parents" value={opt.value}
                      checked={parents === opt.value}
                      onChange={() => { setParents(opt.value); setResult(null); }}
                      className="accent-[var(--primary-green)]"
                    />
                    <span className="text-sm text-[var(--text-dark)]">{opt.label}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>

          {/* Age norm reference */}
          <div className="mt-4 p-3 rounded-xl border border-[var(--border)] text-xs text-[var(--text-muted)]">
            Expected refraction for age {age}:{" "}
            <strong className="text-[var(--primary-green)]">
              +{(NORM[Math.max(5, Math.min(14, age))] ?? 0).toFixed(2)} D
            </strong>{" "}
            (population norm) · Your input:{" "}
            <strong className={se < (NORM[Math.max(5, Math.min(14, age))] ?? 0) - 0.30 ? "text-[var(--warning-coral)]" : "text-[var(--primary-green)]"}>
              {se >= 0 ? "+" : ""}{se.toFixed(2)} D
            </strong>
          </div>

          <motion.button
            whileTap={{ scale: 0.97 }}
            onClick={handleCalc}
            className="mt-6 w-full py-3 rounded-full bg-[var(--primary-green)] hover:bg-[var(--secondary-green)] text-white font-semibold transition-colors"
          >
            Predict Onset Risk →
          </motion.button>
        </motion.div>

        {/* ── Results ──────────────────────────────────────────── */}
        {result && cfg && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

            {/* Risk badge */}
            <div className={`bg-white rounded-3xl p-8 shadow-2xl mb-6`}>
              <div className="flex flex-col md:flex-row gap-6 items-start md:items-center">
                {/* Icon + level */}
                <div className={`flex-shrink-0 flex flex-col items-center justify-center w-40 h-40 rounded-full border-4 ${cfg.bg} ${cfg.border}`}>
                  <cfg.Icon className="w-10 h-10 mb-2" style={{ color: cfg.text }} />
                  <p className="text-xs font-bold text-center" style={{ color: cfg.text }}>{cfg.label}</p>
                  {result.level !== "myopic" && (
                    <p className="text-2xl font-black mt-1" style={{ color: cfg.text }}>{result.probability}%</p>
                  )}
                </div>

                {/* Key numbers */}
                <div className="flex-1 space-y-4">
                  <p className="text-[var(--text-dark)] font-medium">{result.message}</p>

                  {result.level !== "myopic" && result.onsetAge !== null && (
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-[var(--background-mint)] rounded-2xl p-4">
                        <p className="text-xs text-[var(--text-muted)] mb-1">Predicted onset age</p>
                        <p className="text-3xl font-bold text-[var(--text-dark)]">{result.onsetAge}</p>
                        <p className="text-xs text-[var(--text-muted)]">years old</p>
                      </div>
                      <div className="bg-[var(--background-mint)] rounded-2xl p-4">
                        <p className="text-xs text-[var(--text-muted)] mb-1">Years until likely onset</p>
                        <p className="text-3xl font-bold text-[var(--text-dark)]">{result.yearsToOnset}</p>
                        <p className="text-xs text-[var(--text-muted)]">years (est.)</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Hyperopic reserve comparison */}
            {result.level !== "myopic" && (
              <div className="bg-white rounded-2xl p-6 shadow-lg mb-6">
                <h3 className="font-bold text-[var(--text-dark)] mb-4">Hyperopic Reserve vs Age Norm</h3>
                <div className="space-y-3">
                  {Object.entries(NORM)
                    .filter(([a]) => +a >= Math.max(5, age - 1) && +a <= Math.min(14, age + 5))
                    .map(([a, norm]) => {
                      const isCurrentAge = +a === age;
                      const maxVal = 3.0;
                      const normPct   = (norm / maxVal) * 100;
                      const childPct  = isCurrentAge ? (Math.max(0, se) / maxVal) * 100 : 0;
                      return (
                        <div key={a} className={`flex items-center gap-3 ${isCurrentAge ? "font-bold" : ""}`}>
                          <span className="text-xs text-[var(--text-muted)] w-8 text-right flex-shrink-0">
                            {a}{isCurrentAge ? "★" : ""}
                          </span>
                          <div className="flex-1 relative h-5 rounded-full bg-gray-100 overflow-hidden">
                            <div
                              className="absolute h-full rounded-full opacity-20 transition-all"
                              style={{ width: `${Math.max(0, normPct)}%`, backgroundColor: "var(--secondary-green)" }}
                            />
                            {isCurrentAge && (
                              <div
                                className="absolute h-full rounded-full transition-all"
                                style={{ width: `${Math.max(0, childPct)}%`, backgroundColor: se > norm ? "var(--primary-green)" : "var(--warning-coral)" }}
                              />
                            )}
                          </div>
                          <span className="text-xs w-24 flex-shrink-0 text-[var(--text-muted)]">
                            Norm: +{norm.toFixed(2)} D
                            {isCurrentAge ? ` · Child: ${se >= 0 ? "+" : ""}${se.toFixed(2)} D` : ""}
                          </span>
                        </div>
                      );
                    })}
                </div>
                <div className="flex gap-4 mt-3 text-xs text-[var(--text-muted)]">
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded-full inline-block opacity-20 bg-[var(--secondary-green)]" />
                    Age norm
                  </span>
                  <span className="flex items-center gap-1.5">★ = current age</span>
                </div>
              </div>
            )}

            {/* Recommendations */}
            <div className="bg-gradient-to-br from-[var(--secondary-green)]/10 to-white rounded-2xl p-6 border border-[var(--secondary-green)]/30 mb-6">
              <h3 className="font-bold text-[var(--text-dark)] mb-4">Recommended Actions</h3>
              <ul className="space-y-3">
                {RECOMMENDATIONS[result.level].map((rec, i) => (
                  <motion.li
                    key={i}
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.08 }}
                    className="flex gap-3 items-start"
                  >
                    <CheckCircle className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: "var(--primary-green)" }} />
                    <span className="text-sm text-[var(--text-dark)]">{rec}</span>
                  </motion.li>
                ))}
              </ul>
            </div>

            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-2xl text-sm text-[var(--text-dark)]">
              <strong>Disclaimer:</strong> This tool provides a population-based risk estimate using published research norms. It is not a substitute for a clinical eye examination. Cycloplegic refraction by a qualified optometrist or ophthalmologist is the gold standard for assessing a child's refractive status.
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
