import { useState } from "react";
import { motion } from "motion/react";
import { Ruler, Info, ChevronDown } from "lucide-react";

// ── Evidence base ─────────────────────────────────────────────
// Annual untreated progression rates (D/year) by age
// Source: Donovan et al. (2012) Optom Vis Sci meta-analysis
const RATE: Record<number, number> = {
  5: 0.80, 6: 0.75, 7: 0.75, 8: 0.68, 9: 0.63,
  10: 0.56, 11: 0.49, 12: 0.41, 13: 0.31,
  14: 0.21, 15: 0.13, 16: 0.07, 17: 0.03,
};

// ── Axial Elongation constants ─────────────────────────────────
// SE → Axial Length conversion: ~0.35 mm per 1 D myopia
// Source: BHVI / Brien Holden Vision Institute methodology;
//         Brennan (2015) Optom Vis Sci; Flitcroft (2012) Ophthalmology
const SE_TO_AL  = 0.35;     // mm per diopter — base annual AL growth factor
const NORMAL_AL = 23.58;    // mm — emmetropic adult eye (Tideman et al. 2016)

// Ethnicity multipliers on annual progression rate
const ETHNICITIES = [
  { value: "asian",    label: "Asian",           mult: 1.20 },
  { value: "european", label: "European / White", mult: 0.95 },
  { value: "hispanic", label: "Hispanic / Latino",mult: 1.00 },
  { value: "african",  label: "African",          mult: 0.85 },
  { value: "other",    label: "Other",             mult: 1.00 },
];

const GENDERS = [
  { value: "female", label: "Female", mult: 1.05 },
  { value: "male",   label: "Male",   mult: 0.95 },
];

// Treatment effect 95% CI bounds (mm/year AL reduction)
// alBetter = upper 95% CI bound (more reduction, more negative = BETTER)
// alWorse  = lower 95% CI bound (less reduction, less negative = WORSE)
// Source: BHVI / IMI Myopia Management Guidelines
const TREATMENTS = [
  { value: "none",    label: "No management / Single vision", alBetter:  0,     alWorse:  0     },
  { value: "atr_low", label: "Atropine — Low dose <0.02%",    alBetter: -0.08,  alWorse: -0.04  },
  { value: "atr_mid", label: "Atropine — 0.025–0.05%",        alBetter: -0.12,  alWorse: -0.06  },
  { value: "atr_hi",  label: "Atropine — 0.1–1%",             alBetter: -0.16,  alWorse: -0.08  },
  { value: "orthok",  label: "Orthokeratology (Ortho-K)",      alBetter: -0.18,  alWorse: -0.10  },
  { value: "msc",     label: "MiSight / Soft multifocal CL",  alBetter: -0.14,  alWorse: -0.08  },
  { value: "dims",    label: "DIMS / HAL spectacles",          alBetter: -0.10,  alWorse: -0.04  },
];

// Axial Length dropdown options: 20.00 – 28.00 mm in 0.25 mm steps
const AL_OPTIONS = Array.from(
  { length: Math.round((28.00 - 20.00) / 0.25) + 1 },
  (_, i) => +(20.00 + i * 0.25).toFixed(2),
);

function getRate(age: number) {
  return RATE[Math.max(5, Math.min(17, Math.round(age)))] ?? 0.03;
}

function projectAL(
  startAge: number, startAL: number,
  ethnicMult: number, genderMult: number,
  alEffect: number,   // mm/year AL change from treatment (negative = reduction; 0 = none)
) {
  const rows: { age: number; al: number }[] = [{ age: startAge, al: startAL }];
  let al = startAL;
  for (let a = startAge + 1; a <= 18; a++) {
    const baseGrowth = getRate(a - 1) * SE_TO_AL * ethnicMult * genderMult;
    const netGrowth  = Math.max(0, baseGrowth + alEffect); // alEffect is negative
    al = Math.round((al + netGrowth) * 1000) / 1000;
    rows.push({ age: a, al });
  }
  return rows;
}

function classifyALRisk(al: number): { label: string; color: string } {
  if (al < 24.0) return { label: "Normal range",          color: "var(--low-risk)" };
  if (al < 25.5) return { label: "Elevated AL",           color: "var(--moderate-risk)" };
  if (al < 26.0) return { label: "High AL",               color: "var(--warning-coral)" };
  return               { label: "Pathological risk zone", color: "#dc2626" };
}

// ── SVG Line Chart ─────────────────────────────────────────────
interface LineSpec { values: number[]; color: string; label: string; dashed?: boolean; }
interface RefLine  { value: number; color: string; label: string; }

function LineChart({
  lines, ages, yMin, yMax, yTickFmt, refLines = [],
}: {
  lines: LineSpec[]; ages: number[];
  yMin: number; yMax: number;
  yTickFmt: (v: number) => string;
  refLines?: RefLine[];
}) {
  const W = 580, H = 220;
  const PL = 52, PR = 16, PT = 16, PB = 36;
  const cW = W - PL - PR, cH = H - PT - PB;
  const n  = ages.length;
  const xOf = (i: number) => PL + (i / (n - 1)) * cW;
  const yOf = (v: number) => PT + ((yMax - v) / (yMax - yMin)) * cH;
  const toD = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");

  const yTicks = Array.from({ length: 6 }, (_, k) => yMin + (k / 5) * (yMax - yMin));

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 220 }}>
      {yTicks.map(v => (
        <line key={v} x1={PL} y1={yOf(v)} x2={W - PR} y2={yOf(v)} stroke="#e5e7eb" strokeWidth="1" />
      ))}
      {refLines.map((r, i) => (
        <g key={i}>
          <line x1={PL} y1={yOf(r.value)} x2={W - PR} y2={yOf(r.value)}
            stroke={r.color} strokeWidth="1.5" strokeDasharray="5 4" />
          <text x={W - PR - 2} y={yOf(r.value) - 4} textAnchor="end" fontSize="9" fill={r.color}>{r.label}</text>
        </g>
      ))}
      {lines.map((ln, li) => (
        <path key={`fill-${li}`}
          d={`${toD(ln.values)} L${xOf(n - 1).toFixed(1)},${(H - PB).toFixed(1)} L${PL},${(H - PB).toFixed(1)} Z`}
          fill={ln.color} fillOpacity={ln.dashed ? 0 : 0.06} />
      ))}
      {lines.map((ln, li) => (
        <path key={`line-${li}`} d={toD(ln.values)} fill="none"
          stroke={ln.color} strokeWidth="2.5"
          strokeDasharray={ln.dashed ? "6 4" : undefined}
          strokeOpacity={ln.dashed ? 0.55 : 1} />
      ))}
      {lines.map((ln, li) => ln.values.map((v, i) => (
        <circle key={`${li}-${i}`} cx={xOf(i)} cy={yOf(v)} r="3.5"
          fill={ln.color} stroke="white" strokeWidth="1.5" />
      )))}
      {ages.map((a, i) => (
        <text key={a} x={xOf(i)} y={H - PB + 16} textAnchor="middle" fontSize="10" fill="#9ca3af">{a}</text>
      ))}
      {yTicks.map(v => (
        <text key={v} x={PL - 6} y={yOf(v) + 4} textAnchor="end" fontSize="10" fill="#9ca3af">
          {yTickFmt(v)}
        </text>
      ))}
      <line x1={PL} y1={PT} x2={PL} y2={H - PB} stroke="#d1d5db" strokeWidth="1" />
      <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke="#d1d5db" strokeWidth="1" />
      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize="10" fill="#9ca3af">Age (years)</text>
    </svg>
  );
}

// ── Styled select wrapper ──────────────────────────────────────
function StyledSelect({ label, value, onChange, children }: {
  label: string; value: string | number;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="block text-sm font-semibold text-[var(--text-dark)] mb-2">{label}</label>
      <div className="relative">
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="w-full px-3 py-2.5 pr-9 rounded-xl border border-[var(--border)] bg-[var(--background-mint)] text-[var(--text-dark)] text-sm outline-none focus:ring-2 focus:ring-[var(--secondary-green)] appearance-none"
        >
          {children}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────
export default function AxialElongation() {
  const [age,       setAge]       = useState(8);
  const [al,        setAL]        = useState(24.00);
  const [ethnicity, setEthnicity] = useState("asian");
  const [gender,    setGender]    = useState("female");
  const [treatment, setTreatment] = useState("none");
  const [ciPct,     setCiPct]     = useState(50); // 0 = WORSE, 100 = BETTER
  const [showResult, setShowResult] = useState(false);

  const tx         = TREATMENTS.find(t => t.value === treatment)!;
  const ethnicMult = ETHNICITIES.find(e => e.value === ethnicity)!.mult;
  const genderMult = GENDERS.find(g => g.value === gender)!.mult;
  const hasCI      = tx.value !== "none";

  // Linear interpolation: 0% → alWorse (less negative, WORSE), 100% → alBetter (more negative, BETTER)
  const txEffect = hasCI
    ? tx.alWorse + (tx.alBetter - tx.alWorse) * (ciPct / 100)
    : 0;

  const alUntreated = projectAL(age, al, ethnicMult, genderMult, 0);
  const alTreated   = projectAL(age, al, ethnicMult, genderMult, txEffect);
  const finalALNo   = alUntreated[alUntreated.length - 1].al;
  const finalALTx   = alTreated[alTreated.length - 1].al;
  const maxAL       = Math.max(finalALNo, NORMAL_AL + 0.5);

  return (
    <div className="min-h-screen bg-gradient-to-br from-[var(--background-mint)] to-white py-12 px-4">
      <div className="max-w-4xl mx-auto">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-3xl p-8 shadow-2xl mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-full bg-[var(--primary-green)] flex items-center justify-center flex-shrink-0">
              <Ruler className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[var(--text-dark)]">Axial Elongation Progression</h1>
              <p className="text-sm text-[var(--text-muted)]">
                Project eye axial length growth year-by-year to age 18, with and without treatment
              </p>
            </div>
          </div>

          <div className="mt-6 p-3 bg-[var(--background-mint)] rounded-xl flex gap-2 text-xs text-[var(--text-muted)]">
            <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>
              Based on BHVI / Brien Holden Vision Institute methodology · SE → AL: 0.35 mm/D ·
              Normal emmetropic eye ≈ {NORMAL_AL} mm (Tideman et al. 2016) · &gt;26 mm = elevated pathological risk ·
              Treatment CI ranges from IMI / BHVI Myopia Management Guidelines.
            </span>
          </div>

          {/* Inputs */}
          <div className="grid md:grid-cols-2 gap-6 mt-6">
            <StyledSelect label="Ethnicity" value={ethnicity}
              onChange={v => { setEthnicity(v); setShowResult(false); }}>
              {ETHNICITIES.map(e => <option key={e.value} value={e.value}>{e.label}</option>)}
            </StyledSelect>

            <StyledSelect label="Age" value={age}
              onChange={v => { setAge(+v); setShowResult(false); }}>
              {Array.from({ length: 13 }, (_, i) => i + 5).map(a => (
                <option key={a} value={a}>{a} years</option>
              ))}
            </StyledSelect>

            <StyledSelect label="Gender" value={gender}
              onChange={v => { setGender(v); setShowResult(false); }}>
              {GENDERS.map(g => <option key={g.value} value={g.value}>{g.label}</option>)}
            </StyledSelect>

            <StyledSelect label="Current Axial Length (mm)" value={al}
              onChange={v => { setAL(+v); setShowResult(false); }}>
              {AL_OPTIONS.map(v => <option key={v} value={v}>{v.toFixed(2)} mm</option>)}
            </StyledSelect>

            <div className="md:col-span-2">
              <StyledSelect label="Myopia Management Option" value={treatment}
                onChange={v => { setTreatment(v); setCiPct(50); setShowResult(false); }}>
                {TREATMENTS.map(t => <option key={t.value} value={t.value}>{t.label}</option>)}
              </StyledSelect>
            </div>
          </div>

          {/* CI Slider */}
          {hasCI && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-semibold text-[var(--text-dark)]">Treatment Effect</label>
                <span className="text-xs text-[var(--text-muted)]">
                  95% CI: {tx.alWorse.toFixed(2)} to {tx.alBetter.toFixed(2)} mm/yr
                </span>
              </div>
              <div className="relative pt-9 pb-5">
                {/* Tooltip badge above thumb */}
                <div
                  className="absolute top-0 pointer-events-none transition-[left]"
                  style={{
                    left: `${ciPct}%`,
                    transform: "translateX(-50%)",
                  }}
                >
                  <div className="bg-[#1e3a5f] text-white text-xs font-bold px-2.5 py-1 rounded-lg whitespace-nowrap text-center">
                    {txEffect.toFixed(3)} mm/yr
                  </div>
                  <div className="flex justify-center mt-0.5">
                    <div className="w-0 h-0 border-l-[5px] border-r-[5px] border-t-[5px] border-l-transparent border-r-transparent border-t-[#1e3a5f]" />
                  </div>
                </div>

                {/* WORSE / BETTER labels */}
                <div className="flex justify-between text-xs font-semibold mb-2">
                  <span style={{ color: "var(--warning-coral)" }}>WORSE</span>
                  <span style={{ color: "var(--low-risk)" }}>BETTER</span>
                </div>
                <input
                  type="range" min={0} max={100} step={1} value={ciPct}
                  onChange={e => { setCiPct(+e.target.value); setShowResult(false); }}
                  className="w-full cursor-pointer"
                  style={{ '--fill': `${ciPct}%` } as never}
                />
                <p className="text-center text-xs text-[var(--text-muted)] mt-2">Range is 95% CI Limits</p>
              </div>
            </div>
          )}

          <motion.button
            whileTap={{ scale: 0.97 }}
            onClick={() => setShowResult(true)}
            className="mt-6 w-full py-3 rounded-full bg-[var(--primary-green)] hover:bg-[var(--secondary-green)] text-white font-semibold transition-colors"
          >
            Calculate Axial Elongation →
          </motion.button>
        </motion.div>

        {/* Results */}
        {showResult && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

            {/* Summary cards */}
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-white rounded-2xl p-6 shadow-lg border-l-4" style={{ borderColor: "var(--warning-coral)" }}>
                <p className="text-xs text-[var(--text-muted)] mb-1">Without treatment — AL at age 18</p>
                <p className="text-4xl font-bold" style={{ color: "var(--warning-coral)" }}>{finalALNo.toFixed(2)} mm</p>
                <p className="text-sm mt-1 font-medium" style={{ color: classifyALRisk(finalALNo).color }}>
                  {classifyALRisk(finalALNo).label}
                </p>
                {finalALNo >= 26.0 && (
                  <p className="text-xs mt-1 font-medium" style={{ color: "#dc2626" }}>
                    ⚠ &gt;26 mm: elevated risk of retinal complications
                  </p>
                )}
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border-l-4" style={{ borderColor: "var(--primary-green)" }}>
                <p className="text-xs text-[var(--text-muted)] mb-1">With {tx.label} — AL at age 18</p>
                <p className="text-4xl font-bold" style={{ color: "var(--primary-green)" }}>{finalALTx.toFixed(2)} mm</p>
                <p className="text-sm mt-1 font-medium" style={{ color: classifyALRisk(finalALTx).color }}>
                  {classifyALRisk(finalALTx).label}
                </p>
                {hasCI && (
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    {(finalALNo - finalALTx).toFixed(2)} mm less elongation vs. no treatment
                  </p>
                )}
              </div>
            </div>

            {/* Line chart */}
            <div className="bg-white rounded-2xl p-6 shadow-lg mb-6">
              <h3 className="font-bold text-[var(--text-dark)] mb-1">Axial Length Chart (mm)</h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">
                Year-by-year axial length vs. normal emmetropic eye ({NORMAL_AL} mm)
              </p>
              <LineChart
                ages={alUntreated.map(r => r.age)}
                yMin={Math.min(al, NORMAL_AL) - 0.3}
                yMax={maxAL + 0.3}
                yTickFmt={v => `${v.toFixed(1)}`}
                lines={[
                  { values: alUntreated.map(r => r.al), color: "var(--warning-coral)", label: "No treatment", dashed: hasCI },
                  { values: alTreated.map(r => r.al),   color: "var(--primary-green)", label: tx.label },
                ]}
                refLines={[{ value: NORMAL_AL, color: "#9ca3af", label: `Normal ${NORMAL_AL} mm` }]}
              />
              <div className="flex flex-wrap gap-4 mt-3 text-xs text-[var(--text-muted)]">
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: "var(--primary-green)" }} />
                  {hasCI ? tx.label : "Current trajectory"}
                </span>
                {hasCI && (
                  <span className="flex items-center gap-1.5">
                    <span className="w-8 h-0 border-t-2 border-dashed inline-block" style={{ borderColor: "var(--warning-coral)", opacity: 0.6 }} />
                    No treatment
                  </span>
                )}
                <span className="flex items-center gap-1.5">
                  <span className="w-8 h-0 border-t-2 border-dashed inline-block border-gray-400" />
                  Normal {NORMAL_AL} mm
                </span>
              </div>

              {/* Risk zone legend */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-4">
                {[
                  { range: "< 24.0 mm",    label: "Normal",     color: "var(--low-risk)" },
                  { range: "24.0–25.5 mm", label: "Elevated",   color: "var(--moderate-risk)" },
                  { range: "25.5–26.0 mm", label: "High AL",    color: "var(--warning-coral)" },
                  { range: "> 26.0 mm",    label: "Path. risk", color: "#dc2626" },
                ].map(z => (
                  <div key={z.range} className="flex items-center gap-1.5 text-xs">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: z.color }} />
                    <span className="text-[var(--text-muted)]">{z.range} — {z.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Year-by-year AL table */}
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden mb-6">
              <div className="px-6 py-4 bg-[var(--background-mint)]">
                <h3 className="font-bold text-[var(--text-dark)]">Year-by-Year Axial Length Table</h3>
                <p className="text-xs text-[var(--text-muted)] mt-0.5">Normal emmetropic eye ≈ {NORMAL_AL} mm</p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-100">
                      <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">Age</th>
                      <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">No Treatment (AL)</th>
                      <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">{tx.label} (AL)</th>
                      <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">Risk Zone</th>
                      {hasCI && (
                        <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">AL Saved</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {alUntreated.map((row, i) => {
                      const risk  = classifyALRisk(row.al);
                      const saved = row.al - alTreated[i].al;
                      return (
                        <tr key={row.age} className="border-b border-gray-50 hover:bg-[var(--background-mint)]/50">
                          <td className="px-6 py-3 font-semibold text-[var(--text-dark)]">{row.age}</td>
                          <td className="px-6 py-3">
                            <span className="font-medium" style={{ color: risk.color }}>{row.al.toFixed(2)} mm</span>
                          </td>
                          <td className="px-6 py-3">
                            <span className="font-medium" style={{ color: "var(--primary-green)" }}>{alTreated[i].al.toFixed(2)} mm</span>
                          </td>
                          <td className="px-6 py-3">
                            <span className="text-xs font-medium px-2 py-0.5 rounded-full"
                              style={{ color: risk.color, background: `${risk.color}18` }}>
                              {risk.label}
                            </span>
                          </td>
                          {hasCI && (
                            <td className="px-6 py-3 text-[var(--text-muted)]">
                              {i > 0 ? `${saved.toFixed(3)} mm` : "—"}
                            </td>
                          )}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-2xl text-sm text-[var(--text-dark)]">
              <strong>Disclaimer:</strong> This calculator uses published population-average data. Individual outcomes vary based on genetics, environment, and treatment adherence. For personalised advice, consult a qualified ophthalmologist or optometrist.
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
