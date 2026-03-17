import { useState } from "react";
import { motion } from "motion/react";
import { TrendingDown, Info, ChevronDown } from "lucide-react";

// ── Evidence base ─────────────────────────────────────────────
// Annual untreated base progression rates (D/year) by age
// Source: Donovan et al. (2012) Optom Vis Sci meta-analysis
const RATE: Record<number, number> = {
  5: 0.80, 6: 0.75, 7: 0.75, 8: 0.68, 9: 0.63,
  10: 0.56, 11: 0.49, 12: 0.41, 13: 0.31,
  14: 0.21, 15: 0.13, 16: 0.07, 17: 0.03,
};

// Ethnicity multipliers on annual progression rate
// Source: IMI White Papers / BHVI methodology
const ETHNICITIES = [
  { value: "asian",    label: "Asian",          mult: 1.20 },
  { value: "european", label: "European / White",  mult: 0.95 },
  { value: "hispanic", label: "Hispanic / Latino", mult: 1.00 },
  { value: "african",  label: "African",           mult: 0.85 },
  { value: "other",    label: "Other",              mult: 1.00 },
];

const GENDERS = [
  { value: "female", label: "Female", mult: 1.05 },
  { value: "male",   label: "Male",   mult: 0.95 },
];

// Treatment effect 95% CI bounds (D/year reduction in progression rate)
// seLow = lower 95% CI bound (WORSE end), seHigh = upper 95% CI bound (BETTER end)
// Source: BHVI / IMI Myopia Management Guidelines
const TREATMENTS = [
  { value: "none",    label: "No management / Single vision",  seLow: 0,    seHigh: 0    },
  { value: "atr_low", label: "Atropine — Low dose <0.02%",     seLow: 0.10, seHigh: 0.32 },
  { value: "atr_mid", label: "Atropine — 0.025–0.05%",         seLow: 0.15, seHigh: 0.45 },
  { value: "atr_hi",  label: "Atropine — 0.1–1%",              seLow: 0.25, seHigh: 0.55 },
  { value: "orthok",  label: "Orthokeratology (Ortho-K)",       seLow: 0.22, seHigh: 0.50 },
  { value: "msc",     label: "MiSight / Soft multifocal CL",   seLow: 0.18, seHigh: 0.44 },
  { value: "dims",    label: "DIMS / HAL spectacles",           seLow: 0.10, seHigh: 0.28 },
];

const SE_OPTIONS = [
  -0.25, -0.50, -0.75, -1.00, -1.25, -1.50, -1.75, -2.00,
  -2.25, -2.50, -2.75, -3.00, -3.50, -4.00, -4.50, -5.00,
  -5.50, -6.00, -7.00, -8.00, -9.00, -10.00,
];

function getRate(age: number) {
  return RATE[Math.max(5, Math.min(17, Math.round(age)))] ?? 0.03;
}

function project(
  startAge: number, startSE: number,
  ethnicMult: number, genderMult: number,
  txEffect: number,   // D/year treatment reduction (0 = no treatment)
) {
  const rows: { age: number; se: number }[] = [{ age: startAge, se: startSE }];
  let se = startSE;
  for (let a = startAge + 1; a <= 18; a++) {
    const baseRate = getRate(a - 1) * ethnicMult * genderMult;
    const netRate  = Math.max(0, baseRate - txEffect);
    se = Math.round((se - netRate) * 100) / 100;
    rows.push({ age: a, se });
  }
  return rows;
}

function classifySeverity(d: number) {
  const abs = Math.abs(d);
  if (abs < 0.50) return { label: "No / Trace myopia", color: "var(--low-risk)" };
  if (abs < 3.00) return { label: "Low myopia",         color: "var(--low-risk)" };
  if (abs < 6.00) return { label: "Moderate myopia",    color: "var(--moderate-risk)" };
  return             { label: "High myopia",            color: "var(--warning-coral)" };
}

// ── SVG Line Chart ─────────────────────────────────────────────
interface LineSpec { values: number[]; color: string; label: string; dashed?: boolean; }

function LineChart({
  lines, ages, yMin, yMax, yTickFmt,
}: {
  lines: LineSpec[]; ages: number[];
  yMin: number; yMax: number;
  yTickFmt: (v: number) => string;
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
export default function Progression() {
  const [age,       setAge]       = useState(8);
  const [se,        setSE]        = useState(-1.00);
  const [ethnicity, setEthnicity] = useState("asian");
  const [gender,    setGender]    = useState("female");
  const [treatment, setTreatment] = useState("none");
  const [ciPct,     setCiPct]     = useState(50); // 0 = WORSE, 100 = BETTER
  const [showResult, setShowResult] = useState(false);

  const tx         = TREATMENTS.find(t => t.value === treatment)!;
  const ethnicMult = ETHNICITIES.find(e => e.value === ethnicity)!.mult;
  const genderMult = GENDERS.find(g => g.value === gender)!.mult;
  const hasCI      = tx.value !== "none";

  // Linear interpolation: 0% → seLow (WORSE), 100% → seHigh (BETTER)
  const txEffect   = hasCI ? tx.seLow + (tx.seHigh - tx.seLow) * (ciPct / 100) : 0;

  const untreated = project(age, se, ethnicMult, genderMult, 0);
  const treated   = project(age, se, ethnicMult, genderMult, txEffect);
  const finalNo   = untreated[untreated.length - 1].se;
  const finalTx   = treated[treated.length - 1].se;

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
              <TrendingDown className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[var(--text-dark)]">Myopia Progression Calculator</h1>
              <p className="text-sm text-[var(--text-muted)]">
                Predict how myopia will progress year-by-year to age 18, with and without treatment
              </p>
            </div>
          </div>

          <div className="mt-6 p-3 bg-[var(--background-mint)] rounded-xl flex gap-2 text-xs text-[var(--text-muted)]">
            <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span>
              Based on Donovan et al. (2012) meta-analysis with ethnicity &amp; gender adjustments ·
              Treatment CI ranges from IMI / BHVI Myopia Management Guidelines.
              Individual results vary — consult an eye care professional.
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

            <StyledSelect label="Refractive Error (SE)" value={se}
              onChange={v => { setSE(+v); setShowResult(false); }}>
              {SE_OPTIONS.map(v => <option key={v} value={v}>{v.toFixed(2)} D</option>)}
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
                  95% CI: {tx.seLow.toFixed(2)}–{tx.seHigh.toFixed(2)} D/yr reduction
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
                    {txEffect.toFixed(2)} D/yr
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
            Calculate Progression →
          </motion.button>
        </motion.div>

        {/* Results */}
        {showResult && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

            {/* Summary cards */}
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-white rounded-2xl p-6 shadow-lg border-l-4" style={{ borderColor: "var(--warning-coral)" }}>
                <p className="text-xs text-[var(--text-muted)] mb-1">Without treatment — at age 18</p>
                <p className="text-4xl font-bold" style={{ color: "var(--warning-coral)" }}>{finalNo.toFixed(2)} D</p>
                <p className="text-sm mt-1 font-medium" style={{ color: classifySeverity(finalNo).color }}>
                  {classifySeverity(finalNo).label}
                </p>
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border-l-4" style={{ borderColor: "var(--primary-green)" }}>
                <p className="text-xs text-[var(--text-muted)] mb-1">With {tx.label} — at age 18</p>
                <p className="text-4xl font-bold" style={{ color: "var(--primary-green)" }}>{finalTx.toFixed(2)} D</p>
                <p className="text-sm mt-1 font-medium" style={{ color: classifySeverity(finalTx).color }}>
                  {classifySeverity(finalTx).label}
                </p>
                {hasCI && (
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    {Math.abs(finalNo - finalTx).toFixed(2)} D saved vs. no treatment
                  </p>
                )}
              </div>
            </div>

            {/* Line chart */}
            <div className="bg-white rounded-2xl p-6 shadow-lg mb-6">
              <h3 className="font-bold text-[var(--text-dark)] mb-1">Progression Chart</h3>
              <p className="text-xs text-[var(--text-muted)] mb-4">Spherical equivalent (D) year-by-year to age 18</p>
              <LineChart
                ages={untreated.map(r => r.age)}
                yMin={Math.min(...untreated.map(r => r.se)) - 0.3}
                yMax={0}
                yTickFmt={v => `${v.toFixed(1)} D`}
                lines={[
                  { values: untreated.map(r => r.se), color: "var(--warning-coral)", label: "No treatment", dashed: hasCI },
                  { values: treated.map(r => r.se),   color: "var(--primary-green)",  label: tx.label },
                ]}
              />
              <div className="flex gap-5 mt-3 text-xs text-[var(--text-muted)]">
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
              </div>
            </div>

            {/* Year-by-year table */}
            <div className="bg-white rounded-2xl shadow-lg overflow-hidden mb-6">
              <div className="px-6 py-4 bg-[var(--background-mint)]">
                <h3 className="font-bold text-[var(--text-dark)]">Year-by-Year Table</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-100">
                      <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">Age</th>
                      <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">No Treatment</th>
                      <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">{tx.label}</th>
                      {hasCI && (
                        <th className="px-6 py-3 text-left text-[var(--text-muted)] font-medium">D Saved</th>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {untreated.map((row, i) => {
                      const sv    = classifySeverity(row.se);
                      const saved = Math.abs(treated[i].se - row.se);
                      return (
                        <tr key={row.age} className="border-b border-gray-50 hover:bg-[var(--background-mint)]/50">
                          <td className="px-6 py-3 font-semibold text-[var(--text-dark)]">{row.age}</td>
                          <td className="px-6 py-3">
                            <span className="font-medium" style={{ color: sv.color }}>{row.se.toFixed(2)} D</span>
                          </td>
                          <td className="px-6 py-3">
                            <span className="font-medium" style={{ color: "var(--primary-green)" }}>{treated[i].se.toFixed(2)} D</span>
                          </td>
                          {hasCI && (
                            <td className="px-6 py-3 text-[var(--text-muted)]">
                              {i > 0 ? `+${saved.toFixed(2)} D` : "—"}
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
              <strong>Disclaimer:</strong> This calculator uses published population-average progression rates adjusted for ethnicity and gender. Individual outcomes vary based on genetics, environment, and adherence to treatment. For personalised advice, consult a qualified ophthalmologist or optometrist.
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
