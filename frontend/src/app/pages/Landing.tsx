import { motion } from "motion/react";
import { useNavigate } from "react-router";
import { Sparkles, BarChart3, FileText, Sun, Dna, Glasses, Shield, Star, Smartphone, Eye } from "lucide-react";
import AnimatedCounter from "../components/AnimatedCounter";
import BlinkingEye from "../components/BlinkingEye";
import BokehBackground from "../components/BokehBackground";

// Orbit cards that circle the eye
const ORBIT_CARDS = [
  { label: "1 in 3", sub: "Indian children", color: "var(--primary-green)", angle: 0 },
  { label: "↑ 40%", sub: "Rise since 2010", color: "var(--warning-coral)", angle: 90 },
  { label: "AUC 0.88", sub: "Model accuracy", color: "var(--accent-blue)", angle: 180 },
  { label: "2 hrs", sub: "Outdoor target", color: "var(--secondary-green)", angle: 270 },
];

function OrbitCard({ label, sub, color, angle, radius = 220 }: { label: string; sub: string; color: string; angle: number; radius?: number }) {
  const rad = (angle * Math.PI) / 180;
  const x = Math.cos(rad) * radius;
  const y = Math.sin(rad) * radius;
  return (
    <motion.div
      className="absolute"
      style={{
        left: "50%",
        top: "50%",
        x: x - 56,
        y: y - 28,
      }}
      animate={{
        x: [x - 56, x - 56 + Math.cos(rad + 0.3) * 8, x - 56],
        y: [y - 28, y - 28 + Math.sin(rad + 0.3) * 8, y - 28],
      }}
      transition={{ duration: 4 + angle / 90, repeat: Infinity, ease: "easeInOut" }}
    >
      <div className="bg-white/95 backdrop-blur-sm rounded-2xl px-4 py-2.5 shadow-xl border border-white/60 w-28 text-center">
        <p className="text-sm font-bold leading-none mb-0.5" style={{ color }}>{label}</p>
        <p className="text-[10px] text-[var(--text-muted)] leading-tight">{sub}</p>
      </div>
    </motion.div>
  );
}

export default function Landing() {
  const navigate = useNavigate();

  return (
    <div className="w-full">
      {/* HERO SECTION */}
      <section className="relative min-h-[90vh] flex items-center overflow-hidden">
        <BokehBackground />

        {/* Subtle grid overlay */}
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.03]"
          style={{
            backgroundImage: "linear-gradient(var(--primary-green) 1px, transparent 1px), linear-gradient(90deg, var(--primary-green) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 relative z-10 w-full">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8 }}
            >
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full mb-6 border border-[var(--secondary-green)] shadow-sm"
              >
                <Sparkles className="w-4 h-4 text-[var(--primary-green)]" />
                <span className="text-sm font-medium text-[var(--text-dark)]">
                  AI-Powered · Free · For Indian Families
                </span>
              </motion.div>

              <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-[var(--text-dark)] mb-6 leading-tight">
                Is Your Child{" "}
                <span className="relative inline-block">
                  <span className="relative z-10" style={{ color: "var(--primary-green)" }}>at Risk</span>
                  <motion.span
                    className="absolute bottom-1 left-0 right-0 h-3 rounded-full -z-0 opacity-30"
                    style={{ background: "var(--secondary-green)" }}
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ delay: 0.9, duration: 0.6, ease: "easeOut" }}
                  />
                </span>{" "}
                of Myopia?
              </h1>

              <p className="text-xl text-[var(--text-muted)] mb-8 leading-relaxed">
                Answer 12 questions. Get an instant AI risk score backed by LVPEI research.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 mb-6">
                <div className="relative inline-flex">
                  <motion.div
                    className="absolute inset-0 rounded-full bg-[var(--primary-green)]"
                    animate={{ scale: [1, 1.22, 1], opacity: [0.45, 0, 0.45] }}
                    transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
                  />
                  <button
                    onClick={() => navigate("/screen")}
                    className="relative px-8 py-4 bg-[var(--primary-green)] text-white rounded-full hover:bg-[var(--secondary-green)] transition-all font-semibold text-lg shadow-lg hover:shadow-xl transform hover:scale-105"
                  >
                    Start Free Screening →
                  </button>
                </div>
              </div>

              <p className="text-sm text-[var(--text-muted)] flex items-center gap-2">
                <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" />
                </svg>
                Takes less than 3 minutes
              </p>
            </motion.div>

            {/* Right — Eye with orbiting data cards */}
            <motion.div
              initial={{ opacity: 0, scale: 0.85 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.9, delay: 0.2 }}
              className="relative flex items-center justify-center"
              style={{ minHeight: 420 }}
            >
              {/* Outer slow rotation ring behind eye */}
              <motion.div
                className="absolute rounded-full border border-dashed border-[var(--secondary-green)]/25"
                style={{ width: 460, height: 460 }}
                animate={{ rotate: 360 }}
                transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
              />
              <motion.div
                className="absolute rounded-full border border-[var(--accent-blue)]/15"
                style={{ width: 380, height: 380 }}
                animate={{ rotate: -360 }}
                transition={{ duration: 40, repeat: Infinity, ease: "linear" }}
              />

              {/* Orbiting stat cards */}
              <div className="absolute" style={{ width: 0, height: 0 }}>
                {ORBIT_CARDS.map((card) => (
                  <OrbitCard key={card.angle} {...card} radius={210} />
                ))}
              </div>

              {/* The eye itself */}
              <div className="relative z-10 w-72">
                <BlinkingEye />
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* STATS BAR */}
      <section className="bg-[var(--primary-green)] text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
            <div>
              <div className="text-4xl font-bold mb-2">
                <AnimatedCounter end={2.6} decimals={1} suffix="B" />
              </div>
              <p className="text-white/90">People affected by 2050</p>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">
                <AnimatedCounter end={40} suffix="%" />
              </div>
              <p className="text-white/90">Increase in Indian children past decade</p>
            </div>
            <div>
              <div className="text-4xl font-bold mb-2">
                <AnimatedCounter end={60} suffix="%" />
              </div>
              <p className="text-white/90">Slower progression with early treatment</p>
            </div>
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}
      <section id="how-it-works" className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-[var(--text-dark)] mb-4">
              How It Works
            </h2>
            <p className="text-xl text-[var(--text-muted)] max-w-2xl mx-auto">
              Simple, fast, and scientifically validated
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8 relative">
            {/* Connecting line */}
            <div className="hidden md:block absolute top-24 left-0 right-0 h-0.5 bg-gradient-to-r from-[var(--secondary-green)] via-[var(--accent-blue)] to-[var(--secondary-green)] opacity-30" />

            {[
              {
                icon: FileText,
                title: "Answer 12 Questions",
                description: "Simple lifestyle and family history questions about your child",
                step: 1,
              },
              {
                icon: BarChart3,
                title: "AI Analyses Risk",
                description: "Our model processes 35+ risk factors using advanced XGBoost algorithm",
                step: 2,
              },
              {
                icon: Sparkles,
                title: "Get Personalized Report",
                description: "Instant risk score with actionable recommendations",
                step: 3,
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                className="relative"
              >
                <div className="bg-white rounded-3xl p-8 shadow-lg hover:shadow-xl transition-shadow border border-[var(--border)] h-full">
                  <div className="w-16 h-16 bg-gradient-to-br from-[var(--secondary-green)] to-[var(--accent-blue)] rounded-2xl flex items-center justify-center mb-6 relative z-10">
                    <item.icon className="w-8 h-8 text-white" />
                  </div>
                  <div className="absolute top-6 right-6 text-6xl font-bold text-[var(--secondary-green)]/10">
                    {item.step}
                  </div>
                  <h3 className="text-2xl font-bold text-[var(--text-dark)] mb-4">
                    {item.title}
                  </h3>
                  <p className="text-[var(--text-muted)] leading-relaxed">
                    {item.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* RISK FACTORS EDUCATION */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-[var(--text-dark)] mb-4">
              What Puts Children at Risk?
            </h2>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: Smartphone,
                title: "Excessive Screen Time",
                description: "More than 2 hours daily increases risk significantly",
                color: "var(--warning-coral)",
              },
              {
                icon: Sun,
                title: "Not Enough Outdoor Time",
                description: "Less than 2 hours outdoor time daily is a major risk factor",
                color: "var(--moderate-risk)",
              },
              {
                icon: Dna,
                title: "Family History",
                description: "Both parents with myopia increases child's risk 6x",
                color: "var(--warning-coral)",
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="bg-gradient-to-br from-white to-[var(--background-mint)] p-6 rounded-2xl border-2 border-[var(--border)] hover:border-[var(--secondary-green)] transition-colors"
              >
                <div
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                  style={{ backgroundColor: `${item.color}20` }}
                >
                  <item.icon className="w-6 h-6" style={{ color: item.color }} />
                </div>
                <h3 className="text-xl font-bold text-[var(--text-dark)] mb-2">
                  {item.title}
                </h3>
                <p className="text-[var(--text-muted)]">{item.description}</p>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mt-16"
          >
            <h3 className="text-3xl font-bold text-[var(--text-dark)] mb-8 text-center">
              What Protects Children?
            </h3>
            <div className="grid md:grid-cols-4 gap-6">
              {[
                { icon: Sun, text: "2+ hours outdoor time daily" },
                { icon: Shield, text: "Vitamin D supplementation" },
                { icon: Eye, text: "Regular eye exams" },
                { icon: Glasses, text: "Myopia control lenses" },
              ].map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className="bg-[var(--secondary-green)]/10 p-6 rounded-2xl text-center border border-[var(--secondary-green)]/30"
                >
                  <item.icon className="w-10 h-10 text-[var(--secondary-green)] mx-auto mb-3" />
                  <p className="font-medium text-[var(--text-dark)]">{item.text}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* CORRECTION METHODS */}
      <section className="py-20 bg-gradient-to-br from-[var(--background-mint)] to-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-[var(--text-dark)] mb-4">
              Treatment Options Available
            </h2>
            <p className="text-xl text-[var(--text-muted)] max-w-2xl mx-auto">
              Multiple proven methods to slow myopia progression
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                title: "Regular Glasses",
                description: "Vision correction only",
                effectiveness: 0,
                border: "gray",
              },
              {
                title: "Myopia Control Glasses",
                description: "~30% slower progression",
                effectiveness: 1,
                border: "var(--secondary-green)",
              },
              {
                title: "Atropine 0.01% Drops",
                description: "50-60% slower · Gold standard",
                effectiveness: 2,
                border: "var(--primary-green)",
              },
              {
                title: "Ortho-K Lenses",
                description: "Night-wear · Strong control",
                effectiveness: 2,
                border: "var(--primary-green)",
              },
            ].map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-shadow relative overflow-hidden"
                style={{
                  borderTop: `4px solid ${item.border}`,
                }}
              >
                <div className="flex gap-1 mb-3">
                  {Array.from({ length: item.effectiveness }).map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-[var(--moderate-risk)] text-[var(--moderate-risk)]" />
                  ))}
                </div>
                <h3 className="text-lg font-bold text-[var(--text-dark)] mb-2">
                  {item.title}
                </h3>
                <p className="text-sm text-[var(--text-muted)]">{item.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* RESEARCH BACKING */}
      <section id="research" className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center"
          >
            <p className="text-[var(--text-muted)] mb-8 font-medium text-lg">
              Backed by peer-reviewed research from leading eye institutions
            </p>
            <div className="flex flex-wrap justify-center items-center gap-4">
              {[
                { name: "LVPEI", sub: "L V Prasad Eye Institute" },
                { name: "PREMo", sub: "Pre-Myopia Risk Score" },
                { name: "Nature", sub: "Scientific Reports" },
                { name: "BMJ", sub: "Open Ophthalmology" },
              ].map((org, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 10 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.1 }}
                  className="flex flex-col items-center px-8 py-4 bg-gradient-to-b from-white to-[var(--background-mint)] rounded-2xl border border-[var(--border)] shadow-sm hover:shadow-md hover:border-[var(--secondary-green)] transition-all"
                >
                  <span className="font-bold text-[var(--primary-green)] text-lg">{org.name}</span>
                  <span className="text-xs text-[var(--text-muted)] mt-0.5">{org.sub}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* FINAL CTA */}
      <section className="py-20 bg-gradient-to-br from-[var(--primary-green)] to-[var(--secondary-green)] text-white relative overflow-hidden">
        <motion.div
          className="absolute inset-0 opacity-10"
          animate={{ backgroundPosition: ["0% 0%", "100% 100%"] }}
          transition={{ duration: 20, repeat: Infinity, repeatType: "reverse" }}
          style={{ backgroundImage: "radial-gradient(circle at 30% 50%, white 1px, transparent 1px), radial-gradient(circle at 70% 20%, white 1px, transparent 1px)", backgroundSize: "60px 60px" }}
        />
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Early Detection Saves Vision
            </h2>
            <p className="text-xl mb-8 text-white/90">
              Start your child's free myopia risk screening today
            </p>
            <div className="relative inline-flex">
              <motion.div
                className="absolute inset-0 rounded-full bg-white"
                animate={{ scale: [1, 1.2, 1], opacity: [0.4, 0, 0.4] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              />
              <button
                onClick={() => navigate("/screen")}
                className="relative px-10 py-5 bg-white text-[var(--primary-green)] rounded-full hover:bg-gray-50 transition-all font-bold text-lg shadow-xl hover:shadow-2xl transform hover:scale-105"
              >
                Begin Screening Now →
              </button>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
}
