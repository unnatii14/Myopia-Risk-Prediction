import { motion } from "motion/react";

// Iris fibre lines radiating from centre
const IRIS_FIBRES = Array.from({ length: 32 }, (_, i) => {
  const angle = (Math.PI * 2 * i) / 32;
  return {
    x1: 200 + Math.cos(angle) * 28,
    y1: 150 + Math.sin(angle) * 28,
    x2: 200 + Math.cos(angle) * 58,
    y2: 150 + Math.sin(angle) * 58,
  };
});

// Diagnostic tick marks on outer scanner ring
const TICKS = Array.from({ length: 36 }, (_, i) => {
  const angle = (Math.PI * 2 * i) / 36;
  const isMain = i % 9 === 0;
  const r1 = isMain ? 155 : 160;
  const r2 = 168;
  return {
    x1: 200 + Math.cos(angle) * r1,
    y1: 150 + Math.sin(angle) * r1,
    x2: 200 + Math.cos(angle) * r2,
    y2: 150 + Math.sin(angle) * r2,
    isMain,
  };
});

export default function BlinkingEye() {
  return (
    <div className="relative w-full max-w-[440px] mx-auto select-none">
      {/* Outer ambient glow layers */}
      <motion.div
        className="absolute inset-0 rounded-full pointer-events-none"
        style={{ background: "radial-gradient(circle, rgba(82,183,136,0.18) 0%, transparent 70%)" }}
        animate={{ scale: [1, 1.08, 1], opacity: [0.7, 1, 0.7] }}
        transition={{ duration: 3.5, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.div
        className="absolute inset-0 rounded-full pointer-events-none"
        style={{ background: "radial-gradient(circle, rgba(116,194,225,0.12) 0%, transparent 60%)" }}
        animate={{ scale: [1.05, 1, 1.05], opacity: [0.5, 0.9, 0.5] }}
        transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut", delay: 1 }}
      />

      <motion.svg
        viewBox="0 0 400 300"
        className="w-full h-auto"
        xmlns="http://www.w3.org/2000/svg"
        animate={{ y: [0, -12, 0] }}
        transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
      >
        <defs>
          {/* Iris radial gradient — deep green → mint → teal */}
          <radialGradient id="irisGrad" cx="50%" cy="45%">
            <stop offset="0%" stopColor="#1B3A2A" />
            <stop offset="35%" stopColor="#2D6A4F" />
            <stop offset="65%" stopColor="#52B788" />
            <stop offset="100%" stopColor="#74C2E1" stopOpacity="0.7" />
          </radialGradient>

          {/* Pupil gradient */}
          <radialGradient id="pupilGrad" cx="40%" cy="38%">
            <stop offset="0%" stopColor="#3a3a3a" />
            <stop offset="100%" stopColor="#0a0f0c" />
          </radialGradient>

          {/* Sclera (white of eye) gradient */}
          <radialGradient id="scleraGrad" cx="50%" cy="40%">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#e8f3ee" />
          </radialGradient>

          {/* Scanner ring glow filter */}
          <filter id="scanGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Soft eye glow */}
          <filter id="eyeGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Clip path for eyelid blink */}
          <clipPath id="eyeShape">
            <ellipse cx="200" cy="150" rx="138" ry="88" />
          </clipPath>
        </defs>

        {/* ─── Pulse rings behind eye ─── */}
        <motion.circle
          cx="200" cy="150" r="120"
          fill="none" stroke="#52B788" strokeWidth="1.5"
          opacity="0"
          animate={{ r: [100, 170], opacity: [0.6, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeOut", delay: 0 }}
        />
        <motion.circle
          cx="200" cy="150" r="100"
          fill="none" stroke="#74C2E1" strokeWidth="1"
          opacity="0"
          animate={{ r: [100, 175], opacity: [0.4, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeOut", delay: 1.5 }}
        />

        {/* ─── Rotating diagnostic scanner ring ─── */}
        <motion.g
          style={{ transformOrigin: "200px 150px" }}
          animate={{ rotate: 360 }}
          transition={{ duration: 18, repeat: Infinity, ease: "linear" }}
        >
          {/* Dashed ring */}
          <circle
            cx="200" cy="150" r="163"
            fill="none"
            stroke="#52B788"
            strokeWidth="1.5"
            strokeDasharray="6 8"
            opacity="0.5"
            filter="url(#scanGlow)"
          />
          {/* Tick marks */}
          {TICKS.map((t, i) => (
            <line
              key={i}
              x1={t.x1} y1={t.y1} x2={t.x2} y2={t.y2}
              stroke={t.isMain ? "#74C2E1" : "#52B788"}
              strokeWidth={t.isMain ? 2.5 : 1}
              opacity={t.isMain ? 0.9 : 0.4}
            />
          ))}
        </motion.g>

        {/* Counter-rotating accent arc */}
        <motion.g
          style={{ transformOrigin: "200px 150px" }}
          animate={{ rotate: -360 }}
          transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
        >
          <circle
            cx="200" cy="150" r="178"
            fill="none"
            stroke="#2D6A4F"
            strokeWidth="2"
            strokeDasharray="30 120 10 120"
            opacity="0.35"
          />
        </motion.g>

        {/* ─── Eye white (sclera) ─── */}
        <ellipse
          cx="200" cy="150"
          rx="138" ry="88"
          fill="url(#scleraGrad)"
          filter="url(#eyeGlow)"
        />

        {/* ─── Iris ─── */}
        <g clipPath="url(#eyeShape)">
          <circle cx="200" cy="150" r="65" fill="url(#irisGrad)" />

          {/* Iris fibres */}
          {IRIS_FIBRES.map((f, i) => (
            <line
              key={i}
              x1={f.x1} y1={f.y1} x2={f.x2} y2={f.y2}
              stroke="rgba(116,194,225,0.25)"
              strokeWidth="0.8"
            />
          ))}

          {/* Iris limbal ring */}
          <circle cx="200" cy="150" r="63" fill="none" stroke="#1B2B26" strokeWidth="4" opacity="0.5" />

          {/* Iris inner ring highlight */}
          <circle cx="200" cy="150" r="30" fill="none" stroke="rgba(116,194,225,0.3)" strokeWidth="1.5" />

          {/* ─── Pupil with dilation animation ─── */}
          <motion.circle
            cx="200" cy="150"
            fill="url(#pupilGrad)"
            animate={{ r: [24, 28, 24] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
          />

          {/* Primary reflection */}
          <circle cx="186" cy="136" r="9" fill="white" opacity="0.92" />
          {/* Secondary reflection */}
          <circle cx="211" cy="143" r="4" fill="white" opacity="0.55" />
          {/* Tiny sparkle */}
          <circle cx="194" cy="158" r="2" fill="white" opacity="0.3" />
        </g>

        {/* ─── Eye outline (always on top) ─── */}
        <ellipse
          cx="200" cy="150"
          rx="138" ry="88"
          fill="none"
          stroke="#2D6A4F"
          strokeWidth="2.5"
        />

        {/* Upper eyelid curve line */}
        <path
          d="M 62 150 Q 130 72, 200 62 Q 270 72, 338 150"
          fill="none" stroke="#2D6A4F" strokeWidth="3" strokeLinecap="round"
        />
        {/* Lower eyelid curve line */}
        <path
          d="M 62 150 Q 130 228, 200 238 Q 270 228, 338 150"
          fill="none" stroke="#2D6A4F" strokeWidth="2.5" strokeLinecap="round"
        />

        {/* ─── Eyelashes (top) ─── */}
        {Array.from({ length: 12 }, (_, i) => {
          const t = i / 11;
          const angle = Math.PI + t * Math.PI; // left to right across top
          const ex = 62 + t * 276;
          const baseY = 150 - 88 * Math.sin(Math.acos((ex - 200) / 138));
          const lashAngle = -Math.PI / 2 - (t - 0.5) * 0.6;
          const len = 14 + Math.sin(t * Math.PI) * 6;
          return (
            <line
              key={`lash-top-${i}`}
              x1={ex} y1={baseY}
              x2={ex + Math.cos(lashAngle) * len}
              y2={baseY + Math.sin(lashAngle) * len}
              stroke="#1B2B26"
              strokeWidth="2"
              strokeLinecap="round"
            />
          );
        })}

        {/* ─── Animated blink (eyelid covers eye) ─── */}
        <motion.ellipse
          cx="200" cy="150"
          rx="140" ry="90"
          fill="#F0F7F4"
          animate={{ scaleY: [0, 0, 0, 1, 0.05, 0] }}
          transition={{
            duration: 6,
            repeat: Infinity,
            times: [0, 0.78, 0.79, 0.815, 0.83, 0.845],
            ease: "easeInOut",
          }}
          style={{ transformOrigin: "200px 150px" }}
        />

        {/* ─── Scan line sweep inside eye ─── */}
        <clipPath id="eyeClip2">
          <ellipse cx="200" cy="150" rx="136" ry="86" />
        </clipPath>
        <g clipPath="url(#eyeClip2)">
          <motion.rect
            x="60" y="62" width="280" height="6"
            fill="url(#scanLineGrad)"
            opacity="0.25"
            animate={{ y: [62, 238, 62] }}
            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
          />
          <defs>
            <linearGradient id="scanLineGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#74C2E1" stopOpacity="0" />
              <stop offset="50%" stopColor="#74C2E1" stopOpacity="1" />
              <stop offset="100%" stopColor="#74C2E1" stopOpacity="0" />
            </linearGradient>
          </defs>
        </g>
      </motion.svg>
    </div>
  );
}
