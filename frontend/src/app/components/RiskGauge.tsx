import { motion } from "motion/react";
import { useEffect, useState } from "react";

interface RiskGaugeProps {
  score: number; // 0-100
}

export default function RiskGauge({ score }: RiskGaugeProps) {
  const [displayScore, setDisplayScore] = useState(0);

  useEffect(() => {
    // Animate score from 0 to actual value
    const duration = 1500;
    const steps = 60;
    const increment = score / steps;
    let current = 0;

    const interval = setInterval(() => {
      current += increment;
      if (current >= score) {
        setDisplayScore(score);
        clearInterval(interval);
      } else {
        setDisplayScore(Math.round(current));
      }
    }, duration / steps);

    return () => clearInterval(interval);
  }, [score]);

  // Calculate rotation (180 degrees for semicircle)
  const rotation = (displayScore / 100) * 180;

  // Determine color based on score
  const getColor = (score: number) => {
    if (score < 40) return "var(--low-risk)";
    if (score < 70) return "var(--moderate-risk)";
    return "var(--high-risk)";
  };

  const color = getColor(displayScore);

  return (
    <div className="relative w-64 h-32">
      <svg
        viewBox="0 0 200 100"
        className="w-full h-full"
      >
        {/* Background arc */}
        <path
          d="M 20 90 A 80 80 0 0 1 180 90"
          fill="none"
          stroke="#E8F3EE"
          strokeWidth="16"
          strokeLinecap="round"
        />

        {/* Low risk zone (green) - 0-40% */}
        <path
          d="M 20 90 A 80 80 0 0 1 92 18"
          fill="none"
          stroke="var(--low-risk)"
          strokeWidth="16"
          strokeLinecap="round"
          opacity="0.3"
        />

        {/* Moderate risk zone (amber) - 40-70% */}
        <path
          d="M 92 18 A 80 80 0 0 1 146 12"
          fill="none"
          stroke="var(--moderate-risk)"
          strokeWidth="16"
          strokeLinecap="round"
          opacity="0.3"
        />

        {/* High risk zone (coral) - 70-100% */}
        <path
          d="M 146 12 A 80 80 0 0 1 180 90"
          fill="none"
          stroke="var(--high-risk)"
          strokeWidth="16"
          strokeLinecap="round"
          opacity="0.3"
        />

        {/* Animated progress arc */}
        <motion.path
          d="M 20 90 A 80 80 0 0 1 180 90"
          fill="none"
          stroke={color}
          strokeWidth="16"
          strokeLinecap="round"
          strokeDasharray="251.2"
          initial={{ strokeDashoffset: 251.2 }}
          animate={{ strokeDashoffset: 251.2 - (displayScore / 100) * 251.2 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
        />

        {/* Needle */}
        <motion.g
          initial={{ rotate: 0 }}
          animate={{ rotate: rotation }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          style={{ transformOrigin: "100px 90px" }}
        >
          <line
            x1="100"
            y1="90"
            x2="100"
            y2="25"
            stroke={color}
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx="100" cy="90" r="6" fill={color} />
        </motion.g>

        {/* Center decoration */}
        <circle cx="100" cy="90" r="4" fill="white" />
      </svg>

      {/* Score display */}
      <div className="absolute inset-0 flex items-end justify-center pb-2">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, type: "spring" }}
          className="text-center"
        >
          <div className="text-4xl font-bold" style={{ color }}>
            {displayScore}%
          </div>
        </motion.div>
      </div>

      {/* Labels */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-[var(--text-muted)] px-2">
        <span>0</span>
        <span>50</span>
        <span>100</span>
      </div>
    </div>
  );
}
