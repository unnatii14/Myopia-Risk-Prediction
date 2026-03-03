import { motion } from "motion/react";

export default function BokehBackground() {
  const circles = [
    { size: 300, x: "10%", y: "20%", color: "#52B788", delay: 0 },
    { size: 200, x: "80%", y: "10%", color: "#74C2E1", delay: 1 },
    { size: 250, x: "70%", y: "70%", color: "#52B788", delay: 2 },
    { size: 180, x: "20%", y: "80%", color: "#74C2E1", delay: 1.5 },
    { size: 150, x: "50%", y: "50%", color: "#2D6A4F", delay: 0.5 },
  ];

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {circles.map((circle, index) => (
        <motion.div
          key={index}
          className="absolute rounded-full blur-3xl"
          style={{
            width: circle.size,
            height: circle.size,
            left: circle.x,
            top: circle.y,
            backgroundColor: circle.color,
            opacity: 0.1,
          }}
          animate={{
            x: [0, 20, -20, 0],
            y: [0, -20, 20, 0],
            scale: [1, 1.1, 0.9, 1],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            delay: circle.delay,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
}
