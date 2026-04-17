require("dotenv").config();
const express = require("express");
const cors = require("cors");
const connectDB = require("./config/db");

// Connect to MongoDB
connectDB();

const app = express();

// Middleware
const defaultAllowedOrigins = [
  "http://localhost:5173",
  "http://localhost:5174",
  "http://localhost:3000",
  "https://mayopia-frontend.vercel.app",
];

const envAllowedOrigins = (process.env.CORS_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

const allowedOrigins = envAllowedOrigins.length > 0 ? envAllowedOrigins : defaultAllowedOrigins;

const isTrustedVercelPreviewOrigin = (origin) =>
  /^https:\/\/mayopia-frontend(?:-[a-z0-9-]+)?-unnatis-projects-[a-z0-9]+\.vercel\.app$/i.test(origin) ||
  /^https:\/\/mayopia-frontend(?:-[a-z0-9-]+)?\.vercel\.app$/i.test(origin);

app.use(
  cors({
    origin(origin, callback) {
      // Allow non-browser clients with no origin header.
      if (!origin) return callback(null, true);
      if (allowedOrigins.includes(origin) || isTrustedVercelPreviewOrigin(origin)) {
        return callback(null, true);
      }
      return callback(new Error("CORS origin not allowed"));
    },
    methods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);
app.use(express.json());

// Routes
app.use("/api/auth",   require("./routes/auth"));
app.use("/api/myopia", require("./routes/myopia"));

// Health check
app.get("/health", (_req, res) => res.json({ status: "ok", service: "myopia-guard-node" }));

// 404 handler
app.use((_req, res) => res.status(404).json({ error: "Route not found" }));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`[OK]  Node server running on http://localhost:${PORT}`);
  console.log(`      Routes: /api/auth/signup  /api/auth/login`);
  console.log(`              /api/auth/google`);
  console.log(`              /api/myopia/save   /api/myopia/history`);
});
