require("dotenv").config();
const express = require("express");
const cors = require("cors");
const connectDB = require("./config/db");

// Connect to MongoDB
connectDB();

const app = express();

// Middleware
app.use(cors({ origin: ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"] }));
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
  console.log(`              /api/myopia/save   /api/myopia/history`);
});
