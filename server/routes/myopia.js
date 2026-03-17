const express = require("express");
const protect = require("../middleware/auth");
const MyopiaRecord = require("../models/MyopiaRecord");

const router = express.Router();

// All myopia routes require the user to be logged in
router.use(protect);

// ── POST /api/myopia/save ─────────────────────────────────────
// Body: { screeningData: {...}, prediction: {...} }
router.post("/save", async (req, res) => {
  try {
    const { screeningData, prediction } = req.body;

    if (!screeningData || !prediction) {
      return res.status(400).json({ error: "screeningData and prediction are required" });
    }

    const record = await MyopiaRecord.create({
      user: req.user._id,
      screeningData,
      prediction,
    });

    res.status(201).json({ message: "Record saved", id: record._id });
  } catch (err) {
    console.error("Save record error:", err.message);
    res.status(500).json({ error: "Failed to save record" });
  }
});

// ── GET /api/myopia/history ───────────────────────────────────
// Returns all past records for the logged-in user, newest first
router.get("/history", async (req, res) => {
  try {
    const records = await MyopiaRecord.find({ user: req.user._id })
      .sort({ createdAt: -1 })
      .lean();

    res.json({ records });
  } catch (err) {
    console.error("History error:", err.message);
    res.status(500).json({ error: "Failed to fetch history" });
  }
});

module.exports = router;
