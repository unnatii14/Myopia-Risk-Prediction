const mongoose = require("mongoose");

const screeningDataSchema = new mongoose.Schema({
  age:            { type: Number, required: true },
  sex:            { type: String, enum: ["male", "female"] },
  height:         Number,
  weight:         Number,
  familyHistory:  Boolean,
  parentsMyopic:  { type: String, enum: ["none", "one", "both"] },
  screenTime:     Number,
  nearWork:       Number,
  outdoorTime:    Number,
  sports:         String,
  vitaminD:       Boolean,
  // extended fields used by ML model
  locationType:   String,
  schoolType:     String,
  tuition:        Boolean,
  competitiveExam: Boolean,
  state:          String,
}, { _id: false });

const predictionSchema = new mongoose.Schema({
  risk_score:       Number,
  risk_level:       { type: String, enum: ["LOW", "MODERATE", "HIGH"] },
  risk_probability: Number,
  has_re:           Boolean,
  re_probability:   Number,
  diopters:         Number,
  severity:         String,
  source:           { type: String, enum: ["ml", "rule-based"], default: "ml" },
}, { _id: false });

const myopiaRecordSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    screeningData: { type: screeningDataSchema, required: true },
    prediction:    { type: predictionSchema, required: true },
  },
  { timestamps: true }
);

module.exports = mongoose.model("MyopiaRecord", myopiaRecordSchema);
