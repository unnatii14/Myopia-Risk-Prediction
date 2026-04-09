import React, { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { useNavigate } from "react-router";
import {
  User, Users, Clock, Sun, Smartphone, Book, Eye,
  ChevronRight, ChevronLeft
} from "lucide-react";
import { Slider } from "../components/ui/slider";

interface FormData {
  // Step 0 (NEW)
  existingMyopiaStatus: "none" | "near" | "distance" | "";
  currentPrescription?: number; // diopters (negative)
  diagnosisAge?: number;
  myopiaControl?: string; // "none" | "atropine" | "ortho-k" | "glasses"
  progressionRate?: "slow" | "moderate" | "fast" | "";

  // Step 1
  childName: string;
  age: number;
  sex: "male" | "female" | "";
  height: number;
  weight: number;

  // Step 2
  familyHistory: boolean | null;
  parentsMyopic: "none" | "one" | "both" | "";

  // Step 3
  screenTime: number;
  nearWork: number;
  outdoorTime: number;
  sports: "regular" | "occasional" | "rare" | "";
  vitaminD: boolean | null;
}

const initialFormData: FormData = {
  existingMyopiaStatus: "",
  currentPrescription: undefined,
  diagnosisAge: undefined,
  myopiaControl: undefined,
  progressionRate: "",
  childName: "",
  age: 10,
  sex: "",
  height: 0,
  weight: 0,
  familyHistory: null,
  parentsMyopic: "",
  screenTime: 4,
  nearWork: 4,
  outdoorTime: 1,
  sports: "",
  vitaminD: null,
};

export default function Screen() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [direction, setDirection] = useState(1);
  const [touched, setTouched] = useState<Record<string, boolean>>({});

  const totalSteps = 3;

  const updateFormData = (field: keyof FormData, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    setTouched((prev) => ({ ...prev, [String(field)]: true }));
  };

  const isStepValid = (step: number) => {
    if (step === 0) {
      return formData.existingMyopiaStatus !== "";
    }
    if (step === 1) {
      const childNameValid = formData.childName.trim().length > 0;
      const heightValid = formData.height >= 50 && formData.height <= 220;
      const weightValid = formData.weight >= 10 && formData.weight <= 200;
      const sexValid = formData.sex !== "";
      return childNameValid && heightValid && weightValid && sexValid;
    }
    if (step === 2) {
      const famValid = formData.familyHistory !== null;
      const parentsValid = formData.parentsMyopic !== "";
      return famValid && parentsValid;
    }
    return true;
  };

  const nextStep = () => {
    if (currentStep < totalSteps && isStepValid(currentStep)) {
      setDirection(1);
      setCurrentStep((prev) => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setDirection(-1);
      setCurrentStep((prev) => prev - 1);
    }
  };

  const handleSubmit = () => {
    // Store form data in sessionStorage to pass to results page
    sessionStorage.setItem("screeningData", JSON.stringify(formData));
    navigate("/results");
  };

  const getBMI = () => {
    if (formData.height > 0 && formData.weight > 0) {
      const heightInMeters = formData.height / 100;
      return (formData.weight / (heightInMeters * heightInMeters)).toFixed(1);
    }
    return null;
  };

  const getSliderColor = (value: number, type: "screen" | "outdoor" | "nearwork") => {
    if (type === "screen") {
      if (value <= 2) return "var(--secondary-green)";
      if (value <= 5) return "var(--moderate-risk)";
      return "var(--warning-coral)";
    }
    if (type === "outdoor") {
      if (value >= 2) return "var(--secondary-green)";
      if (value >= 1) return "var(--moderate-risk)";
      return "var(--warning-coral)";
    }
    return "var(--secondary-green)";
  };

  const slideVariants = {
    enter: (direction: number) => ({
      x: direction > 0 ? 300 : -300,
      opacity: 0,
    }),
    center: {
      x: 0,
      opacity: 1,
    },
    exit: (direction: number) => ({
      x: direction > 0 ? -300 : 300,
      opacity: 0,
    }),
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[var(--background-mint)] to-white py-12 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Step Progress */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[var(--text-dark)] mb-8">
            Myopia Risk Screening
          </h2>
          <div className="flex items-start">
            {["Myopia Status", "Child Info", "Family History", "Daily Habits"].map((label, i) => (
              <div key={i} className="flex items-start flex-1 last:flex-none">
                <div className="flex flex-col items-center">
                  <motion.div
                    animate={i === currentStep ? { scale: [1, 1.12, 1] } : {}}
                    transition={{ duration: 1.6, repeat: Infinity }}
                    className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all duration-300 ${
                      i < currentStep
                        ? "bg-[var(--primary-green)] text-white shadow-md"
                        : i === currentStep
                        ? "bg-[var(--primary-green)] text-white ring-4 ring-[var(--secondary-green)]/30 shadow-lg"
                        : "bg-gray-100 text-gray-400"
                    }`}
                  >
                    {i < currentStep ? "✓" : i + 1}
                  </motion.div>
                  <span className={`text-xs mt-2 font-medium text-center hidden sm:block leading-tight max-w-[64px] transition-colors ${
                    i === currentStep ? "text-[var(--primary-green)]" : "text-[var(--text-muted)]"
                  }`}>{label}</span>
                </div>
                {i < 3 && (
                  <div className={`flex-1 h-0.5 mt-5 mx-1 transition-all duration-500 ${
                    i < currentStep ? "bg-[var(--primary-green)]" : "bg-gray-200"
                  }`} />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Form Steps */}
        <AnimatePresence mode="wait" custom={direction}>
          <motion.div
            key={currentStep}
            custom={direction}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            className="bg-white/95 backdrop-blur-sm rounded-3xl p-8 shadow-2xl border border-white"
          >
            {/* STEP 0 - MYOPIA STATUS */}
            {currentStep === 0 && (
              <div className="space-y-8">
                <div className="flex items-center gap-3 mb-2">
                  <Eye className="w-8 h-8 text-[var(--primary-green)]" />
                  <h3 className="text-3xl font-bold text-[var(--text-dark)]">
                    Does your child wear glasses?
                  </h3>
                </div>
                <p className="text-sm text-[var(--text-muted)] mb-4">
                  Your answer determines what we assess: <strong>No glasses</strong> → we predict risk of developing myopia. <strong>Distance vision glasses</strong> → we assess progression risk. <strong>Near work glasses only</strong> → this tool is designed for myopia (distance vision) screening.
                </p>

                <div className="grid gap-4">
                  {[
                    { value: "none", label: "No glasses", emoji: "✅", desc: "Child has not been diagnosed with myopia — we'll assess risk of developing it" },
                    { value: "near", label: "Yes, for near work only", emoji: "📖", desc: "For reading/close-up (hyperopia or astigmatism) — this is NOT myopia" },
                    { value: "distance", label: "Yes, for distance vision", emoji: "👓", desc: "Child cannot see far clearly — myopia is already diagnosed" },
                  ].map((option) => (
                    <button
                      key={option.value}
                      onClick={() => updateFormData("existingMyopiaStatus", option.value as any)}
                      className={`p-6 rounded-2xl border-2 transition-all text-left ${
                        formData.existingMyopiaStatus === option.value
                          ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                          : "border-gray-200 hover:border-[var(--secondary-green)]"
                      }`}
                    >
                      <div className="flex items-start gap-4">
                        <div className="text-4xl">{option.emoji}</div>
                        <div className="flex-1">
                          <div className="font-bold text-lg text-[var(--text-dark)]">{option.label}</div>
                          <div className="text-sm text-[var(--text-muted)]">{option.desc}</div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>

                {/* Warning for hyperopia/near case */}
                {formData.existingMyopiaStatus === "near" && (
                  <div className="p-5 bg-amber-50 border-2 border-amber-300 rounded-2xl">
                    <p className="font-semibold text-amber-800 mb-2">⚠️ Different condition — please read</p>
                    <p className="text-sm text-amber-700 mb-3">
                      Glasses for <strong>near work only</strong> suggest <strong>hyperopia</strong> (farsightedness) or astigmatism — not myopia (nearsightedness). This tool is designed to screen for <strong>myopia risk</strong>.
                    </p>
                    <p className="text-sm text-amber-700">
                      You can still continue — the lifestyle risk factors (screen time, outdoor time, family history) are still relevant for predicting whether your child may also develop myopia. However, the result should be interpreted carefully.
                    </p>
                  </div>
                )}

                {/* Conditional fields for existing myopia */}
                {formData.existingMyopiaStatus === "distance" && (
                  <div className="space-y-6 p-6 bg-[var(--background-mint)]/50 rounded-xl border border-[var(--secondary-green)]/20">
                    <div>
                      <label className="block text-sm font-medium mb-4">
                        Current glasses prescription (diopters):
                        <span className="text-lg font-bold text-[var(--primary-green)] ml-2">
                          {formData.currentPrescription ? `-${formData.currentPrescription.toFixed(2)}D` : "Not specified"}
                        </span>
                      </label>
                      <Slider
                        value={[formData.currentPrescription || 0.5]}
                        onValueChange={(value) => updateFormData("currentPrescription", Number(value[0].toFixed(2)))}
                        min={0.25}
                        max={10}
                        step={0.25}
                        className="w-full"
                      />
                      <p className="text-xs text-[var(--text-muted)] mt-2">Optional: Leave as 0 if not known</p>
                    </div>

                   <div>
                      <label className="block text-sm font-medium mb-4">
                        Age diagnosed: <span className="text-lg font-bold text-[var(--primary-green)]">{formData.diagnosisAge || "?"} years</span>
                      </label>
                      <Slider
                        value={[formData.diagnosisAge || 5]}
                        onValueChange={(value) => updateFormData("diagnosisAge", value[0])}
                        min={3}
                        max={15}
                        step={1}
                        className="w-full"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-4">Current myopia control method</label>
                      <div className="grid grid-cols-2 gap-3">
                        {[
                          { value: "none", label: "None", emoji: "❌" },
                          { value: "atropine", label: "Atropine drops", emoji: "💧" },
                          { value: "ortho-k", label: "Ortho-K lenses", emoji: "🔵" },
                          { value: "glasses", label: "Control glasses", emoji: "👓" },
                        ].map((option) => (
                          <button
                            key={option.value}
                            onClick={() => updateFormData("myopiaControl", option.value)}
                            className={`p-3 rounded-lg border-2 transition-all text-center text-sm ${
                              formData.myopiaControl === option.value
                                ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                                : "border-gray-200 hover:border-[var(--secondary-green)]"
                            }`}
                          >
                            <div className="text-2xl mb-1">{option.emoji}</div>
                            <div className="font-medium">{option.label}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium mb-4">How fast is myopia progressing?</label>
                      <div className="grid grid-cols-3 gap-3">
                        {[
                          { value: "slow", label: "Slow", emoji: "🐢" },
                          { value: "moderate", label: "Moderate", emoji: "💨" },
                          { value: "fast", label: "Fast", emoji: "🚀" },
                        ].map((option) => (
                          <button
                            key={option.value}
                            onClick={() => updateFormData("progressionRate", option.value as any)}
                            className={`p-4 rounded-lg border-2 transition-all text-center ${
                              formData.progressionRate === option.value
                                ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                                : "border-gray-200 hover:border-[var(--secondary-green)]"
                            }`}
                          >
                            <div className="text-3xl mb-1">{option.emoji}</div>
                            <div className="font-medium text-sm">{option.label}</div>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* STEP 1 - CHILD INFO */}
            {currentStep === 1 && (
              <div className="space-y-8">
                <div className="flex items-center gap-3 mb-6">
                  <User className="w-8 h-8 text-[var(--primary-green)]" />
                  <h3 className="text-3xl font-bold text-[var(--text-dark)]">
                    Tell us about your child
                  </h3>
                </div>

                {/* Age Slider */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    Age: <span className="text-2xl font-bold text-[var(--primary-green)]">{formData.age}</span> years
                  </label>
                  <Slider
                    value={[formData.age]}
                    onValueChange={(value) => updateFormData("age", value[0])}
                    min={5}
                    max={18}
                    step={1}
                    className="w-full"
                  />
                </div>

                {/* Child Name */}
                <div>
                  <label className="block text-sm font-medium mb-2">Child Name</label>
                  <input
                    type="text"
                    value={formData.childName}
                    onChange={(e) => updateFormData("childName", e.target.value)}
                    className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-[var(--primary-green)] outline-none"
                    placeholder="Enter child name"
                  />
                  {touched["childName"] && !formData.childName.trim() && (
                    <p className="mt-1 text-xs text-[var(--warning-coral)]">
                      Child name is required for the report.
                    </p>
                  )}
                </div>

                {/* Sex */}
                <div>
                  <label className="block text-sm font-medium mb-4">Sex</label>
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      { value: "male", label: "Male", emoji: "👦" },
                      { value: "female", label: "Female", emoji: "👧" },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => updateFormData("sex", option.value)}
                        className={`p-6 rounded-2xl border-2 transition-all ${
                          formData.sex === option.value
                            ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                            : "border-gray-200 hover:border-[var(--secondary-green)]"
                        }`}
                      >
                        <div className="text-4xl mb-2">{option.emoji}</div>
                        <div className="font-medium">{option.label}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Height & Weight */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Height (cm)
                    </label>
                    <input
                      type="number"
                      value={formData.height || ""}
                      onChange={(e) => updateFormData("height", Number(e.target.value))}
                      className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-[var(--primary-green)] outline-none"
                      placeholder="140"
                    />
                    {touched["height"] && (formData.height < 50 || formData.height > 220) && (
                      <p className="mt-1 text-xs text-[var(--warning-coral)]">
                        Please enter a realistic height between 50 and 220 cm.
                      </p>
                    )}
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      Weight (kg)
                    </label>
                    <input
                      type="number"
                      value={formData.weight || ""}
                      onChange={(e) => updateFormData("weight", Number(e.target.value))}
                      className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-[var(--primary-green)] outline-none"
                      placeholder="35"
                    />
                    {touched["weight"] && (formData.weight < 10 || formData.weight > 200) && (
                      <p className="mt-1 text-xs text-[var(--warning-coral)]">
                        Please enter a realistic weight between 10 and 200 kg.
                      </p>
                    )}
                  </div>
                </div>

                {getBMI() && (formData.height >= 50 && formData.height <= 220 && formData.weight >= 10 && formData.weight <= 200) && (
                  <div className="p-4 bg-[var(--secondary-green)]/10 rounded-xl">
                    <p className="text-sm text-[var(--text-muted)]">
                      BMI: <span className="font-bold text-[var(--primary-green)]">{getBMI()}</span>
                    </p>
                  </div>
                )}
                {!isStepValid(1) && (
                  <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-sm">
                    Please complete this section with valid values to continue.
                  </div>
                )}
              </div>
            )}

            {/* STEP 2 - FAMILY HISTORY */}
            {currentStep === 2 && (
              <div className="space-y-8">
                <div className="flex items-center gap-3 mb-6">
                  <Users className="w-8 h-8 text-[var(--primary-green)]" />
                  <h3 className="text-3xl font-bold text-[var(--text-dark)]">
                    Family History
                  </h3>
                </div>

                {/* Family History */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    Does anyone in the family wear glasses for distance vision?
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      { value: true, label: "Yes", emoji: "👨‍👩‍👧‍👦" },
                      { value: false, label: "No", emoji: "✅" },
                    ].map((option) => (
                      <button
                        key={option.label}
                        onClick={() => updateFormData("familyHistory", option.value)}
                        className={`p-6 rounded-2xl border-2 transition-all ${
                          formData.familyHistory === option.value
                            ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                            : "border-gray-200 hover:border-[var(--secondary-green)]"
                        }`}
                      >
                        <div className="text-4xl mb-2">{option.emoji}</div>
                        <div className="font-medium">{option.label}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Parents Myopic */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    Do the parents wear glasses for distance vision (myopia)?
                  </label>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { value: "none", label: "None", emoji: "👨👩" },
                      { value: "one", label: "One Parent", emoji: "👨‍🦯👩" },
                      { value: "both", label: "Both Parents", emoji: "👨‍🦯👩‍🦯" },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => updateFormData("parentsMyopic", option.value)}
                        className={`p-6 rounded-2xl border-2 transition-all ${
                          formData.parentsMyopic === option.value
                            ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                            : "border-gray-200 hover:border-[var(--secondary-green)]"
                        }`}
                      >
                        <div className="text-3xl mb-2">{option.emoji}</div>
                        <div className="font-medium text-sm">{option.label}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* STEP 3 - DAILY HABITS */}
            {currentStep === 3 && (
              <div className="space-y-8">
                <div className="flex items-center gap-3 mb-6">
                  <Clock className="w-8 h-8 text-[var(--primary-green)]" />
                  <h3 className="text-3xl font-bold text-[var(--text-dark)]">
                    A typical school day
                  </h3>
                </div>

                {/* Screen Time */}
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <Smartphone className="w-5 h-5" />
                    <label className="block text-sm font-medium">
                      Screen time (phone, tablet, computer): <span className="text-xl font-bold" style={{ color: getSliderColor(formData.screenTime, "screen") }}>{formData.screenTime}h</span>
                    </label>
                  </div>
                  <Slider
                    value={[formData.screenTime]}
                    onValueChange={(value) => updateFormData("screenTime", value[0])}
                    min={0}
                    max={16}
                    step={0.5}
                    className="w-full"
                  />
                  {formData.screenTime > 2 && (
                    <p className="mt-2 text-sm text-[var(--warning-coral)] flex items-center gap-1">
                      ⚠️ That's above the recommended 2 hours
                    </p>
                  )}
                </div>

                {/* Near Work */}
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <Book className="w-5 h-5" />
                    <label className="block text-sm font-medium">
                      Near work (reading, homework): <span className="text-xl font-bold text-[var(--primary-green)]">{formData.nearWork}h</span>
                    </label>
                  </div>
                  <Slider
                    value={[formData.nearWork]}
                    onValueChange={(value) => updateFormData("nearWork", value[0])}
                    min={0}
                    max={12}
                    step={0.5}
                    className="w-full"
                  />
                </div>

                {/* Outdoor Time */}
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <Sun className="w-5 h-5" />
                    <label className="block text-sm font-medium">
                      Outdoor time (daylight): <span className="text-xl font-bold" style={{ color: getSliderColor(formData.outdoorTime, "outdoor") }}>{formData.outdoorTime}h</span>
                    </label>
                  </div>
                  <Slider
                    value={[formData.outdoorTime]}
                    onValueChange={(value) => updateFormData("outdoorTime", value[0])}
                    min={0}
                    max={10}
                    step={0.5}
                    className="w-full"
                  />
                  {formData.outdoorTime >= 2 ? (
                    <p className="mt-2 text-sm text-[var(--secondary-green)] flex items-center gap-1">
                      ✅ Great! Outdoor time is the #1 natural protector
                    </p>
                  ) : (
                    <p className="mt-2 text-sm text-[var(--warning-coral)] flex items-center gap-1">
                      ⚠️ Low outdoor time increases risk significantly
                    </p>
                  )}
                </div>

                {/* Sports */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    Sports participation
                  </label>
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { value: "regular", label: "Regular" },
                      { value: "occasional", label: "Occasional" },
                      { value: "rare", label: "Rare" },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => updateFormData("sports", option.value)}
                        className={`px-4 py-3 rounded-full border-2 transition-all font-medium ${
                          formData.sports === option.value
                            ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10 text-[var(--primary-green)]"
                            : "border-gray-200 hover:border-[var(--secondary-green)]"
                        }`}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Vitamin D */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    Taking Vitamin D supplement?
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      { value: true, label: "Yes" },
                      { value: false, label: "No" },
                    ].map((option) => (
                      <button
                        key={option.label}
                        onClick={() => updateFormData("vitaminD", option.value)}
                        className={`p-4 rounded-2xl border-2 transition-all ${
                          formData.vitaminD === option.value
                            ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                            : "border-gray-200 hover:border-[var(--secondary-green)]"
                        }`}
                      >
                        <div className="font-medium">{option.label}</div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}


          </motion.div>
        </AnimatePresence>

        {/* Navigation Buttons */}
        <div className="flex justify-between mt-8">
          <button
            onClick={prevStep}
            disabled={currentStep === 0}
            className={`flex items-center gap-2 px-6 py-3 rounded-full font-medium transition-all ${
              currentStep === 0
                ? "opacity-0 pointer-events-none"
                : "bg-white text-[var(--primary-green)] hover:bg-gray-50 shadow-lg"
            }`}
          >
            <ChevronLeft className="w-5 h-5" />
            Previous
          </button>

          {currentStep < totalSteps ? (
            <button
              onClick={nextStep}
              disabled={!isStepValid(currentStep)}
              className={`flex items-center gap-2 px-6 py-3 rounded-full transition-all font-medium shadow-lg ${
                isStepValid(currentStep)
                  ? "bg-[var(--primary-green)] text-white hover:bg-[var(--secondary-green)]"
                  : "bg-gray-200 text-gray-500 cursor-not-allowed"
              }`}
            >
              Next
              <ChevronRight className="w-5 h-5" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              className="flex items-center gap-2 px-8 py-4 bg-[var(--primary-green)] text-white rounded-full hover:bg-[var(--secondary-green)] transition-all font-bold text-lg shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Get My Child's Risk Score 🔍
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
