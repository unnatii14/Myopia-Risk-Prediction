import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { useNavigate } from "react-router";
import { 
  User, Users, Clock, Sun, Smartphone, Book, 
  GraduationCap, School, ChevronRight, ChevronLeft 
} from "lucide-react";
import { Slider } from "../components/ui/slider";

interface FormData {
  // Step 1
  age: number;
  sex: "male" | "female" | "";
  height: number;
  weight: number;
  state: string;
  
  // Step 2
  familyHistory: boolean | null;
  parentsMyopic: "none" | "one" | "both" | "";
  
  // Step 3
  screenTime: number;
  nearWork: number;
  outdoorTime: number;
  sports: "regular" | "occasional" | "rare" | "";
  vitaminD: boolean | null;
  
  // Step 4
  schoolType: "government" | "private" | "international" | "";
  tuition: boolean | null;
  competitiveExam: boolean | null;
}

const initialFormData: FormData = {
  age: 10,
  sex: "",
  height: 0,
  weight: 0,
  state: "",
  familyHistory: null,
  parentsMyopic: "",
  screenTime: 4,
  nearWork: 4,
  outdoorTime: 1,
  sports: "",
  vitaminD: null,
  schoolType: "",
  tuition: null,
  competitiveExam: null,
};

export default function Screen() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [direction, setDirection] = useState(1);

  const totalSteps = 4;

  const updateFormData = (field: keyof FormData, value: any) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const nextStep = () => {
    if (currentStep < totalSteps) {
      setDirection(1);
      setCurrentStep((prev) => prev + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
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
            {["Child Info", "Family History", "Daily Habits", "School & Study"].map((label, i) => (
              <div key={i} className="flex items-start flex-1 last:flex-none">
                <div className="flex flex-col items-center">
                  <motion.div
                    animate={i + 1 === currentStep ? { scale: [1, 1.12, 1] } : {}}
                    transition={{ duration: 1.6, repeat: Infinity }}
                    className={`w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm transition-all duration-300 ${
                      i + 1 < currentStep
                        ? "bg-[var(--primary-green)] text-white shadow-md"
                        : i + 1 === currentStep
                        ? "bg-[var(--primary-green)] text-white ring-4 ring-[var(--secondary-green)]/30 shadow-lg"
                        : "bg-gray-100 text-gray-400"
                    }`}
                  >
                    {i + 1 < currentStep ? "✓" : i + 1}
                  </motion.div>
                  <span className={`text-xs mt-2 font-medium text-center hidden sm:block leading-tight max-w-[64px] transition-colors ${
                    i + 1 === currentStep ? "text-[var(--primary-green)]" : "text-[var(--text-muted)]"
                  }`}>{label}</span>
                </div>
                {i < 3 && (
                  <div className={`flex-1 h-0.5 mt-5 mx-1 transition-all duration-500 ${
                    i + 1 < currentStep ? "bg-[var(--primary-green)]" : "bg-gray-200"
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
                  </div>
                </div>

                {getBMI() && (
                  <div className="p-4 bg-[var(--secondary-green)]/10 rounded-xl">
                    <p className="text-sm text-[var(--text-muted)]">
                      BMI: <span className="font-bold text-[var(--primary-green)]">{getBMI()}</span>
                    </p>
                  </div>
                )}

                {/* State */}
                <div>
                  <label className="block text-sm font-medium mb-2">State</label>
                  <select
                    value={formData.state}
                    onChange={(e) => updateFormData("state", e.target.value)}
                    className="w-full px-4 py-3 rounded-xl border-2 border-gray-200 focus:border-[var(--primary-green)] outline-none"
                  >
                    <option value="">Select state</option>
                    {["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat", "Uttar Pradesh", "West Bengal", "Telangana", "Rajasthan", "Kerala"].map((state) => (
                      <option key={state} value={state}>{state}</option>
                    ))}
                  </select>
                </div>
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

            {/* STEP 4 - SCHOOL & ACADEMIC */}
            {currentStep === 4 && (
              <div className="space-y-8">
                <div className="flex items-center gap-3 mb-6">
                  <GraduationCap className="w-8 h-8 text-[var(--primary-green)]" />
                  <h3 className="text-3xl font-bold text-[var(--text-dark)]">
                    School and study pressure
                  </h3>
                </div>

                {/* School Type */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    School type
                  </label>
                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { value: "government", label: "Government", emoji: "🏫" },
                      { value: "private", label: "Private", emoji: "🏢" },
                      { value: "international", label: "International", emoji: "🌐" },
                    ].map((option) => (
                      <button
                        key={option.value}
                        onClick={() => updateFormData("schoolType", option.value)}
                        className={`p-6 rounded-2xl border-2 transition-all ${
                          formData.schoolType === option.value
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

                {/* Tuition */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    Attends tuition/coaching classes?
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      { value: true, label: "Yes" },
                      { value: false, label: "No" },
                    ].map((option) => (
                      <button
                        key={option.label}
                        onClick={() => updateFormData("tuition", option.value)}
                        className={`p-4 rounded-2xl border-2 transition-all ${
                          formData.tuition === option.value
                            ? "border-[var(--primary-green)] bg-[var(--secondary-green)]/10"
                            : "border-gray-200 hover:border-[var(--secondary-green)]"
                        }`}
                      >
                        <div className="font-medium">{option.label}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Competitive Exam */}
                <div>
                  <label className="block text-sm font-medium mb-4">
                    Preparing for competitive exams (JEE, NEET, etc.)?
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    {[
                      { value: true, label: "Yes" },
                      { value: false, label: "No" },
                    ].map((option) => (
                      <button
                        key={option.label}
                        onClick={() => updateFormData("competitiveExam", option.value)}
                        className={`p-4 rounded-2xl border-2 transition-all ${
                          formData.competitiveExam === option.value
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
            disabled={currentStep === 1}
            className={`flex items-center gap-2 px-6 py-3 rounded-full font-medium transition-all ${
              currentStep === 1
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
              className="flex items-center gap-2 px-6 py-3 bg-[var(--primary-green)] text-white rounded-full hover:bg-[var(--secondary-green)] transition-all font-medium shadow-lg"
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
