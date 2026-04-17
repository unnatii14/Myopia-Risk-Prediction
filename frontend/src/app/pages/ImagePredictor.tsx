import { useCallback, useMemo, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Upload, Loader2, Eye, AlertTriangle, CheckCircle2, XCircle, Camera, Info, RotateCcw } from "lucide-react";
import { predictMyopiaFromImage, type ImagePredictionResult } from "../lib/imageApi";

export default function ImagePredictor() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImagePredictionResult | null>(null);
  const [dragging, setDragging] = useState(false);

  const previewUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  const handleFile = (selected: File | null) => {
    setResult(null);
    setError(null);
    setFile(selected);
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFile(e.target.files?.[0] ?? null);
  };

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped && dropped.type.startsWith("image/")) handleFile(dropped);
  }, []);

  const onPredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictMyopiaFromImage(file);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to predict image");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const isMyopia = result?.label === "MYOPIA";
  const prob = result ? result.myopia_probability * 100 : 0;

  return (
    <div className="min-h-screen bg-[var(--background-mint)] px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="mb-8 text-center"
        >
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-purple-100">
            <Camera className="h-8 w-8 text-purple-600" />
          </div>
          <h1 className="text-3xl font-bold text-[var(--text-dark)]">Image-Based Myopia Detection</h1>
          <p className="mt-2 text-[var(--text-muted)]">
            Upload a <strong>fundus (retinal) photograph</strong> taken with medical eye equipment. The model is not designed for regular phone photos.
          </p>
        </motion.div>

        {/* Fundus image warning banner */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.05 }}
          className="mb-6 rounded-2xl border border-amber-200 bg-amber-50 p-4"
        >
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" />
            <div>
              <p className="font-semibold text-amber-800 text-sm">This tool requires fundus / retinal photographs</p>
              <p className="mt-1 text-xs text-amber-700 leading-relaxed">
                Regular phone selfies or close-up eye photos <strong>will not work</strong> — the model was trained on medical fundus camera images (the orange circular retinal scans taken by eye doctors). Uploading a phone photo will always give unreliable results.
              </p>
              <p className="mt-2 text-xs text-amber-700">
                <strong>How to get a fundus image:</strong> Visit an optometrist and ask for a retinal photograph, or use a sample image from the{" "}
                <a href="https://www.kaggle.com/datasets/sshikamaru/fundus-image-dataset" target="_blank" rel="noreferrer" className="underline font-medium">ORIGA / Kaggle fundus dataset</a>.
              </p>
            </div>
          </div>
        </motion.div>

        <div className="grid gap-6 lg:grid-cols-5">

          {/* Left: Upload + Preview */}
          <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.05 }}
            className="lg:col-span-3 flex flex-col gap-4"
          >
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              className={`relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-all cursor-pointer ${
                dragging
                  ? "border-purple-400 bg-purple-50"
                  : file
                  ? "border-[var(--primary-green)] bg-[var(--background-mint)]"
                  : "border-[var(--border)] bg-white hover:border-purple-300 hover:bg-purple-50/30"
              }`}
              style={{ minHeight: "260px" }}
              onClick={() => !file && document.getElementById("file-input")?.click()}
            >
              <input
                id="file-input"
                type="file"
                accept="image/*"
                className="hidden"
                onChange={onFileChange}
              />

              {previewUrl ? (
                <>
                  <img
                    src={previewUrl}
                    alt="Uploaded preview"
                    className="max-h-52 w-full rounded-xl object-contain"
                  />
                  <p className="mt-3 text-sm font-medium text-[var(--text-dark)]">{file?.name}</p>
                  <p className="text-xs text-[var(--text-muted)]">
                    {file ? (file.size / 1024).toFixed(0) + " KB" : ""}
                  </p>
                </>
              ) : (
                <>
                  <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-purple-100">
                    <Upload className="h-6 w-6 text-purple-500" />
                  </div>
                  <p className="font-semibold text-[var(--text-dark)]">
                    {dragging ? "Drop your image here" : "Drag & drop or click to upload"}
                  </p>
                  <p className="mt-1 text-sm text-[var(--text-muted)]">PNG, JPG, WEBP up to 5 MB</p>
                  <p className="mt-3 text-xs text-[var(--text-muted)] opacity-70">
                    Model input auto-resized to 224 × 224 px
                  </p>
                </>
              )}
            </div>

            {/* Action buttons */}
            <div className="flex gap-3">
              {file && (
                <button
                  onClick={reset}
                  className="flex items-center gap-2 rounded-xl border border-[var(--border)] bg-white px-4 py-3 text-sm font-medium text-[var(--text-muted)] transition-colors hover:bg-red-50 hover:text-red-500"
                >
                  <RotateCcw className="h-4 w-4" />
                  Clear
                </button>
              )}
              <button
                onClick={file ? onPredict : () => document.getElementById("file-input")?.click()}
                disabled={loading}
                className="flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-3 font-semibold text-white transition-all disabled:cursor-not-allowed disabled:opacity-60"
                style={{ background: loading ? "#6B8F80" : "var(--primary-green)" }}
              >
                {loading ? (
                  <><Loader2 className="h-4 w-4 animate-spin" />Analysing image…</>
                ) : file ? (
                  <><Eye className="h-4 w-4" />Run Detection</>
                ) : (
                  <><Upload className="h-4 w-4" />Choose Image</>
                )}
              </button>
            </div>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700"
                >
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Right: Result + Info */}
          <motion.div
            initial={{ opacity: 0, x: 16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="lg:col-span-2 flex flex-col gap-4"
          >
            {/* Result card */}
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`rounded-2xl border-2 p-6 ${
                    isMyopia ? "border-red-200 bg-red-50" : "border-green-200 bg-green-50"
                  }`}
                >
                  {/* Verdict */}
                  <div className="mb-4 flex items-center gap-3">
                    {isMyopia
                      ? <XCircle className="h-8 w-8 text-red-500" />
                      : <CheckCircle2 className="h-8 w-8 text-green-500" />
                    }
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                        Detection Result
                      </p>
                      <p className={`text-2xl font-bold ${isMyopia ? "text-red-600" : "text-green-600"}`}>
                        {result.label}
                      </p>
                    </div>
                  </div>

                  {/* Probability bar */}
                  <div className="mb-4">
                    <div className="mb-1.5 flex justify-between text-xs font-medium text-[var(--text-muted)]">
                      <span>Myopia probability</span>
                      <span className="font-bold text-[var(--text-dark)]">{prob.toFixed(1)}%</span>
                    </div>
                    <div className="h-3 w-full overflow-hidden rounded-full bg-white/60">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${prob}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                        className="h-full rounded-full"
                        style={{ background: isMyopia ? "#EF4444" : "#22C55E" }}
                      />
                    </div>
                  </div>

                  {/* Stats grid */}
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    {[
                      { label: "Myopia", value: `${(result.myopia_probability * 100).toFixed(1)}%` },
                      { label: "Normal", value: `${(result.normal_probability * 100).toFixed(1)}%` },
                      { label: "Threshold", value: `${result.threshold * 100}%` },
                      { label: "Response", value: `${result.duration_ms} ms` },
                    ].map((s) => (
                      <div key={s.label} className="rounded-xl bg-white/70 p-3">
                        <p className="text-xs text-[var(--text-muted)]">{s.label}</p>
                        <p className="font-bold text-[var(--text-dark)]">{s.value}</p>
                      </div>
                    ))}
                  </div>

                  {(prob > 98 || prob < 2) && (
                    <div className="mt-3 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
                      <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                      <span>
                        <strong>Unreliable result.</strong> Extreme confidence ({prob.toFixed(1)}%) usually means the image is not a fundus/retinal photograph. Please upload a medical eye scan, not a regular phone photo.
                      </span>
                    </div>
                  )}
                  <p className="mt-4 text-xs leading-relaxed text-[var(--text-muted)]">
                    This is a research tool. Always consult a qualified eye-care professional for diagnosis.
                  </p>
                </motion.div>
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-[var(--border)] bg-white p-8 text-center"
                  style={{ minHeight: "280px" }}
                >
                  <Eye className="mb-3 h-10 w-10 text-[var(--border)]" />
                  <p className="font-semibold text-[var(--text-muted)]">No result yet</p>
                  <p className="mt-1 text-sm text-[var(--text-muted)] opacity-70">
                    Upload an image and run detection
                  </p>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Info card */}
            <div className="rounded-2xl border border-[var(--border)] bg-white p-5">
              <div className="mb-3 flex items-center gap-2">
                <Info className="h-4 w-4" style={{ color: "var(--primary-green)" }} />
                <p className="text-sm font-semibold text-[var(--text-dark)]">How it works</p>
              </div>
              <ul className="space-y-2 text-xs text-[var(--text-muted)]">
                {[
                  "Requires fundus (retinal) photographs from medical eye equipment",
                  "Phone selfies or casual eye photos will give wrong results",
                  "Image is auto-resized to 224 × 224 px before inference",
                  "Deep-learning classifier (Keras → ONNX, no GPU required)",
                  "Threshold: ≥ 50% probability = MYOPIA classification",
                ].map((tip) => (
                  <li key={tip} className="flex items-start gap-2">
                    <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--primary-green)]" />
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
