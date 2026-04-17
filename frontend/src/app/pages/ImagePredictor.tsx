import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { Upload, Loader2, Eye, AlertTriangle, CheckCircle2 } from "lucide-react";
import { predictMyopiaFromImage, type ImagePredictionResult } from "../lib/imageApi";

export default function ImagePredictor() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImagePredictionResult | null>(null);

  const previewUrl = useMemo(() => {
    if (!file) return null;
    return URL.createObjectURL(file);
  }, [file]);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0] ?? null;
    setResult(null);
    setError(null);
    setFile(selected);
  };

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

  return (
    <div className="min-h-screen bg-[var(--background-light)] px-4 py-10 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="rounded-3xl border border-[var(--border)] bg-white p-6 shadow-sm sm:p-8"
        >
          <div className="mb-6 flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[var(--primary-green)]/10">
              <Eye className="h-5 w-5" style={{ color: "var(--primary-green)" }} />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-[var(--text-dark)]">Image-Based Myopia Classifier</h1>
              <p className="text-sm text-[var(--text-muted)]">
                Upload a retinal/eye image to run your Keras classifier via backend API.
              </p>
            </div>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-dashed border-[var(--border)] p-5">
              <label className="mb-3 block text-sm font-semibold text-[var(--text-dark)]">Upload Image</label>
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded-xl border border-[var(--border)] bg-[var(--background-mint)] px-4 py-3 text-sm font-medium text-[var(--text-dark)] hover:opacity-90">
                <Upload className="h-4 w-4" />
                Choose File
                <input type="file" accept="image/*" className="hidden" onChange={onFileChange} />
              </label>

              <p className="mt-3 text-xs text-[var(--text-muted)]">
                Model input is auto-resized to 224 x 224 x 3.
              </p>

              {file && (
                <p className="mt-3 text-sm text-[var(--text-dark)]">
                  Selected: <span className="font-semibold">{file.name}</span>
                </p>
              )}

              <button
                onClick={onPredict}
                disabled={!file || loading}
                className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--primary-green)] px-4 py-3 font-semibold text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eye className="h-4 w-4" />}
                {loading ? "Predicting..." : "Run Prediction"}
              </button>
            </div>

            <div className="rounded-2xl border border-[var(--border)] bg-white p-5">
              <p className="mb-3 text-sm font-semibold text-[var(--text-dark)]">Preview</p>
              {previewUrl ? (
                <img src={previewUrl} alt="Uploaded preview" className="h-64 w-full rounded-xl object-cover" />
              ) : (
                <div className="flex h-64 items-center justify-center rounded-xl bg-[var(--background-mint)] text-sm text-[var(--text-muted)]">
                  No image selected
                </div>
              )}
            </div>
          </div>

          {error && (
            <div className="mt-6 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              <AlertTriangle className="mt-0.5 h-4 w-4" />
              <span>{error}</span>
            </div>
          )}

          {result && (
            <div className="mt-6 rounded-2xl border border-[var(--border)] bg-[var(--background-mint)] p-5">
              <div className="mb-3 flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4" style={{ color: "var(--primary-green)" }} />
                <p className="text-sm font-semibold text-[var(--text-dark)]">Prediction Result</p>
              </div>
              <div className="grid gap-2 text-sm text-[var(--text-dark)] sm:grid-cols-2">
                <p>
                  Label: <span className="font-bold">{result.label}</span>
                </p>
                <p>
                  Myopia Probability: <span className="font-bold">{(result.myopia_probability * 100).toFixed(2)}%</span>
                </p>
                <p>
                  Normal Probability: <span className="font-bold">{(result.normal_probability * 100).toFixed(2)}%</span>
                </p>
                <p>
                  Response Time: <span className="font-bold">{result.duration_ms} ms</span>
                </p>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
