import { API_URL } from "./apiConfig";

export interface ImagePredictionResult {
  label: "MYOPIA" | "NORMAL";
  myopia_probability: number;
  normal_probability: number;
  threshold: number;
  model_input_size: [number, number];
  duration_ms: number;
  input_ok?: boolean;
  input_warning?: string | null;
}

export async function predictMyopiaFromImage(file: File): Promise<ImagePredictionResult> {
  const formData = new FormData();
  formData.append("image", file);

  const res = await fetch(`${API_URL}/predict-image`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    const message = typeof errBody?.error === "string" ? errBody.error : `Image API error ${res.status}`;
    throw new Error(message);
  }

  return res.json() as Promise<ImagePredictionResult>;
}

export type ReportedLabel = "myopia" | "normal" | "unknown";

export async function contributeImage(
  token: string,
  file: File,
  consent: boolean,
  reportedLabel: ReportedLabel,
  modelPrediction?: string,
  modelConfidence?: number,
): Promise<{ message: string; status: string }> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("consent", String(consent));
  fd.append("reported_label", reportedLabel);
  if (modelPrediction) fd.append("model_prediction", modelPrediction);
  if (modelConfidence != null) fd.append("model_confidence", String(modelConfidence));

  const res = await fetch(`${API_URL}/contribute-image`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: fd,
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    const message = typeof errBody?.error === "string" ? errBody.error : `Contribution error ${res.status}`;
    throw new Error(message);
  }

  return res.json() as Promise<{ message: string; status: string }>;
}
