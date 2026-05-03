import { API_URL } from "./apiConfig";

export interface ImagePredictionResult {
  label: "MYOPIA" | "NORMAL";
  myopia_probability: number;
  normal_probability: number;
  threshold: number;
  model_input_size: [number, number];
  duration_ms: number;
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
