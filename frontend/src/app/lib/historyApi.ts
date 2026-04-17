import { API_URL } from "./apiConfig";

export interface ScreeningRecord {
  id: number;
  child_name: string | null;
  screened_at: string;
  risk_score: number;
  risk_level: "LOW" | "MODERATE" | "HIGH";
  has_re: boolean;
  diopters: number | null;
  severity: string | null;
  input_data: Record<string, unknown>;
}

function authHeaders(token: string) {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };
}

export async function saveScreening(
  token: string,
  inputData: Record<string, unknown>,
  result: {
    risk_score: number;
    risk_level: string;
    has_re: boolean;
    diopters: number | null;
    severity: string | null;
  }
): Promise<void> {
  await fetch(`${API_URL}/history/save`, {
    method: "POST",
    headers: authHeaders(token),
    body: JSON.stringify({ input_data: inputData, result }),
  });
}

export async function fetchHistory(token: string): Promise<ScreeningRecord[]> {
  const res = await fetch(`${API_URL}/history`, {
    headers: authHeaders(token),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function fetchLatestScreening(
  token: string
): Promise<ScreeningRecord | null> {
  const res = await fetch(`${API_URL}/history/latest`, {
    headers: authHeaders(token),
  });
  if (!res.ok) return null;
  const data = await res.json();
  return data ?? null;
}
