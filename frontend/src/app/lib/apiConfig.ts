const envApiUrl = (import.meta.env.VITE_API_URL || "").trim();
const isDev = import.meta.env.DEV;

const defaultApiUrl = isDev ? "http://localhost:5001" : "";

// Remove trailing slash to avoid accidental double slashes in fetch URLs.
export const API_URL = (envApiUrl || defaultApiUrl).replace(/\/+$/, "");
