const envApiUrl = (import.meta.env.VITE_API_URL || "").trim();
const isLocalHost =
	typeof window !== "undefined" &&
	(window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");

const defaultApiUrl = isLocalHost ? "http://localhost:5001" : "";

// Remove trailing slash to avoid accidental double slashes in fetch URLs.
export const API_URL = (envApiUrl || defaultApiUrl).replace(/\/+$/, "");
