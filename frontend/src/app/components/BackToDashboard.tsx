import { Link } from "react-router";
import { ArrowLeft } from "lucide-react";

export default function BackToDashboard() {
  return (
    <Link
      to="/dashboard"
      className="inline-flex items-center gap-2 mb-6 px-4 py-2 rounded-full bg-white shadow-sm border border-[var(--border)] text-sm font-semibold text-[var(--text-dark)] hover:shadow-md hover:-translate-x-0.5 transition-all"
    >
      <ArrowLeft className="w-4 h-4" />
      Back to Dashboard
    </Link>
  );
}
