import { Link } from "react-router";
import { Home } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[var(--background-mint)] to-white px-4">
      <div className="text-center">
        <h1 className="text-9xl font-bold text-[var(--primary-green)] mb-4">404</h1>
        <h2 className="text-3xl font-bold text-[var(--text-dark)] mb-4">
          Page Not Found
        </h2>
        <p className="text-[var(--text-muted)] mb-8 max-w-md mx-auto">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-6 py-3 bg-[var(--primary-green)] text-white rounded-full hover:bg-[var(--secondary-green)] transition-colors font-medium"
        >
          <Home className="w-5 h-5" />
          Back to Home
        </Link>
      </div>
    </div>
  );
}
