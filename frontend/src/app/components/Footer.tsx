import { Link } from "react-router";

export default function Footer() {
  return (
    <footer className="bg-white border-t border-[var(--border)] mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Disclaimer */}
        <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-2xl">
          <p className="text-sm text-[var(--text-dark)]">
            <strong>Disclaimer:</strong> This is not a medical diagnosis. This tool provides a risk assessment based on lifestyle factors and family history using a machine learning model trained on Indian children's data. Please consult a qualified ophthalmologist for proper eye examination and diagnosis.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <div>
            <h4 className="font-semibold mb-4">About MyopiaGuard</h4>
            <p className="text-sm text-[var(--text-muted)]">
              A myopia risk screening tool built as a research project for Indian school children (ages 5–18). Uses a GradientBoosting ML model trained on a dataset of 5,000 children.
            </p>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link to="/" className="text-[var(--text-muted)] hover:text-[var(--primary-green)]">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/screen" className="text-[var(--text-muted)] hover:text-[var(--primary-green)]">
                  Start Screening
                </Link>
              </li>
              <li>
                <Link to="/faq" className="text-[var(--text-muted)] hover:text-[var(--primary-green)]">
                  FAQ
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-[var(--text-muted)] hover:text-[var(--primary-green)]">
                  About &amp; Research
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Research Reference</h4>
            <ul className="space-y-2 text-sm text-[var(--text-muted)]">
              <li>
                <Link to="/about" className="hover:text-[var(--primary-green)]">
                  View Research Paper
                </Link>
              </li>
              <li>ML Model: GradientBoosting (AUC 0.893)</li>
              <li>Dataset: 5,000 Indian school children</li>
              <li>Academic Research Project</li>
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-[var(--border)] text-center text-sm text-[var(--text-muted)]">
          <p>&copy; 2026 MyopiaGuard. Academic Research Project.</p>
        </div>
      </div>
    </footer>
  );
}
