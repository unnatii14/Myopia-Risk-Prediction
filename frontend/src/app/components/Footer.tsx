import { Link } from "react-router";

export default function Footer() {
  return (
    <footer className="bg-white border-t border-[var(--border)] mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Disclaimer */}
        <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-2xl">
          <p className="text-sm text-[var(--text-dark)]">
            <strong>Disclaimer:</strong> This is not a medical diagnosis. This AI-powered tool provides a risk assessment based on lifestyle factors and family history. Please consult a qualified ophthalmologist for proper eye examination and diagnosis.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-8">
          <div>
            <h4 className="font-semibold mb-4">About MyopiaGuard</h4>
            <p className="text-sm text-[var(--text-muted)]">
              AI-powered myopia risk screening for Indian school children, backed by research from leading institutions.
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
                  About & Research
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Resources</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link to="/about" className="text-[var(--text-muted)] hover:text-[var(--primary-green)]">
                  Research Papers
                </Link>
              </li>
              <li>
                <a href="#" className="text-[var(--text-muted)] hover:text-[var(--primary-green)]">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="text-[var(--text-muted)] hover:text-[var(--primary-green)]">
                  Contact Us
                </a>
              </li>
            </ul>
          </div>

        </div>

        <div className="pt-8 border-t border-[var(--border)] text-center text-sm text-[var(--text-muted)]">
          <p>&copy; 2026 MyopiaGuard. Made for Indian families with care.</p>
        </div>
      </div>
    </footer>
  );
}
