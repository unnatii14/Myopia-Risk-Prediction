import { Eye, Menu, X } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { useEffect, useState } from "react";

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSectionClick = (sectionId: string) => {
    setMobileMenuOpen(false);
    if (location.pathname !== "/") {
      navigate("/");
      setTimeout(() => {
        const element = document.getElementById(sectionId);
        element?.scrollIntoView({ behavior: "smooth" });
      }, 100);
    } else {
      const element = document.getElementById(sectionId);
      element?.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <>
      <motion.nav
        className={`sticky top-0 z-50 transition-all duration-300 ${
          scrolled
            ? "bg-white/80 backdrop-blur-lg shadow-sm"
            : "bg-white"
        }`}
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <Link to="/" className="flex items-center gap-2 group">
              <div className="w-10 h-10 rounded-full bg-[var(--primary-green)] flex items-center justify-center">
                <Eye className="w-5 h-5 text-white" />
              </div>
              <span
                className="text-xl font-bold text-[var(--primary-green)]"
                style={{ fontFamily: "var(--font-heading)" }}
              >
                MyopiaGuard
              </span>
            </Link>

            <div className="hidden md:flex items-center gap-8">
              <button
                onClick={() => handleSectionClick("how-it-works")}
                className="text-[var(--text-dark)] hover:text-[var(--primary-green)] transition-colors"
              >
                How It Works
              </button>
              <button
                onClick={() => handleSectionClick("research")}
                className="text-[var(--text-dark)] hover:text-[var(--primary-green)] transition-colors"
              >
                Research
              </button>
              <Link
                to="/faq"
                className="text-[var(--text-dark)] hover:text-[var(--primary-green)] transition-colors"
              >
                FAQ
              </Link>
              <Link
                to="/about"
                className="text-[var(--text-dark)] hover:text-[var(--primary-green)] transition-colors"
              >
                About
              </Link>
            </div>

            <div className="flex items-center gap-4">
              <Link
                to="/screen"
                className="hidden sm:block px-6 py-3 bg-[var(--primary-green)] text-white rounded-full hover:bg-[var(--secondary-green)] transition-colors font-medium"
              >
                Check My Child's Risk
              </Link>
              
              <button
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                className="md:hidden p-2 text-[var(--text-dark)]"
              >
                {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
              </button>
            </div>
          </div>
        </div>
      </motion.nav>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-white border-b border-[var(--border)] overflow-hidden fixed top-20 left-0 right-0 z-40 shadow-lg"
          >
            <div className="px-4 py-6 space-y-4">
              <button
                onClick={() => handleSectionClick("how-it-works")}
                className="block w-full text-left px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] rounded-xl transition-colors"
              >
                How It Works
              </button>
              <button
                onClick={() => handleSectionClick("research")}
                className="block w-full text-left px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] rounded-xl transition-colors"
              >
                Research
              </button>
              <Link
                to="/faq"
                onClick={() => setMobileMenuOpen(false)}
                className="block w-full text-left px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] rounded-xl transition-colors"
              >
                FAQ
              </Link>
              <Link
                to="/about"
                onClick={() => setMobileMenuOpen(false)}
                className="block w-full text-left px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] rounded-xl transition-colors"
              >
                About
              </Link>
              <Link
                to="/screen"
                onClick={() => setMobileMenuOpen(false)}
                className="block w-full text-center px-6 py-3 bg-[var(--primary-green)] text-white rounded-full hover:bg-[var(--secondary-green)] transition-colors font-medium"
              >
                Check My Child's Risk
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}