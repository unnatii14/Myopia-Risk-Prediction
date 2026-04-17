import { Eye, Menu, X, LogOut, ChevronDown, Clock, Activity, Ruler } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useAuth } from "../context/AuthContext";

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const [toolsMenuOpen, setToolsMenuOpen] = useState(false);
  const userMenuRef  = useRef<HTMLDivElement>(null);
  const toolsMenuRef = useRef<HTMLDivElement>(null);

  // Close dropdowns on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
      if (toolsMenuRef.current && !toolsMenuRef.current.contains(e.target as Node)) {
        setToolsMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSectionClick = (sectionId: string) => {
    setMobileMenuOpen(false);
    // Logged-in users don't see the landing page sections — go to about/faq instead
    if (user) {
      navigate(sectionId === "research" ? "/about" : "/faq");
      return;
    }
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
            <Link to={user ? "/dashboard" : "/"} className="flex items-center gap-2 group">
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
                {user ? "FAQ" : "How It Works"}
              </button>
              <button
                onClick={() => handleSectionClick("research")}
                className="text-[var(--text-dark)] hover:text-[var(--primary-green)] transition-colors"
              >
                {user ? "About" : "Research"}
              </button>

              {/* Tools dropdown */}
              <div className="relative" ref={toolsMenuRef}>
                <button
                  onClick={() => setToolsMenuOpen(v => !v)}
                  className="flex items-center gap-1 text-[var(--text-dark)] hover:text-[var(--primary-green)] transition-colors"
                >
                  Calculator & IMI
                  <ChevronDown className={`w-3.5 h-3.5 transition-transform ${toolsMenuOpen ? "rotate-180" : ""}`} />
                </button>
                <AnimatePresence>
                  {toolsMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: 8, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 8, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      className="absolute left-0 top-full mt-2 w-72 bg-white rounded-2xl shadow-xl border border-[var(--border)] overflow-hidden z-50"
                    >
                      <Link
                        to="/screen"
                        onClick={() => setToolsMenuOpen(false)}
                        className="flex items-center gap-3 px-4 py-3 hover:bg-[var(--background-mint)] transition-colors"
                      >
                        <div className="w-8 h-8 rounded-full bg-[var(--primary-green)]/10 flex items-center justify-center flex-shrink-0">
                          <Activity className="w-4 h-4" style={{ color: "var(--primary-green)" }} />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-[var(--text-dark)]">Myopia Calculator</p>
                          <p className="text-xs text-[var(--text-muted)]">AI risk score in 3 minutes</p>
                        </div>
                      </Link>
                      <div className="h-px bg-[var(--border)]" />
                      <Link
                        to="/progression"
                        onClick={() => setToolsMenuOpen(false)}
                        className="flex items-center gap-3 px-4 py-3 hover:bg-[var(--background-mint)] transition-colors"
                      >
                        <div className="w-8 h-8 rounded-full bg-[var(--primary-green)]/10 flex items-center justify-center flex-shrink-0">
                          <Ruler className="w-4 h-4" style={{ color: "var(--primary-green)" }} />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-[var(--text-dark)]">Myopia Progression Calculator</p>
                          <p className="text-xs text-[var(--text-muted)]">Project SE (diopters) to age 18</p>
                        </div>
                      </Link>
                      <div className="h-px bg-[var(--border)]" />
                      <Link
                        to="/axial"
                        onClick={() => setToolsMenuOpen(false)}
                        className="flex items-center gap-3 px-4 py-3 hover:bg-[var(--background-mint)] transition-colors"
                      >
                        <div className="w-8 h-8 rounded-full bg-[var(--primary-green)]/10 flex items-center justify-center flex-shrink-0">
                          <Ruler className="w-4 h-4" style={{ color: "var(--primary-green)" }} />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-[var(--text-dark)]">Axial Elongation Progression</p>
                          <p className="text-xs text-[var(--text-muted)]">Project axial length (mm) to age 18</p>
                        </div>
                      </Link>
                      <div className="h-px bg-[var(--border)]" />
                      <Link
                        to="/onset"
                        onClick={() => setToolsMenuOpen(false)}
                        className="flex items-center gap-3 px-4 py-3 hover:bg-[var(--background-mint)] transition-colors"
                      >
                        <div className="w-8 h-8 rounded-full bg-[var(--primary-green)]/10 flex items-center justify-center flex-shrink-0">
                          <Clock className="w-4 h-4" style={{ color: "var(--primary-green)" }} />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-[var(--text-dark)]">Onset Predictor</p>
                          <p className="text-xs text-[var(--text-muted)]">When will myopia start?</p>
                        </div>
                      </Link>
                      <div className="h-px bg-[var(--border)]" />
                      <Link
                        to="/image-predictor"
                        onClick={() => setToolsMenuOpen(false)}
                        className="flex items-center gap-3 px-4 py-3 hover:bg-[var(--background-mint)] transition-colors"
                      >
                        <div className="w-8 h-8 rounded-full bg-[var(--primary-green)]/10 flex items-center justify-center flex-shrink-0">
                          <Eye className="w-4 h-4" style={{ color: "var(--primary-green)" }} />
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-[var(--text-dark)]">Image Classifier</p>
                          <p className="text-xs text-[var(--text-muted)]">Upload image and predict myopia</p>
                        </div>
                      </Link>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
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
              {user ? (
                /* ---- Logged-in user avatar + dropdown ---- */
                <div className="relative" ref={userMenuRef}>
                  <button
                    onClick={() => setUserMenuOpen((v) => !v)}
                    className="hidden md:flex items-center gap-2 px-3 py-2 rounded-full hover:bg-[var(--background-mint)] transition-colors"
                  >
                    <div className="w-8 h-8 rounded-full bg-[var(--primary-green)] flex items-center justify-center text-white text-sm font-bold">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm font-semibold text-[var(--text-dark)] max-w-[120px] truncate">
                      {user.name}
                    </span>
                    <ChevronDown className={`w-4 h-4 text-[var(--text-muted)] transition-transform ${userMenuOpen ? "rotate-180" : ""}`} />
                  </button>

                  <AnimatePresence>
                    {userMenuOpen && (
                      <motion.div
                        initial={{ opacity: 0, y: 8, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 8, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 top-full mt-2 w-52 bg-white rounded-2xl shadow-xl border border-[var(--border)] overflow-hidden z-50"
                      >
                        <div className="px-4 py-3 border-b border-[var(--border)]">
                          <p className="text-sm font-bold text-[var(--text-dark)] truncate">{user.name}</p>
                          <p className="text-xs text-[var(--text-muted)] truncate">{user.email}</p>
                        </div>
                        <Link
                          to="/dashboard"
                          onClick={() => setUserMenuOpen(false)}
                          className="flex items-center gap-2 w-full px-4 py-3 text-sm text-[var(--text-dark)] hover:bg-[var(--background-mint)] transition-colors"
                        >
                          <Activity className="w-4 h-4" style={{ color: "var(--primary-green)" }} />
                          Dashboard
                        </Link>
                        <div className="h-px bg-[var(--border)]" />
                        <button
                          onClick={() => { setUserMenuOpen(false); logout(); navigate("/"); }}
                          className="flex items-center gap-2 w-full px-4 py-3 text-sm text-[var(--warning-coral)] hover:bg-red-50 transition-colors"
                        >
                          <LogOut className="w-4 h-4" />
                          Log Out
                        </button>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ) : (
                /* ---- Guest links ---- */
                <>
                  <Link
                    to="/login"
                    className="hidden md:block text-[var(--text-dark)] hover:text-[var(--primary-green)] transition-colors font-medium text-sm"
                  >
                    Log In
                  </Link>
                  <Link
                    to="/signup"
                    className="hidden md:block px-5 py-2.5 border-2 border-[var(--primary-green)] text-[var(--primary-green)] hover:bg-[var(--primary-green)] hover:text-white rounded-full transition-colors font-medium text-sm"
                  >
                    Sign Up
                  </Link>
                </>
              )}
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
              {/* Tools section in mobile */}
              <div className="border border-[var(--border)] rounded-2xl overflow-hidden">
                <p className="px-4 py-2 text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider bg-[var(--background-mint)]">
                  Calculator & IMI
                </p>
                <Link
                  to="/screen"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] transition-colors border-t border-[var(--border)]"
                >
                  <Activity className="w-4 h-4 flex-shrink-0" style={{ color: "var(--primary-green)" }} />
                  <div>
                    <p className="text-sm font-semibold">Myopia Calculator</p>
                    <p className="text-xs text-[var(--text-muted)]">AI risk score in 3 minutes</p>
                  </div>
                </Link>
                <Link
                  to="/progression"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] transition-colors border-t border-[var(--border)]"
                >
                  <Ruler className="w-4 h-4 flex-shrink-0" style={{ color: "var(--primary-green)" }} />
                  <div>
                    <p className="text-sm font-semibold">Myopia Progression Calculator</p>
                    <p className="text-xs text-[var(--text-muted)]">Project SE (diopters) to age 18</p>
                  </div>
                </Link>
                <Link
                  to="/axial"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] transition-colors border-t border-[var(--border)]"
                >
                  <Ruler className="w-4 h-4 flex-shrink-0" style={{ color: "var(--primary-green)" }} />
                  <div>
                    <p className="text-sm font-semibold">Axial Elongation Progression</p>
                    <p className="text-xs text-[var(--text-muted)]">Project axial length (mm) to age 18</p>
                  </div>
                </Link>
                <Link
                  to="/onset"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] transition-colors border-t border-[var(--border)]"
                >
                  <Clock className="w-4 h-4 flex-shrink-0" style={{ color: "var(--primary-green)" }} />
                  <div>
                    <p className="text-sm font-semibold">Onset Predictor</p>
                    <p className="text-xs text-[var(--text-muted)]">When will myopia start?</p>
                  </div>
                </Link>
                <Link
                  to="/image-predictor"
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center gap-3 px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] transition-colors border-t border-[var(--border)]"
                >
                  <Eye className="w-4 h-4 flex-shrink-0" style={{ color: "var(--primary-green)" }} />
                  <div>
                    <p className="text-sm font-semibold">Image Classifier</p>
                    <p className="text-xs text-[var(--text-muted)]">Upload image and predict myopia</p>
                  </div>
                </Link>
              </div>
              {user ? (
                /* Mobile: logged-in user info + logout */
                <div className="border border-[var(--border)] rounded-2xl px-4 py-3 space-y-2">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-full bg-[var(--primary-green)] flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-bold text-[var(--text-dark)] truncate">{user.name}</p>
                      <p className="text-xs text-[var(--text-muted)] truncate">{user.email}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => { setMobileMenuOpen(false); logout(); navigate("/"); }}
                    className="flex items-center gap-2 text-sm text-[var(--warning-coral)] font-medium w-full"
                  >
                    <LogOut className="w-4 h-4" />
                    Log Out
                  </button>
                </div>
              ) : (
                <>
                  <Link
                    to="/login"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block w-full text-left px-4 py-3 text-[var(--text-dark)] hover:bg-[var(--background-mint)] rounded-xl transition-colors"
                  >
                    Log In
                  </Link>
                  <Link
                    to="/signup"
                    onClick={() => setMobileMenuOpen(false)}
                    className="block w-full text-center px-6 py-3 border-2 border-[var(--primary-green)] text-[var(--primary-green)] hover:bg-[var(--primary-green)] hover:text-white rounded-full transition-colors font-medium"
                  >
                    Sign Up
                  </Link>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}