import { Navigate } from "react-router";
import { useAuth } from "../context/AuthContext";
import Landing from "../pages/Landing";

/**
 * At the root path ("/"):
 *  - Logged-in user  → redirect to /dashboard
 *  - Guest           → show Landing page
 */
export default function HomeRedirect() {
  const { user } = useAuth();

  if (user) {
    return <Navigate to="/dashboard" replace />;
  }

  return <Landing />;
}
