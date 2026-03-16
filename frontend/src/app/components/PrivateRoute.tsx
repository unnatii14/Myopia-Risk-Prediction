import { Navigate, Outlet, useLocation } from "react-router";
import { useAuth } from "../context/AuthContext";

export default function PrivateRoute() {
  const { user } = useAuth();
  const location = useLocation();

  if (!user) {
    // Redirect to login, remembering where they came from
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <Outlet />;
}
