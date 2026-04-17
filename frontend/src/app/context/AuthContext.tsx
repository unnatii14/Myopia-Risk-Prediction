import { createContext, useContext, useState } from "react";

interface User {
  name: string;
  email: string;
  childName?: string;
  token: string;
}

interface AuthContextType {
  user: User | null;
  login: (name: string, email: string, token: string, childName?: string, rememberMe?: boolean) => void;
  logout: () => void;
  isTokenValid: () => boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

const STORAGE_KEY = "myopia_auth_user";
const STORAGE_TIMESTAMP_KEY = "myopia_auth_timestamp";
const TOKEN_MAX_AGE_HOURS = 24;

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      // Check localStorage first (remember me / persistent login)
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const timestamp = localStorage.getItem(STORAGE_TIMESTAMP_KEY);
        if (timestamp) {
          const lastLoginTime = parseInt(timestamp, 10);
          const ageInHours = (Date.now() - lastLoginTime) / (1000 * 60 * 60);
          if (ageInHours <= TOKEN_MAX_AGE_HOURS) {
            return JSON.parse(stored) as User;
          }
          // Expired — clear it
          localStorage.removeItem(STORAGE_KEY);
          localStorage.removeItem(STORAGE_TIMESTAMP_KEY);
        }
      }

      // Fall back to sessionStorage (non-remember-me login)
      const sessionStored = sessionStorage.getItem(STORAGE_KEY);
      if (sessionStored) {
        return JSON.parse(sessionStored) as User;
      }

      return null;
    } catch {
      return null;
    }
  });

  const isTokenValid = () => {
    if (!user) return false;
    const timestamp = localStorage.getItem(STORAGE_TIMESTAMP_KEY);
    if (!timestamp) return false;

    const lastLoginTime = parseInt(timestamp, 10);
    const ageInHours = (Date.now() - lastLoginTime) / (1000 * 60 * 60);
    return ageInHours <= TOKEN_MAX_AGE_HOURS;
  };

  const login = (name: string, email: string, token: string, childName?: string, rememberMe?: boolean) => {
    const u: User = { name, email, token, childName };
    setUser(u);

    if (rememberMe) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(u));
      localStorage.setItem(STORAGE_TIMESTAMP_KEY, Date.now().toString());
    } else {
      // Session-only login: store temporarily but mark with expired timestamp
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(u));
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_TIMESTAMP_KEY);
    sessionStorage.removeItem(STORAGE_KEY);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, isTokenValid }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside AuthProvider");
  return ctx;
}
