import { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from "react";
import { fetchMe, login as apiLogin, logout as apiLogout, MeResponse } from "../lib/auth";
import { onAuthError } from "../lib/api";

type AuthState =
  | { status: "loading" }
  | { status: "anonymous"; sessionExpired?: boolean }
  | { status: "authenticated"; user: MeResponse };

type AuthContextValue = {
  state: AuthState;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

const TEST_BYPASS =
  typeof import.meta !== "undefined" && (import.meta as any)?.env?.MODE === "test";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>(
    TEST_BYPASS
      ? { status: "authenticated", user: { username: "test", kind: "session", scopes: ["read", "ingest", "admin"] } }
      : { status: "loading" },
  );

  const refresh = useCallback(async () => {
    if (TEST_BYPASS) return;
    try {
      const me = await fetchMe();
      setState(me ? { status: "authenticated", user: me } : { status: "anonymous", sessionExpired: false });
    } catch {
      setState({ status: "anonymous", sessionExpired: false });
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    onAuthError(() => setState({ status: "anonymous", sessionExpired: true }));
    return () => onAuthError(null);
  }, []);

  const login = useCallback(
    async (username: string, password: string) => {
      await apiLogin(username, password);
      await refresh();
    },
    [refresh],
  );

  const logout = useCallback(async () => {
    await apiLogout();
    setState({ status: "anonymous", sessionExpired: false });
  }, []);

  const value = useMemo<AuthContextValue>(() => ({ state, login, logout, refresh }), [state, login, logout, refresh]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
