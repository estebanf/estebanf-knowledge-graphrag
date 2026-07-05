import { FormEvent, useState } from "react";
import { useAuth } from "./AuthContext";

export function Login() {
  const { login, state } = useAuth();
  const sessionExpired = state.status === "anonymous" && state.sessionExpired;
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      await login(username, password);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Login failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-100 px-4">
      <div className="login-card">
        <div className="login-card__header">
          <h1 className="login-card__title">estebanf&apos;s RAG</h1>
          <p className="login-card__subtitle">Knowledge graph retrieval</p>
        </div>
        {sessionExpired ? (
          <p className="login-card__notice login-card__notice--expired" role="alert">
            Your session expired — sign in again
          </p>
        ) : null}
        <form onSubmit={onSubmit} className="login-card__form">
          <div className="space-y-2">
            <label className="block text-sm" htmlFor="username">
              Username
            </label>
            <input
              id="username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
              required
              autoFocus
            />
          </div>
          <div className="space-y-2">
            <label className="block text-sm" htmlFor="password">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full rounded border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
              required
            />
          </div>
          {error && <p className="text-sm text-rose-400" role="alert">{error}</p>}
          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded bg-emerald-600 px-3 py-2 text-sm font-semibold hover:bg-emerald-500 disabled:opacity-50"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
