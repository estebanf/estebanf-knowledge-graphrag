export type MeResponse = {
  username: string;
  kind: string;
  scopes: string[];
};

export async function login(username: string, password: string): Promise<{ username: string }> {
  const response = await fetch("/api/auth/login", {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (response.status === 401) throw new Error("Invalid username or password");
  if (!response.ok) throw new Error(`Login failed: ${response.status}`);
  return (await response.json()) as { username: string };
}

export async function logout(): Promise<void> {
  await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
}

export async function fetchMe(): Promise<MeResponse | null> {
  const response = await fetch("/api/auth/me", { credentials: "include" });
  if (response.status === 401) return null;
  if (!response.ok) throw new Error(`Failed to fetch user: ${response.status}`);
  return (await response.json()) as MeResponse;
}
