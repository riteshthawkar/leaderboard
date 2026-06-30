export async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

export async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

export function readUser() {
  try {
    return JSON.parse(localStorage.getItem("lb_user") || "null");
  } catch {
    return null;
  }
}

export function saveUser(user) {
  localStorage.setItem("lb_user", JSON.stringify(user));
  window.dispatchEvent(new CustomEvent("lb_auth", { detail: user }));
}

export function clearUser() {
  localStorage.removeItem("lb_user");
}