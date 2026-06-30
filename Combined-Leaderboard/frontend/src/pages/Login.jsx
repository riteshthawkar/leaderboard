import { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { postJSON, readUser, saveUser, getJSON } from "@/lib/api";
import { safeNext } from "@/lib/utils";

const oauthProviders = [
  { id: "google", provider: "Google", Icon: GoogleIcon },
  { id: "microsoft", provider: "Microsoft", Icon: MicrosoftIcon },
];

const authModes = [
  { id: "login", label: "Sign in", detail: "Existing account" },
  { id: "register", label: "Create account", detail: "New submission access" },
];

function GoogleIcon() {
  return <svg className="oauth-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="#4285F4" d="M22.6 12.2c0-.8-.1-1.5-.2-2.2H12v4.2h5.9c-.3 1.4-1 2.5-2.1 3.3v2.7h3.4c2-1.8 3.4-4.5 3.4-8z" /><path fill="#34A853" d="M12 23c3 0 5.5-1 7.3-2.7l-3.4-2.7c-1 .6-2.2 1-3.9 1-3 0-5.5-2-6.4-4.7H2.1v2.8C3.9 20.4 7.7 23 12 23z" /><path fill="#FBBC05" d="M5.6 13.9c-.2-.6-.4-1.2-.4-1.9s.1-1.3.4-1.9V7.3H2.1C1.4 8.7 1 10.3 1 12s.4 3.3 1.1 4.7l3.5-2.8z" /><path fill="#EA4335" d="M12 5.4c1.6 0 3.1.6 4.2 1.7l3.1-3.1C17.5 2.1 15 1 12 1 7.7 1 3.9 3.6 2.1 7.3l3.5 2.8c.9-2.7 3.4-4.7 6.4-4.7z" /></svg>;
}

function MicrosoftIcon() {
  return <svg className="oauth-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect fill="#F25022" x="2" y="2" width="9.5" height="9.5" /><rect fill="#7FBA00" x="12.5" y="2" width="9.5" height="9.5" /><rect fill="#00A4EF" x="2" y="12.5" width="9.5" height="9.5" /><rect fill="#FFB900" x="12.5" y="12.5" width="9.5" height="9.5" /></svg>;
}

export function Login() {
  const location = useLocation();
  const next = safeNext(new URLSearchParams(location.search).get("next"));
  const [tab, setTab] = useState("login");
  const [message, setMessage] = useState("");
  const [providers, setProviders] = useState(null);

  useEffect(() => {
    let live = true;
    getJSON("/api/auth/providers")
      .then((data) => { if (live) setProviders(Array.isArray(data?.providers) ? data.providers.map((p) => p.id) : []); })
      .catch(() => { if (live) setProviders(oauthProviders.map((p) => p.id)); });
    return () => { live = false; };
  }, []);

  useEffect(() => {
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    const token = hash.get("oauth_token");
    const oauthError = hash.get("oauth_error");
    if (token) {
      saveUser({ username: hash.get("username") || "oauth_user", api_token: token });
      window.history.replaceState(null, "", next);
      window.location.replace(next);
    } else if (oauthError) {
      setMessage(oauthError);
      window.history.replaceState(null, "", `${location.pathname}${location.search}`);
    }
  }, [location.hash, location.pathname, location.search, next]);

  if (readUser()?.api_token) return <Navigate to={next} replace />;
  const visibleProviders = providers === null ? [] : oauthProviders.filter(({ id }) => providers.includes(id));
  const submit = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    setMessage("");
    try {
      const user = await postJSON(`/api/auth/${tab === "login" ? "login" : "register"}`, { username: form.username.value.trim(), password: form.password.value });
      saveUser(user);
      window.location.replace(next);
    } catch (error) {
      setMessage(error.message);
    }
  };
  return <section className="section tight login-section"><div className="container"><div className="login-wrap"><div className="login-intro"><span className="section-tag">Account</span><h1>Sign in</h1><p>Create a free account or sign in to submit model predictions and track your rate-limit quota.</p></div><Card standalone className="auth-card"><div className="auth-mode-switch" role="tablist" aria-label="Account action">{authModes.map(({ id, label, detail }) => <button className={`auth-mode-btn ${tab === id ? "is-active" : ""}`} type="button" role="tab" aria-selected={tab === id} key={id} onClick={() => setTab(id)}><span>{label}</span><small>{detail}</small></button>)}</div>{visibleProviders.length > 0 && <><div className="oauth-actions" aria-label="Single sign-on options">{visibleProviders.map(({ id, provider, Icon }) => <Button asChild variant="ghost" className="oauth-btn" key={id}><a href={`/api/auth/oauth/${id}?next=${encodeURIComponent(next)}`}><Icon />{tab === "login" ? "Sign in" : "Sign up"} with {provider}</a></Button>)}</div><div className="oauth-divider"><span>or continue with username</span></div></>}<form onSubmit={submit}><label className="field"><span>Username</span><input name="username" type="text" autoComplete="username" placeholder={tab === "login" ? "your-username" : "choose-a-username"} required /></label><label className="field"><span>Password {tab === "register" && <em>(min 6 chars)</em>}</span><input name="password" type="password" autoComplete={tab === "login" ? "current-password" : "new-password"} placeholder="••••••••" required /></label><Button type="submit" variant="brand" className="auth-submit-btn">{tab === "login" ? "Sign in to VISTA" : "Create VISTA account"}</Button>{message && <p className="form-msg err">{message}</p>}</form></Card></div></div></section>;
}