import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { WorkspacePage } from "@/components/WorkspacePage";
import { Button } from "@/components/ui/button";
import { apiUrl, postJSON, saveUser, getJSON, fetchMe, errorMessage, IS_STATIC_DEMO } from "@/lib/api";
import { cn, safeNext } from "@/lib/utils";
import { ui } from "@/lib/styles";

const oauthProviders = [
  { id: "google", provider: "Google", Icon: GoogleIcon },
  { id: "microsoft", provider: "Microsoft", Icon: MicrosoftIcon },
];
const MAX_EMAIL_LENGTH = 254;
const MIN_NEW_PASSWORD_LENGTH = 15;
const MAX_PASSWORD_LENGTH = 128;

function GoogleIcon() {
  return <svg className="size-5" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path fill="#4285F4" d="M22.6 12.2c0-.8-.1-1.5-.2-2.2H12v4.2h5.9c-.3 1.4-1 2.5-2.1 3.3v2.7h3.4c2-1.8 3.4-4.5 3.4-8z" /><path fill="#34A853" d="M12 23c3 0 5.5-1 7.3-2.7l-3.4-2.7c-1 .6-2.2 1-3.9 1-3 0-5.5-2-6.4-4.7H2.1v2.8C3.9 20.4 7.7 23 12 23z" /><path fill="#FBBC05" d="M5.6 13.9c-.2-.6-.4-1.2-.4-1.9s.1-1.3.4-1.9V7.3H2.1C1.4 8.7 1 10.3 1 12s.4 3.3 1.1 4.7l3.5-2.8z" /><path fill="#EA4335" d="M12 5.4c1.6 0 3.1.6 4.2 1.7l3.1-3.1C17.5 2.1 15 1 12 1 7.7 1 3.9 3.6 2.1 7.3l3.5 2.8c.9-2.7 3.4-4.7 6.4-4.7z" /></svg>;
}

function MicrosoftIcon() {
  return <svg className="size-5" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect fill="#F25022" x="2" y="2" width="9.5" height="9.5" /><rect fill="#7FBA00" x="12.5" y="2" width="9.5" height="9.5" /><rect fill="#00A4EF" x="2" y="12.5" width="9.5" height="9.5" /><rect fill="#FFB900" x="12.5" y="12.5" width="9.5" height="9.5" /></svg>;
}

function resetNotice(response, email) {
  let text = `If an account exists for ${response.email || email}, a password reset link has been requested. Check the inbox and spam folder; delivery can take a few minutes.`;
  if (response.dev_reset_url) text += ` (dev link: ${response.dev_reset_url})`;
  return text;
}

export function Login() {
  const location = useLocation();
  const navigate = useNavigate();
  const searchParams = new URLSearchParams(location.search);
  const next = safeNext(searchParams.get("next"));
  const [tab, setTab] = useState(searchParams.get("mode") === "register" ? "register" : "login");
  const [message, setMessage] = useState("");
  const [notice, setNotice] = useState("");
  const [unverifiedEmail, setUnverifiedEmail] = useState("");
  const [providers, setProviders] = useState(null);
  const [resetToken, setResetToken] = useState("");
  const [fieldErrors, setFieldErrors] = useState({});
  const [busy, setBusy] = useState(false);
  const [providerWarning, setProviderWarning] = useState("");

  useEffect(() => {
    if (IS_STATIC_DEMO) {
      setProviders([]);
      return undefined;
    }
    let live = true;
    getJSON("/api/auth/providers")
      .then((data) => { if (live) setProviders(Array.isArray(data?.providers) ? data.providers.map((p) => p.id) : []); })
      .catch((error) => {
        if (!live) return;
        setProviders([]);
        setProviderWarning(`${errorMessage(error, "Single sign-on options could not be checked.")} You can still use email and password.`);
      });
    return () => { live = false; };
  }, []);

  useEffect(() => {
    if (IS_STATIC_DEMO) return undefined;
    const hash = new URLSearchParams(location.hash.replace(/^#/, ""));
    const oauthError = hash.get("oauth_error");
    const verifyError = hash.get("verify_error");
    const verifyToken = hash.get("verify_token");
    const token = hash.get("reset_token");
    const verified = hash.get("verified");
    let live = true;
    if (verifyToken) {
      window.history.replaceState(null, "", `${location.pathname}${location.search}`);
      setBusy(true);
      setMessage("");
      setNotice("Verifying your email address...");
      postJSON("/api/auth/verify", { token: verifyToken })
        .then((response) => {
          if (!live) return;
          saveUser({ email: response.email, csrfToken: response.csrf_token });
          navigate(next, { replace: true });
        })
        .catch((error) => {
          if (!live) return;
          setNotice("");
          setMessage(errorMessage(error, "Your email address could not be verified."));
        })
        .finally(() => { if (live) setBusy(false); });
      return () => { live = false; };
    }
    if (token) {
      setResetToken(token);
      setTab("reset");
      setMessage("");
      setNotice("");
      window.history.replaceState(null, "", `${location.pathname}${location.search}`);
      return undefined;
    }
    if (oauthError || verifyError) {
      setMessage(oauthError || verifyError);
      window.history.replaceState(null, "", `${location.pathname}${location.search}`);
    } else if (verified) {
      setNotice("Email verified. You can now submit models.");
      window.history.replaceState(null, "", `${location.pathname}${location.search}`);
    }
    fetchMe()
      .then((user) => {
        if (!live) return;
        if (user) { saveUser(user); window.location.replace(next); }
      })
      .catch((error) => {
        if (live) setProviderWarning(errorMessage(error, "Account status could not be checked. You can still try to sign in."));
      });
    return () => { live = false; };
  }, [location.hash, location.pathname, location.search, navigate, next]);

  const visibleProviders = providers === null ? [] : oauthProviders.filter(({ id }) => providers.includes(id));
  const showAuthProviders = !IS_STATIC_DEMO && visibleProviders.length > 0 && (tab === "login" || tab === "register");

  const switchMode = (mode) => {
    setTab(mode);
    setMessage("");
    setNotice("");
    setUnverifiedEmail("");
    setFieldErrors({});
    if (mode !== "reset") setResetToken("");
    if (mode === "login" || mode === "register") {
      const params = new URLSearchParams(window.location.search);
      if (mode === "register") params.set("mode", "register");
      else params.delete("mode");
      const query = params.toString();
      window.history.replaceState(null, "", `${location.pathname}${query ? `?${query}` : ""}`);
    }
  };

  const submit = async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    const email = String(formData.get("email") || "").trim();
    const password = String(formData.get("password") || "");
    const confirmPassword = String(formData.get("confirm_password") || "");
    setMessage("");
    setNotice("");
    setUnverifiedEmail("");
    setFieldErrors({});
    setBusy(true);
    try {
      if (tab === "forgot") {
        const response = await postJSON("/api/auth/forgot-password", { email });
        setNotice(resetNotice(response, email));
        return;
      }
      if (tab === "reset") {
        if (password !== confirmPassword) {
          setMessage("Passwords do not match.");
          return;
        }
        await postJSON("/api/auth/reset-password", { token: resetToken, password });
        switchMode("login");
        setNotice("Password updated. Sign in with your new password.");
        return;
      }
      if (tab === "register") {
        const response = await postJSON("/api/auth/register", { email, password });
        let text = `We've sent a verification link to ${response.email || email}. Click it to activate your account, then sign in.`;
        if (response.dev_verify_url) text += ` (dev link: ${response.dev_verify_url})`;
        switchMode("login");
        setNotice(text);
      } else {
        const user = await postJSON("/api/auth/login", { email, password });
        saveUser({ email: user.email, csrfToken: user.csrf_token });
        window.location.replace(next);
      }
    } catch (error) {
      if (error.status === 403 && error.code === "unverified") {
        setUnverifiedEmail(email);
        setMessage(errorMessage(error));
      } else {
        if (error.code === "verification_delivery_failed" && error.data?.email) {
          setUnverifiedEmail(error.data.email);
        }
        setFieldErrors(error.fieldErrors || {});
        setMessage(errorMessage(error));
      }
    } finally {
      setBusy(false);
    }
  };

  const resend = async () => {
    setMessage("");
    setNotice("");
    setBusy(true);
    try {
      const response = await postJSON("/api/auth/resend", { email: unverifiedEmail });
      let text = `If ${unverifiedEmail} belongs to an unverified account, a new verification link has been requested. Check your inbox and spam folder.`;
      if (response.dev_verify_url) text += ` (dev link: ${response.dev_verify_url})`;
      setNotice(text);
      setUnverifiedEmail("");
    } catch (error) {
      setMessage(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  const title =
    IS_STATIC_DEMO ? "Account access" :
      tab === "forgot" ? "Reset password" :
      tab === "reset" ? "Choose a new password" :
        tab === "register" ? "Create an account" :
          "Sign in";
  const submitLabel =
    tab === "forgot" ? "Send reset link" :
      tab === "reset" ? "Update password" :
        tab === "register" ? "Create account" :
          "Sign in";
  const formTitle =
    tab === "forgot" ? "Recovery email" :
      tab === "reset" ? "New password" :
        "Account details";
  const formTag =
    tab === "register" ? "New account" :
      tab === "login" ? "Member access" :
        "Account recovery";
  const description =
    IS_STATIC_DEMO ? "Authentication is not included in this static review build. Explore the leaderboard data and filters without signing in." :
      tab === "register" ? "Create a verified account to submit model responses and track evaluation records." :
      tab === "forgot" ? "Enter the email associated with your account to request a password reset link." :
        tab === "reset" ? "Choose a new password for your account." :
          "Sign in with a verified identity to submit model responses and manage evaluation records.";
  const hasFieldErrors = Object.values(fieldErrors).some(Boolean);

  return (
    <WorkspacePage
      eyebrow="Account access"
      title={title}
      description={description}
    >
      <div className="border-b border-border-strong">
        <div className="mx-auto w-full max-w-[560px] px-6 py-8 sm:px-8">
          <header className="mb-5">
            <span className="text-xs font-medium uppercase text-faint">{formTag}</span>
            <h2 className="mt-3 font-display text-xl font-medium leading-tight text-foreground">{formTitle}</h2>
          </header>

          {IS_STATIC_DEMO ? (
            <div>
              <p className={cn(ui.message, "mt-0 text-muted")} role="status">
                This frozen review build does not connect to the account service. Sign in, registration, password recovery, and submissions are unavailable.
              </p>
              <Button asChild variant="brand" className="mt-5 min-h-12 w-full font-medium">
                <a href="/leaderboard">View leaderboard</a>
              </Button>
            </div>
          ) : (
            <>
              {showAuthProviders && (
                <>
                  <div className="grid gap-2" aria-label="Single sign-on options">
                    {visibleProviders.map(({ id, provider, Icon }) => (
                      <Button asChild variant="ghost" className="min-h-12 w-full font-medium" key={id}>
                        <a href={apiUrl(`/api/auth/oauth/${id}?next=${encodeURIComponent(next)}`)}>
                          <Icon />{tab === "login" ? "Sign in" : "Sign up"} with {provider}
                        </a>
                      </Button>
                    ))}
                  </div>
                  <div className="my-4 flex items-center gap-4 text-xs text-faint before:h-px before:flex-1 before:bg-border after:h-px after:flex-1 after:bg-border"><span>or use email</span></div>
                </>
              )}

              <form onSubmit={submit}>
            {tab !== "reset" && (
              <label className="mb-3 flex flex-col gap-2 text-sm font-medium text-foreground">
                <span>Email</span>
                <input className={cn(ui.input, fieldErrors.email && "border-negative focus:ring-negative-soft")} name="email" type="email" autoComplete="email" maxLength={MAX_EMAIL_LENGTH} placeholder="you@example.com" required aria-invalid={Boolean(fieldErrors.email)} onChange={() => setFieldErrors((current) => ({ ...current, email: "" }))} />
                {fieldErrors.email && <span className={ui.fieldError} role="alert">{fieldErrors.email}</span>}
              </label>
            )}
            {tab !== "forgot" && (
              <label className="mb-3 flex flex-col gap-2 text-sm font-medium text-foreground">
                <span className="flex items-baseline justify-between gap-4">
                  <span>Password</span>
                  {(tab === "register" || tab === "reset") && <span className="text-xs font-normal text-faint">{MIN_NEW_PASSWORD_LENGTH} characters minimum</span>}
                </span>
                <input className={cn(ui.input, fieldErrors.password && "border-negative focus:ring-negative-soft")} name="password" type="password" autoComplete={tab === "login" ? "current-password" : "new-password"} minLength={tab === "login" ? undefined : MIN_NEW_PASSWORD_LENGTH} maxLength={MAX_PASSWORD_LENGTH} placeholder="••••••••" required aria-invalid={Boolean(fieldErrors.password)} onChange={() => setFieldErrors((current) => ({ ...current, password: "" }))} />
                {fieldErrors.password && <span className={ui.fieldError} role="alert">{fieldErrors.password}</span>}
              </label>
            )}
            {tab === "reset" && (
              <label className="mb-3 flex flex-col gap-2 text-sm font-medium text-foreground">
                <span>Confirm password</span>
                <input className={ui.input} name="confirm_password" type="password" autoComplete="new-password" minLength={MIN_NEW_PASSWORD_LENGTH} maxLength={MAX_PASSWORD_LENGTH} placeholder="••••••••" required />
              </label>
            )}
            {tab === "login" && (
              <button type="button" className={cn(ui.linkButton, "mb-4 text-sm font-normal text-muted hover:text-foreground")} onClick={() => switchMode("forgot")}>
                Forgot password?
              </button>
            )}
            <Button type="submit" variant="brand" className="min-h-12 w-full font-medium" disabled={busy}>{busy ? "Please wait..." : submitLabel}</Button>
            {(tab === "forgot" || tab === "reset") && (
              <button type="button" className={cn(ui.linkButton, "mt-5 text-sm font-normal text-muted hover:text-foreground")} onClick={() => switchMode("login")}>
                Back to sign in
              </button>
            )}
            {providerWarning && !message && !notice && <p className={cn(ui.message, ui.messageError)} role="status">{providerWarning}</p>}
            {notice && <p className={cn(ui.message, ui.messageSuccess)} role="status">{notice}</p>}
            {message && !hasFieldErrors && <p className={cn(ui.message, ui.messageError)} role="alert">{message}</p>}
            {unverifiedEmail && <button type="button" className={cn(ui.linkButton, "mt-3 text-sm")} onClick={resend} disabled={busy}>Resend verification email</button>}
              </form>

              {(tab === "login" || tab === "register") && (
                <p className="mt-6 text-sm text-muted">
                  {tab === "register" ? "Already have an account?" : "New to MS VISTA?"}{" "}
                  <button type="button" className={cn(ui.linkButton, "font-medium text-foreground")} onClick={() => switchMode(tab === "register" ? "login" : "register")}>
                    {tab === "register" ? "Sign in" : "Create an account"}
                  </button>
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </WorkspacePage>
  );
}
