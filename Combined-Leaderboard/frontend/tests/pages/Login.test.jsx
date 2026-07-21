import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Login } from "@/pages/Login";


const apiMocks = vi.hoisted(() => ({
  getJSON: vi.fn(),
  fetchMe: vi.fn(),
  postJSON: vi.fn(),
  saveUser: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  IS_STATIC_DEMO: false,
  apiUrl: (path) => `http://api.test${path}`,
  getJSON: apiMocks.getJSON,
  fetchMe: apiMocks.fetchMe,
  postJSON: apiMocks.postJSON,
  saveUser: apiMocks.saveUser,
  errorMessage: (error, fallback = "The action could not be completed.") => error?.message || fallback,
}));


function renderLogin(entry = "/login") {
  return render(
    <MemoryRouter initialEntries={[entry]}>
      <Login />
    </MemoryRouter>,
  );
}


describe("authentication workspace", () => {
  afterEach(() => cleanup());

  beforeEach(() => {
    apiMocks.getJSON.mockResolvedValue({ providers: [] });
    apiMocks.fetchMe.mockResolvedValue(null);
    apiMocks.postJSON.mockReset();
    apiMocks.saveUser.mockReset();
    window.history.replaceState(null, "", "/");
  });

  it("applies production password and email bounds without blocking existing logins", async () => {
    const user = userEvent.setup();
    renderLogin("/login?mode=register");

    const email = screen.getByLabelText("Email");
    const password = screen.getByLabelText(/^Password/);
    expect(email).toHaveAttribute("maxlength", "254");
    expect(password).toHaveAttribute("minlength", "15");
    expect(password).toHaveAttribute("maxlength", "128");

    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(screen.getByLabelText(/^Password/)).not.toHaveAttribute("minlength");
    expect(screen.getByLabelText(/^Password/)).toHaveAttribute("maxlength", "128");
  });

  it("registers an account and shows the verification delivery state", async () => {
    const user = userEvent.setup();
    apiMocks.postJSON.mockResolvedValue({
      email: "new@example.com",
      dev_verify_url: "http://api.test/api/auth/verify?token=verification-token",
    });
    renderLogin("/login?mode=register");

    await user.type(screen.getByLabelText("Email"), "new@example.com");
    await user.type(screen.getByLabelText(/^Password/), "violet telescope cedar glacier");
    await user.click(screen.getByRole("button", { name: "Create account" }));

    await waitFor(() => expect(apiMocks.postJSON).toHaveBeenCalledWith(
      "/api/auth/register",
      { email: "new@example.com", password: "violet telescope cedar glacier" },
    ));
    expect(await screen.findByText(/We've sent a verification link to new@example.com/)).toBeVisible();
    expect(screen.getByRole("heading", { name: "Sign in" })).toBeVisible();
  });

  it("consumes email verification tokens from the URL fragment", async () => {
    apiMocks.postJSON.mockResolvedValue({
      status: "verified",
      email: "verified@example.com",
      csrf_token: "csrf-token",
    });

    renderLogin("/login#verify_token=verification-token");

    await waitFor(() => expect(apiMocks.postJSON).toHaveBeenCalledWith(
      "/api/auth/verify",
      { token: "verification-token" },
    ));
    expect(apiMocks.saveUser).toHaveBeenCalledWith({
      email: "verified@example.com",
      csrfToken: "csrf-token",
    });
    expect(window.location.hash).toBe("");
  });

  it("offers a resend action for an unverified account", async () => {
    const user = userEvent.setup();
    apiMocks.postJSON
      .mockRejectedValueOnce({
        status: 403,
        code: "unverified",
        message: "Your email address has not been verified.",
      })
      .mockResolvedValueOnce({ status: "verification_requested" });
    renderLogin();

    await user.type(screen.getByLabelText("Email"), "pending@example.com");
    await user.type(screen.getByLabelText(/^Password/), "violet telescope cedar glacier");
    await user.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("has not been verified");

    await user.click(screen.getByRole("button", { name: "Resend verification email" }));
    await waitFor(() => expect(apiMocks.postJSON).toHaveBeenLastCalledWith(
      "/api/auth/resend",
      { email: "pending@example.com" },
    ));
    expect(await screen.findByText(/a new verification link has been requested/i)).toBeVisible();
  });

  it("keeps forgot-password responses generic", async () => {
    const user = userEvent.setup();
    apiMocks.postJSON.mockResolvedValue({
      status: "reset_requested",
      email: "unknown@example.com",
    });
    renderLogin();

    await user.click(screen.getByRole("button", { name: "Forgot password?" }));
    await user.type(screen.getByLabelText("Email"), "unknown@example.com");
    await user.click(screen.getByRole("button", { name: "Send reset link" }));

    expect(await screen.findByText(/If an account exists for unknown@example.com/)).toBeVisible();
    expect(apiMocks.postJSON).toHaveBeenCalledWith(
      "/api/auth/forgot-password",
      { email: "unknown@example.com" },
    );
  });

  it("does not submit mismatched replacement passwords", async () => {
    const user = userEvent.setup();
    renderLogin("/login#reset_token=reset-token-1");

    await user.type(await screen.findByLabelText(/^Password/), "violet telescope cedar glacier");
    await user.type(screen.getByLabelText("Confirm password"), "different meadow lantern phrase");
    await user.click(screen.getByRole("button", { name: "Update password" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Passwords do not match");
    expect(apiMocks.postJSON).not.toHaveBeenCalled();
  });

  it("renders only OAuth providers reported by the backend", async () => {
    apiMocks.getJSON.mockResolvedValue({ providers: [{ id: "microsoft" }] });
    renderLogin();

    const microsoft = await screen.findByRole("link", { name: "Sign in with Microsoft" });
    expect(microsoft).toHaveAttribute("href", expect.stringContaining("/api/auth/oauth/microsoft"));
    expect(screen.queryByRole("link", { name: "Sign in with Google" })).not.toBeInTheDocument();
  });
});
