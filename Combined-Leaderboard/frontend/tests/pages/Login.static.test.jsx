import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";
import { cleanup, render, screen } from "@testing-library/react";
import { Login } from "@/pages/Login";


const apiMocks = vi.hoisted(() => ({
  getJSON: vi.fn(),
  fetchMe: vi.fn(),
  postJSON: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  IS_STATIC_DEMO: true,
  apiUrl: (path) => path,
  getJSON: apiMocks.getJSON,
  fetchMe: apiMocks.fetchMe,
  postJSON: apiMocks.postJSON,
  saveUser: vi.fn(),
  errorMessage: (error, fallback = "The action could not be completed.") => error?.message || fallback,
}));


describe("static authentication workspace", () => {
  afterEach(() => cleanup());

  it("shows a review-only state without active authentication controls", () => {
    render(
      <MemoryRouter initialEntries={["/login?mode=register#verify_token=unused"]}>
        <Login />
      </MemoryRouter>,
    );

    expect(screen.getByRole("heading", { name: "Account access" })).toBeVisible();
    expect(screen.getByRole("status")).toHaveTextContent("does not connect to the account service");
    expect(screen.getByRole("link", { name: "View leaderboard" })).toHaveAttribute("href", "/leaderboard");
    expect(screen.queryByRole("button", { name: /create account|sign in|reset/i })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Email")).not.toBeInTheDocument();
    expect(apiMocks.getJSON).not.toHaveBeenCalled();
    expect(apiMocks.fetchMe).not.toHaveBeenCalled();
    expect(apiMocks.postJSON).not.toHaveBeenCalled();
  });
});
