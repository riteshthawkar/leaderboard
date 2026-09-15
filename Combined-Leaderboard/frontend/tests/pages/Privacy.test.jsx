import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api", () => ({ AUTH_TRANSPORT: "bearer" }));
import { Privacy } from "@/pages/Privacy";

afterEach(cleanup);

describe("privacy notice", () => {
  it("discloses current hosting, public raw outputs, deletion limits, and essential storage", () => {
    render(<MemoryRouter><Privacy /></MemoryRouter>);
    expect(screen.getByText(/GitHub Pages for the frontend and Oracle Cloud Infrastructure/)).toBeInTheDocument();
    expect(screen.queryByText(/Hugging Face provides/)).not.toBeInTheDocument();
    expect(screen.queryByText(/traces are not required or published/)).not.toBeInTheDocument();
    expect(screen.getByText(/Model outputs may contain reasoning text/)).toBeInTheDocument();
    expect(screen.getByText(/not automatically scrubbed/)).toBeInTheDocument();
    expect(screen.getByText(/does not independently verify a person's age/)).toBeInTheDocument();
    expect(screen.getByText(/reapply subsequent account-deletion requests/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Browser storage inventory" })).toBeInTheDocument();
    for (const key of ["lb_refresh_token_v1", "lb_user", "lb_csrf_token", "vci-theme", "ms_vista_session", "vista_oauth_state"]) {
      expect(screen.getByText(key)).toBeInTheDocument();
    }
    expect(screen.getByText(/This frontend uses bearer authentication/)).toBeInTheDocument();
  });
});
