import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { NotFound } from "@/pages/NotFound";

describe("not found page", () => {
  it("offers recovery links without hiding the invalid route", () => {
    render(
      <MemoryRouter initialEntries={["/missing-page"]}>
        <NotFound />
      </MemoryRouter>,
    );

    expect(screen.getByRole("heading", { name: "Page not found" })).toBeVisible();
    expect(screen.getByRole("link", { name: /Return to overview/ })).toHaveAttribute("href", "/");
    expect(screen.getByRole("link", { name: /View leaderboard/ })).toHaveAttribute("href", "/leaderboard");
  });
});
