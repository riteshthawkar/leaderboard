import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Layout } from "@/components/Layout";


const apiMocks = vi.hoisted(() => ({
  getJSON: vi.fn(),
  fetchMe: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  IS_STATIC_DEMO: true,
  getJSON: apiMocks.getJSON,
  fetchMe: apiMocks.fetchMe,
  logout: vi.fn(),
  errorMessage: (error, fallback) => error?.message || fallback,
}));


describe("static layout status", () => {
  afterEach(() => cleanup());

  it("labels frozen data without claiming that a backend is online", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<main>Overview content</main>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByTitle("Static data status")).toHaveTextContent("Snapshot");
    expect(screen.queryByText("Online")).not.toBeInTheDocument();
    expect(apiMocks.getJSON).not.toHaveBeenCalled();
    expect(apiMocks.fetchMe).not.toHaveBeenCalled();
  });
});
