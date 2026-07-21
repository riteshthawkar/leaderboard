import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TabBar } from "@/components/ui/tabs";

const tabs = [
  { id: "all", label: "All visual benchmarks" },
  { id: "perception", label: "Do You See Me" },
  { id: "cognition", label: "Mind's Eye" },
];

describe("TabBar", () => {
  it("supports click and arrow-key selection", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<TabBar tabs={tabs} active="all" onChange={onChange} />);

    await user.click(screen.getByRole("tab", { name: "Do You See Me" }));
    expect(onChange).toHaveBeenCalledWith("perception");

    const first = screen.getByRole("tab", { name: "All visual benchmarks" });
    first.focus();
    await user.keyboard("{ArrowRight}");
    expect(onChange).toHaveBeenLastCalledWith("perception");
    expect(screen.getByRole("tab", { name: "Do You See Me" })).toHaveFocus();
  });
});
