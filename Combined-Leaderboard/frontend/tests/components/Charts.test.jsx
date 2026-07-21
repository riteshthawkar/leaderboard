import { describe, expect, it } from "vitest";
import { layoutScatterPoints } from "@/components/Charts";

describe("scatter label layout", () => {
  it("separates coincident points and labels deterministically", () => {
    const points = [
      { key: "a", label: "Alpha", x: 0.4, y: 0.3 },
      { key: "b", label: "Beta", x: 0.4, y: 0.3 },
      { key: "c", label: "Gamma", x: 0.405, y: 0.302 },
    ];
    const layout = layoutScatterPoints(points, {
      xScale: (value) => value * 500,
      yScale: (value) => 400 - value * 400,
      innerWidth: 500,
      innerHeight: 400,
    });

    expect(new Set(layout.map((point) => `${point.screenX},${point.screenY}`)).size).toBe(3);
    expect(new Set(layout.map((point) => point.labelY)).size).toBe(3);
    expect(layout.every((point) => point.labelY >= 8 && point.labelY <= 392)).toBe(true);
  });
});
