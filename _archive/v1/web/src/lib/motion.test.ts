import { describe, it, expect } from "vitest";
import { duration, ease, fadeUp } from "./motion";

describe("motion presets", () => {
  it("duration values are positive seconds in ascending order", () => {
    expect(duration.fast).toBeGreaterThan(0);
    expect(duration.fast).toBeLessThan(duration.base);
    expect(duration.base).toBeLessThan(duration.slow);
  });

  it("ease tuples are 4-tuples", () => {
    expect(ease.out).toHaveLength(4);
    expect(ease.inOut).toHaveLength(4);
    expect(ease.spring).toHaveLength(4);
  });

  it("fadeUp variant has initial/animate/exit", () => {
    expect(fadeUp.initial).toBeDefined();
    expect(fadeUp.animate).toBeDefined();
    expect(fadeUp.exit).toBeDefined();
  });
});
