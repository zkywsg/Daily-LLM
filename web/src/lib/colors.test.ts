import { describe, it, expect } from "vitest";
import { familyColorVar, familyColorToken } from "./colors";

describe("colors", () => {
  it("familyColorVar wraps in var()", () => {
    expect(familyColorVar("01-cnn")).toBe("var(--family-01)");
    expect(familyColorVar("15-reasoning-o1-r1")).toBe("var(--family-15)");
  });

  it("familyColorToken returns bare variable name", () => {
    expect(familyColorToken("05-transformer")).toBe("--family-05");
  });
});
