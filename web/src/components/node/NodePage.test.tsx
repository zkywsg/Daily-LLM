import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { NodePage } from "./NodePage";

// Every real node is now a golden sample (see web/src/components/node/golden/index.ts),
// so the non-golden fallback-rendering path is only reachable by stubbing the registry
// empty here — this decouples the test from which nodes happen to be golden.
vi.mock("./golden", () => ({ goldenSamples: {} }));

describe("NodePage", () => {
  it("renders node meta (name + year) for known non-golden node", () => {
    render(
      <MemoryRouter initialEntries={["/families/01-cnn/03-vgg"]}>
        <Routes>
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
        </Routes>
      </MemoryRouter>
    );
    expect(screen.getByRole("heading", { name: /VGG.*2014/ })).toBeInTheDocument();
  });

  it("redirects to 404 for unknown node", () => {
    render(
      <MemoryRouter initialEntries={["/families/01-cnn/nonexistent-node"]}>
        <Routes>
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
          <Route path="/404" element={<div>404 page</div>} />
        </Routes>
      </MemoryRouter>
    );
    expect(screen.getByText("404 page")).toBeInTheDocument();
  });
});
