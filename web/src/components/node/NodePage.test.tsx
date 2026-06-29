import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { NodePage } from "./NodePage";

describe("NodePage", () => {
  it("renders node meta (name + year) for known non-golden node", () => {
    // Use LeNet (non-golden), since 02-alexnet now routes to lazy-loaded NodePageAlexNet
    render(
      <MemoryRouter initialEntries={["/families/01-cnn/01-lenet"]}>
        <Routes>
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
        </Routes>
      </MemoryRouter>
    );
    expect(screen.getByText(/LeNet/)).toBeInTheDocument();
    expect(screen.getByText(/1998/)).toBeInTheDocument();
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
