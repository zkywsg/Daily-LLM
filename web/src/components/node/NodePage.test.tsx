import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { NodePage } from "./NodePage";

describe("NodePage", () => {
  it("renders node meta (name + year) for known node", () => {
    render(
      <MemoryRouter initialEntries={["/families/01-cnn/05-resnet"]}>
        <Routes>
          <Route
            path="/families/:familyId/:nodeSlug"
            element={<NodePage />}
          />
        </Routes>
      </MemoryRouter>
    );
    expect(screen.getByText(/ResNet/)).toBeInTheDocument();
    expect(screen.getByText(/2015/)).toBeInTheDocument();
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
