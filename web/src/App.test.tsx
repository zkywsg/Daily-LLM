import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { Layout } from "./components/ui/Layout";
import { NotFoundPage } from "./components/ui/NotFoundPage";
import { HomePage } from "./components/home/HomePage";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Layout>
    </MemoryRouter>
  );
}

describe("App router", () => {
  it("renders HomePage at /", () => {
    renderAt("/");
    expect(screen.getAllByText(/被逼出来的历史/)[0]).toBeInTheDocument();
  });

  it("renders 404 at unknown route", () => {
    renderAt("/nonexistent");
    expect(screen.getByText("404")).toBeInTheDocument();
  });
});
