import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { FamilyPage } from "./FamilyPage";

function renderFamily(familyId: string) {
  return render(
    <MemoryRouter initialEntries={[`/families/${familyId}`]}>
      <Routes>
        <Route path="/families/:familyId" element={<FamilyPage />} />
        <Route path="/404" element={<div>404 page</div>} />
      </Routes>
    </MemoryRouter>
  );
}

describe("FamilyPage", () => {
  it("renders CNN family with sub-timeline", () => {
    renderFamily("01-cnn");
    expect(screen.getByText("子时间线")).toBeInTheDocument();
    expect(screen.getAllByText(/ResNet/).length).toBeGreaterThan(0);
  });

  it("shows '待补充' for empty family (02-rnn-lstm)", () => {
    renderFamily("02-rnn-lstm");
    expect(screen.getAllByText(/待补充/).length).toBeGreaterThan(0);
  });

  it("redirects to 404 for unknown family", () => {
    renderFamily("invalid-family");
    expect(screen.getByText("404 page")).toBeInTheDocument();
  });
});
