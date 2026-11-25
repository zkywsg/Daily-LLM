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

  it("renders sub-timeline for word-embedding family", () => {
    // 全 15 家族已无空家族;之前的"待补充"分支保留在组件里作为兜底,
    // 但当前数据下无家族能触发,所以这里改为验证 word-embedding 正常渲染
    renderFamily("03-word-embedding");
    expect(screen.getByText("子时间线")).toBeInTheDocument();
    expect(screen.getAllByText(/Word2Vec/).length).toBeGreaterThan(0);
  });

  it("redirects to 404 for unknown family", () => {
    renderFamily("invalid-family");
    expect(screen.getByText("404 page")).toBeInTheDocument();
  });
});
