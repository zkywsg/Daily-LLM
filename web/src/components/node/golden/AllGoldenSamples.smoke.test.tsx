import { describe, it, expect } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router";
import { NodePage } from "../NodePage";
import { goldenSamples } from "./index";

// 全量冒烟测试:遍历所有金标本路由,只断言"能无报错渲染出标题"——
// 不校验具体文案/数据/交互是否正确。用来在改动 NodePage.tsx、
// MarkdownRenderer、Stage.module.css 等横切文件时,快速发现某个
// 冷门节点被搞挂了(例如 lib/prose.ts 抛异常、widgets 数组越界、
// import 路径写错导致模块加载失败)。
describe("所有金标本节点冒烟测试", () => {
  const routes = Object.keys(goldenSamples);

  it("金标本注册表不为空", () => {
    expect(routes.length).toBeGreaterThan(0);
  });

  it.each(routes)("渲染 %s 不报错、有 h1 标题", async (route) => {
    render(
      <MemoryRouter initialEntries={[`/families/${route}`]}>
        <Routes>
          <Route path="/families/:familyId/:nodeSlug" element={<NodePage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    });
  });
});
