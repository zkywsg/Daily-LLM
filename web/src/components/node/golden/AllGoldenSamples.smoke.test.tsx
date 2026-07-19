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

  it.each(routes)("渲染 %s 不报错、有 h1 标题、无残留 markdown 语法", async (route) => {
    const { container } = render(
      <MemoryRouter initialEntries={[`/families/${route}`]}>
        <Routes>
          <Route path="/families/:familyId/:nodeSlug" element={<NodePage />} />
        </Routes>
      </MemoryRouter>
    );

    await waitFor(() => {
      expect(screen.getByRole("heading", { level: 1 })).toBeInTheDocument();
    });

    // 剥离 <pre>/<code>(代码块里的字面 ** 或 << 是正常内容,如
    // `d_k ** 0.5`、`r=8 << d`,不该被当成没渲染的 markdown 语法)
    const clone = container.cloneNode(true) as HTMLElement;
    clone.querySelectorAll("pre, code").forEach((el) => el.remove());
    const text = clone.textContent ?? "";

    const context = (pattern: RegExp) => {
      const m = pattern.exec(text);
      if (!m || m.index == null) return "";
      const idx = m.index;
      return JSON.stringify(text.slice(Math.max(0, idx - 40), idx + 40));
    };

    expect(
      text,
      `${route} 正文里有残留的 ** 未渲染成粗体,上下文: ${context(/\*\*[^*]/)}`
    ).not.toMatch(/\*\*[^*]/);
    expect(
      text,
      `${route} 正文里有残留的 ## 标题符号,上下文: ${context(/(^|\n)##\s/)}`
    ).not.toMatch(/(^|\n)##\s/);
    expect(
      text,
      `${route} 正文里有残留的 $$ LaTeX 定界符,上下文: ${context(/\$\$/)}`
    ).not.toMatch(/\$\$/);
  });
});
