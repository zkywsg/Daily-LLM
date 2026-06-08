import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import NodePageResNet from "./NodePageResNet";

function renderPage() {
  return render(
    <MemoryRouter>
      <NodePageResNet />
    </MemoryRouter>
  );
}

describe("NodePageResNet", () => {
  it("renders hero with title and key idea", () => {
    renderPage();
    expect(screen.getByText("ResNet (2015)")).toBeInTheDocument();
    expect(
      screen.getByText(/shortcut.*只学残差修正/)
    ).toBeInTheDocument();
  });

  it("renders all 3 stage h2 headings", () => {
    renderPage();
    expect(screen.getByText("之前卡在哪")).toBeInTheDocument();
    expect(screen.getByText("残差的直觉")).toBeInTheDocument();
    // "梯度高速公路" 出现多次（如 stage 标题 + 章节内文本），用 getAllByText
    expect(screen.getAllByText("梯度高速公路").length).toBeGreaterThan(0);
  });

  it("DepthSlider toggles depth and updates legend", () => {
    renderPage();
    // 默认 56 层
    expect(screen.getByText(/plain \(56 层\)/)).toBeInTheDocument();
    // 点击 152 层按钮
    fireEvent.click(screen.getByRole("button", { name: "152" }));
    expect(screen.getByText(/plain \(152 层\)/)).toBeInTheDocument();
  });

  it("BlockTypeToggle changes block params display", () => {
    renderPage();
    // 默认 bottleneck，应显示 69,632
    expect(screen.getByText("69,632")).toBeInTheDocument();
    // 切换到 BasicBlock
    fireEvent.click(screen.getByRole("button", { name: "BasicBlock" }));
    expect(screen.getByText("73,728")).toBeInTheDocument();
  });

  it("StackDepthSlider changes stack count display", () => {
    renderPage();
    // 默认 6 块
    expect(screen.getByText(/6 块串联/)).toBeInTheDocument();
    // DepthSlider 和 StackDepthSlider 都有 "20" 按钮，需通过父容器（含 "堆叠块数："）定位
    // 页面上可能有多个 StackDepthSlider 实例，取第一个
    const stackLabel = screen.getAllByText(/堆叠块数/)[0];
    const stackContainer = stackLabel.parentElement as HTMLElement;
    const stack20 = Array.from(
      stackContainer.querySelectorAll("button")
    ).find((b) => b.textContent === "20") as HTMLButtonElement;
    fireEvent.click(stack20);
    expect(screen.getByText(/20 块串联/)).toBeInTheDocument();
  });

  it("renders footer sections (training / code / aftermath)", () => {
    renderPage();
    expect(screen.getByText("训练细节")).toBeInTheDocument();
    expect(screen.getByText("关键代码")).toBeInTheDocument();
    expect(screen.getByText(/影响.*后续/)).toBeInTheDocument();
  });
});
