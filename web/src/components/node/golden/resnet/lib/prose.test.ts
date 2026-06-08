import { describe, it, expect } from "vitest";
import { extractProse } from "./prose";

const sampleMarkdown = `---
name: "ResNet"
year: 2015
---

# ResNet (2015)

## 之前卡在哪

第一段卡点。
第二段卡点。

## 核心思想

总览段落。

### 直觉

直觉段落 1。
直觉段落 2。

### 机制

机制公式 $y = F(x) + x$。

## 训练细节

| 维度 | 值 |
|------|---|
| lr | 0.1 |

## 关键代码

\`\`\`python
import torch
\`\`\`

## 影响 / 后续

→ 链接 1
→ 链接 2
`;

describe("extractProse", () => {
  it("removes frontmatter", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.beforeStuckOn).not.toContain("---");
    expect(result.beforeStuckOn).not.toContain("name:");
  });

  it("extracts 之前卡在哪 section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.beforeStuckOn).toContain("第一段卡点");
    expect(result.beforeStuckOn).toContain("第二段卡点");
    expect(result.beforeStuckOn).not.toContain("总览段落");
  });

  it("extracts 核心思想 section (only its own prose, before any ###)", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.coreInsight).toContain("总览段落");
    expect(result.coreInsight).not.toContain("直觉段落");
  });

  it("extracts 直觉 sub-section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.intuition).toContain("直觉段落 1");
    expect(result.intuition).toContain("直觉段落 2");
    expect(result.intuition).not.toContain("机制公式");
  });

  it("extracts 机制 sub-section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.mechanism).toContain("机制公式");
    expect(result.mechanism).not.toContain("训练细节");
  });

  it("extracts 训练细节 section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.trainingDetails).toContain("lr");
    expect(result.trainingDetails).toContain("0.1");
  });

  it("extracts 关键代码 section including code block", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.keyCode).toContain("import torch");
  });

  it("extracts 影响 / 后续 section", () => {
    const result = extractProse(sampleMarkdown);
    expect(result.aftermath).toContain("链接 1");
    expect(result.aftermath).toContain("链接 2");
  });

  it("returns empty string for missing section gracefully", () => {
    const minimal = "# Title\n\n## 之前卡在哪\n\nonly this section.\n";
    const result = extractProse(minimal);
    expect(result.beforeStuckOn).toContain("only this section");
    expect(result.coreInsight).toBe("");
    expect(result.trainingDetails).toBe("");
  });
});
