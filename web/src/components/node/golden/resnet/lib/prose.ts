/** 本页 markdown 正本的 repo-root-relative 路径,供相对链接/图片解析 */
export const RESNET_SOURCE_PATH = "01-cnn/05-resnet.md";

export interface ProseSections {
  /** ## 前作进展 章节正文 */
  previousWork: string;
  /** ## 核心思想 章节自己的正文（不含 ### 子段） */
  coreInsight: string;
  /** ### 直觉 子段正文 */
  intuition: string;
  /** ### 机制 子段正文 */
  mechanism: string;
  /** ## 训练细节 章节正文（含表格） */
  trainingDetails: string;
  /** ## 关键代码 章节正文（含 fenced code block） */
  keyCode: string;
  /** ## 影响 / 后续 章节正文 */
  aftermath: string;
}

/**
 * 从节点 markdown 全文按 H2/H3 章节切出 prose 段。
 */
export function extractProse(markdown: string): ProseSections {
  const body = markdown
    .replace(/^---[\s\S]*?---\n?/, "")
    // 剔除 mermaid 图及紧随的"*图 N：…*"图注——本页已有交互版 SVG 表达同一内容
    .replace(/```mermaid\n[\s\S]*?```\n(\*图 ?\d[^\n]*\*\n?)?/g, "");

  const sections: ProseSections = {
    previousWork: "",
    coreInsight: "",
    intuition: "",
    mechanism: "",
    trainingDetails: "",
    keyCode: "",
    aftermath: "",
  };

  const h2Map: Record<string, keyof ProseSections> = {
    "前作进展": "previousWork",
    "核心思想": "coreInsight",
    "训练细节": "trainingDetails",
    "关键代码": "keyCode",
    "影响 / 后续": "aftermath",
    "影响 / 后续 ": "aftermath",
    "影响/后续": "aftermath",
  };
  const h3Map: Record<string, keyof ProseSections> = {
    "直觉": "intuition",
    "机制": "mechanism",
  };

  const lines = body.split("\n");
  let currentKey: keyof ProseSections | null = null;
  let buffer: string[] = [];

  const flush = () => {
    if (currentKey) {
      sections[currentKey] = buffer.join("\n").trim();
    }
    buffer = [];
  };

  for (const line of lines) {
    const h2Match = /^## +(.+?)\s*$/.exec(line);
    const h3Match = /^### +(.+?)\s*$/.exec(line);

    if (h2Match) {
      flush();
      const name = h2Match[1].trim();
      currentKey = h2Map[name] ?? null;
      continue;
    }

    if (
      h3Match &&
      (currentKey === "coreInsight" ||
        currentKey === "intuition" ||
        currentKey === "mechanism")
    ) {
      flush();
      const name = h3Match[1].trim();
      currentKey = h3Map[name] ?? null;
      continue;
    }

    if (h3Match) {
      buffer.push(line);
      continue;
    }

    if (currentKey) {
      buffer.push(line);
    }
  }
  flush();

  return sections;
}
