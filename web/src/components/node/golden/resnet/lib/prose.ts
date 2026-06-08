export interface ProseSections {
  /** ## 之前卡在哪 章节正文 */
  beforeStuckOn: string;
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
  const body = markdown.replace(/^---[\s\S]*?---\n?/, "");

  const sections: ProseSections = {
    beforeStuckOn: "",
    coreInsight: "",
    intuition: "",
    mechanism: "",
    trainingDetails: "",
    keyCode: "",
    aftermath: "",
  };

  const h2Map: Record<string, keyof ProseSections> = {
    "之前卡在哪": "beforeStuckOn",
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
