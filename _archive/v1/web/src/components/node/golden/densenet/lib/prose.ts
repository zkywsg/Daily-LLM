/** 本页 markdown 正本的 repo-root-relative 路径,供相对链接/图片解析 */
export const DENSENET_SOURCE_PATH = "01-cnn/06-densenet.md";

export interface ProseSections {
  /** ## 前作进展 章节正文 */
  previousWork: string;
  /** ### 直觉 子段正文 */
  intuition: string;
  /** ### 机制一:Concat 替代 Add 子段正文 */
  mechanism1: string;
  /** ### 机制二:Growth Rate k 子段正文 */
  mechanism2: string;
  /** ### 机制三:Bottleneck + Compression 子段正文 */
  mechanism3: string;
  /** ### 三件套协同 子段正文 */
  synergy: string;
  /** ## 训练细节 章节正文(含表格) */
  trainingDetails: string;
  /** ## 关键代码 章节正文(含 fenced code block) */
  keyCode: string;
  /** ## 影响 / 后续 章节正文 */
  aftermath: string;
}

const H3_KEYS: Array<{ test: RegExp; key: keyof ProseSections }> = [
  { test: /^直觉/, key: "intuition" },
  { test: /^机制一/, key: "mechanism1" },
  { test: /^机制二/, key: "mechanism2" },
  { test: /^机制三/, key: "mechanism3" },
  { test: /^三件套协同/, key: "synergy" },
];

const H2_KEYS: Array<{ test: RegExp; key: keyof ProseSections | "_coreInsight" }> = [
  { test: /^前作进展/, key: "previousWork" },
  { test: /^核心思想/, key: "_coreInsight" },
  { test: /^训练细节/, key: "trainingDetails" },
  { test: /^关键代码/, key: "keyCode" },
  { test: /^影响/, key: "aftermath" },
];

/**
 * 从节点 markdown 全文按 H2/H3 章节切出 prose 段。
 * 使用 append-safe flush():同一 key 若被多个片段命中(理论上不应发生,但防御性地保留),
 * 用 "\n\n" 拼接而非覆盖。
 */
export function extractProse(markdown: string): ProseSections {
  const body = markdown
    .replace(/^---[\s\S]*?---\n?/, "")
    // 剔除 mermaid 图及紧随的"*图 N:…*"图注——本页已有交互版 SVG 表达同一内容
    .replace(/```mermaid\n[\s\S]*?```\n(\*图 ?\d[^\n]*\*\n?)?/g, "")
    // 剔除内联 markdown 图片及紧随的图注——本页用交互版 SVG widget 替代
    .replace(/!\[[^\]]*\]\([^)]*\)\n(\*图 ?\d[^\n]*\*\n?)?/g, "");

  const sections: ProseSections = {
    previousWork: "",
    intuition: "",
    mechanism1: "",
    mechanism2: "",
    mechanism3: "",
    synergy: "",
    trainingDetails: "",
    keyCode: "",
    aftermath: "",
  };

  let currentKey: keyof ProseSections | null = null;
  let inCoreInsight = false;
  let buffer: string[] = [];

  const flush = () => {
    if (currentKey) {
      const text = buffer.join("\n").trim();
      if (text) {
        sections[currentKey] = sections[currentKey]
          ? `${sections[currentKey]}\n\n${text}`
          : text;
      }
    }
    buffer = [];
  };

  for (const line of body.split("\n")) {
    const h2 = /^## +(.+?)\s*$/.exec(line);
    const h3 = /^### +(.+?)\s*$/.exec(line);

    if (h2) {
      flush();
      const m = H2_KEYS.find((x) => x.test.test(h2[1].trim()));
      if (m && m.key === "_coreInsight") {
        currentKey = null;
        inCoreInsight = true;
      } else if (m) {
        currentKey = m.key as keyof ProseSections;
        inCoreInsight = false;
      } else {
        currentKey = null;
        inCoreInsight = false;
      }
      continue;
    }

    if (h3 && inCoreInsight) {
      flush();
      const m = H3_KEYS.find((x) => x.test.test(h3[1].trim()));
      currentKey = m ? m.key : null;
      continue;
    }

    if (currentKey) buffer.push(line);
  }
  flush();

  return sections;
}
