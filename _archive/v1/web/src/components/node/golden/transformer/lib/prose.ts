export const TRANSFORMER_SOURCE_PATH = "05-transformer/01-transformer.md";

export interface ProseSections {
  /** ## 前作进展 */
  previousWork: string;
  /** ### 直觉:相似度查询替代循环展开 */
  intuition: string;
  /** ### 机制一:Scaled Dot-Product Attention */
  mechanism1: string;
  /** ### 机制二:Multi-Head 多角度切片 */
  mechanism2: string;
  /** ### 机制三:Position Encoding 补回顺序 */
  mechanism3: string;
  /** ### 三件套协同 */
  synergy: string;
  /** ### 完整 encoder / decoder */
  encoderDecoder: string;
  /** ### Post-LN 是原版细节(后被 Pre-LN 取代) */
  postLN: string;
  /** ## 训练细节 */
  trainingDetails: string;
  /** ## 关键代码 */
  keyCode: string;
  /** ## 影响 / 后续 */
  aftermath: string;
}

const H3_KEYS: Array<{ test: RegExp; key: keyof ProseSections }> = [
  { test: /^直觉/, key: "intuition" },
  { test: /^机制一/, key: "mechanism1" },
  { test: /^机制二/, key: "mechanism2" },
  { test: /^机制三/, key: "mechanism3" },
  { test: /^三件套协同/, key: "synergy" },
  { test: /^完整\s*encoder/i, key: "encoderDecoder" },
  { test: /^Post-?LN/i, key: "postLN" },
];

const H2_KEYS: Array<{ test: RegExp; key: keyof ProseSections | "_coreInsight" }> = [
  { test: /^前作进展/, key: "previousWork" },
  { test: /^核心思想/, key: "_coreInsight" },
  { test: /^训练细节/, key: "trainingDetails" },
  { test: /^关键代码/, key: "keyCode" },
  { test: /^影响/, key: "aftermath" },
];

export function extractProse(markdown: string): ProseSections {
  const body = markdown
    .replace(/^---[\s\S]*?---\n?/, "")
    // 剔除 mermaid 图块 + 紧随的图注 —— 交互页有自己的 SVG
    .replace(/```mermaid\n[\s\S]*?```\n(\*图 ?\d[^\n]*\*\n?)?/g, "");

  const sections: ProseSections = {
    previousWork: "",
    intuition: "",
    mechanism1: "",
    mechanism2: "",
    mechanism3: "",
    synergy: "",
    encoderDecoder: "",
    postLN: "",
    trainingDetails: "",
    keyCode: "",
    aftermath: "",
  };

  let currentKey: keyof ProseSections | null = null;
  let inCoreInsight = false;
  let buffer: string[] = [];

  const flush = () => {
    if (currentKey) sections[currentKey] = buffer.join("\n").trim();
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
