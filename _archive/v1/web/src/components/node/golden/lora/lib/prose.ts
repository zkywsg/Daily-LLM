export const LORA_SOURCE_PATH = "11-peft-lora/03-lora.md";

export interface ProseSections {
  previousWork: string;
  intuition: string;
  mechanism1: string;
  mechanism2: string;
  mechanism3: string;
  synergy: string;
  keyCode: string;
  performance: string;
  aftermath: string;
}

// LoRA 的特殊处:机制一/二/三 是 H2(不像 ResNet/Transformer/DDPM 那样塞在 H2 核心思想下)。
// 所以直接按 H2 路由,直觉是核心思想下的 H3。
const H2_KEYS: Array<{ test: RegExp; key: keyof ProseSections | "_coreInsight" }> = [
  { test: /^前作进展/, key: "previousWork" },
  { test: /^核心思想/, key: "_coreInsight" },
  { test: /^机制一/, key: "mechanism1" },
  { test: /^机制二/, key: "mechanism2" },
  { test: /^机制三/, key: "mechanism3" },
  { test: /^三件套协同/, key: "synergy" },
  { test: /^关键代码/, key: "keyCode" },
  { test: /^性能数据/, key: "performance" },
  { test: /^影响/, key: "aftermath" },
];

export function extractProse(markdown: string): ProseSections {
  const body = markdown
    .replace(/^---[\s\S]*?---\n?/, "")
    .replace(/```mermaid\n[\s\S]*?```\n(\*图 ?\d[^\n]*\*\n?)?/g, "");

  const sections: ProseSections = {
    previousWork: "",
    intuition: "",
    mechanism1: "",
    mechanism2: "",
    mechanism3: "",
    synergy: "",
    keyCode: "",
    performance: "",
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

    // 核心思想下的 ### 直觉 → intuition
    if (h3 && inCoreInsight && /^直觉/.test(h3[1].trim())) {
      flush();
      currentKey = "intuition";
      continue;
    }

    // 其他 H3(机制一下的"假设"、"LoRA 层结构"、机制二下的"参数账"等)
    // 留在当前 currentKey 的 buffer 里,作为该 H2 的子内容
    if (currentKey) buffer.push(line);
  }
  flush();

  return sections;
}
