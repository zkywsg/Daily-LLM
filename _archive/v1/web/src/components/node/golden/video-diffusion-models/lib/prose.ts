export const VDM_SOURCE_PATH = "16-world-models/02-video-diffusion-models.md";

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

const H3_KEYS: Array<{ test: RegExp; key: keyof ProseSections }> = [
  { test: /^直觉/, key: "intuition" },
];

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
    previousWork: "", intuition: "", mechanism1: "", mechanism2: "",
    mechanism3: "", synergy: "", keyCode: "", performance: "", aftermath: "",
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
