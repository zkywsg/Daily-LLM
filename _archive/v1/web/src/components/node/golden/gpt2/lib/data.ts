// GPT 系列规模演化
export interface ModelSpec {
  name: string;
  year: number;
  params: number;        // 单位:M
  layers: number;
  dModel: number;
  data: number;          // 单位:B tokens
  isGpt2: boolean;
}

export const GPT_SCALING: ModelSpec[] = [
  { name: "GPT-1",         year: 2018, params: 117,    layers: 12, dModel: 768,  data: 0.8, isGpt2: false },
  { name: "GPT-2 Small",   year: 2019, params: 117,    layers: 12, dModel: 768,  data: 40,  isGpt2: true },
  { name: "GPT-2 Medium",  year: 2019, params: 345,    layers: 24, dModel: 1024, data: 40,  isGpt2: true },
  { name: "GPT-2 Large",   year: 2019, params: 762,    layers: 36, dModel: 1280, data: 40,  isGpt2: true },
  { name: "GPT-2 XL",      year: 2019, params: 1500,   layers: 48, dModel: 1600, data: 40,  isGpt2: true },
  { name: "GPT-3",         year: 2020, params: 175000, layers: 96, dModel: 12288, data: 300, isGpt2: false },
];

// Zero-shot 任务表 (论文 Table 1 简化)
export interface PromptTemplate {
  task: string;
  template: string;
  example_in: string;
  example_out: string;
}

export const PROMPT_TEMPLATES: PromptTemplate[] = [
  {
    task: "翻译 (EN→FR)",
    template: "Translate English to French:\n<英文>\n→",
    example_in: "Translate English to French:\nhello world\n→",
    example_out: " bonjour le monde",
  },
  {
    task: "摘要 (TL;DR)",
    template: "<文章>\n\nTL;DR:",
    example_in: "The quick brown fox jumps over the lazy dog. The dog wakes up surprised.\n\nTL;DR:",
    example_out: " A fox jumps over a sleeping dog and wakes it up.",
  },
  {
    task: "问答 QA",
    template: "<文档>\nQ: <问题>\nA:",
    example_in: "The Eiffel Tower is in Paris.\nQ: Where is the Eiffel Tower?\nA:",
    example_out: " In Paris.",
  },
  {
    task: "常识续写 LAMBADA",
    template: "<上下文最后一句缺词>",
    example_in: "She opened the door and saw a black ___",
    example_out: " cat",
  },
];

// 论文 Table 3 zero-shot 性能 (简化)
export interface ZeroShotPerf {
  task: string;
  prev_zs: number;     // 前作 zero-shot
  gpt2_xl: number;     // GPT-2 XL zero-shot
  sft_sota: number | null;  // 监督学习 SOTA
  higher_is_better: boolean;
}

export const ZERO_SHOT_RESULTS: ZeroShotPerf[] = [
  { task: "LAMBADA",    prev_zs: 59.2, gpt2_xl: 63.2, sft_sota: 76.4, higher_is_better: true  },
  { task: "CBT-NE",     prev_zs: 82.3, gpt2_xl: 89.1, sft_sota: 87.7, higher_is_better: true  },
  { task: "CBT-CN",     prev_zs: 85.7, gpt2_xl: 93.3, sft_sota: 96.0, higher_is_better: true  },
  { task: "Winograd",   prev_zs: 63.7, gpt2_xl: 70.7, sft_sota: null, higher_is_better: true  },
  { task: "Story Cloze", prev_zs: 77.6, gpt2_xl: 77.4, sft_sota: 86.5, higher_is_better: true },
];

// 模拟 zero-shot 准确率 vs 模型规模(4 个任务)— 用平滑函数体现涌现
const SCALE_BREAKPOINTS = [117, 345, 762, 1500]; // M params
export interface TaskScaling {
  task: string;
  color: string;
  vals: number[];   // 4 个 size 上的 acc / score
  isEmergent: boolean;
}
export const ZS_SCALING: TaskScaling[] = [
  { task: "LAMBADA acc", color: "#ec4899", vals: [45.99, 55.48, 60.12, 63.24], isEmergent: false },
  { task: "WikiText perpl. ↓", color: "#3b82f6", vals: [37.50, 26.37, 22.05, 17.48], isEmergent: false },
  { task: "CBT-NE acc",  color: "#10b981", vals: [83.4, 87.1, 88.0, 89.1], isEmergent: false },
  { task: "WMT En→Fr BLEU", color: "#f59e0b", vals: [1.5, 4.1, 8.2, 11.5], isEmergent: true },
];
export { SCALE_BREAKPOINTS };

// 采样策略
export function applyTemperature(logits: number[], temp: number): number[] {
  const scaled = logits.map((x) => x / Math.max(temp, 0.01));
  const maxL = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - maxL));
  const Z = exps.reduce((s, x) => s + x, 0);
  return exps.map((x) => x / Z);
}

export function applyTopK(probs: number[], k: number): number[] {
  const sorted = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const keep = new Set(sorted.slice(0, k).map((x) => x.i));
  const masked = probs.map((p, i) => keep.has(i) ? p : 0);
  const Z = masked.reduce((s, x) => s + x, 0);
  return Z > 0 ? masked.map((p) => p / Z) : probs;
}

export function applyTopP(probs: number[], p: number): number[] {
  const sorted = probs.map((q, i) => ({ q, i })).sort((a, b) => b.q - a.q);
  let cum = 0;
  const keep = new Set<number>();
  for (const { q, i } of sorted) {
    keep.add(i);
    cum += q;
    if (cum >= p) break;
  }
  const masked = probs.map((q, i) => keep.has(i) ? q : 0);
  const Z = masked.reduce((s, x) => s + x, 0);
  return Z > 0 ? masked.map((q) => q / Z) : probs;
}
