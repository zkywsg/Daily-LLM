// GPT-3 数学:Kaplan scaling law + 模型族系数据 + ICL 示例。

/**
 * Kaplan et al. 2020 power-law(简化版):
 *   loss(N) ≈ (N_c / N)^α_N
 * 这里 α_N≈0.076, N_c 选个能让 GPT-1/2/3 落在曲线上的常数。
 * 不追求复现原论文精度,只让 viewer 摸到"参数越大 loss 越低"的趋势。
 */
const N_C = 8.8e13;
const ALPHA_N = 0.076;
export function scalingLoss(N: number): number {
  return Math.pow(N_C / Math.max(1, N), ALPHA_N);
}

export interface ModelPoint {
  name: string;
  /** 参数数量 */
  params: number;
  /** 训练 token 量(参考量级) */
  tokens?: number;
  /** 大致发布年份 */
  year: number;
  /** 颜色 hex */
  color: string;
}

export const MODEL_LINEUP: ModelPoint[] = [
  { name: "GPT-1", params: 117e6, tokens: 4e9, year: 2018, color: "#9ca3af" },
  { name: "GPT-2", params: 1.5e9, tokens: 40e9, year: 2019, color: "#a78bfa" },
  { name: "GPT-3", params: 175e9, tokens: 300e9, year: 2020, color: "#ec4899" },
  { name: "PaLM", params: 540e9, tokens: 780e9, year: 2022, color: "#f59e0b" },
  { name: "GPT-4 (估)", params: 1.8e12, tokens: 13e12, year: 2023, color: "#10b981" },
];

export function fmtParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return `${n}`;
}

// In-Context Learning 示例:同一任务,三种 shot 数
export interface ICLExample {
  task: string;
  description: string;
  zeroShot: string;
  oneShot: string;
  fewShot: string;
  /** 演示用 — 模型在不同 shot 下"假装"的准确率(数量级取自 GPT-3 论文 Fig 1.2) */
  accuracy: { zero: number; one: number; few: number };
}

export const ICL_EXAMPLES: ICLExample[] = [
  {
    task: "翻译",
    description: "英文 → 法文",
    zeroShot: `Translate English to French:\nsea otter =>`,
    oneShot: `Translate English to French:\nsea otter => loutre de mer\npeppermint =>`,
    fewShot: `Translate English to French:\nsea otter => loutre de mer\ncheese => fromage\npeppermint =>`,
    accuracy: { zero: 0.21, one: 0.32, few: 0.42 },
  },
  {
    task: "三位数加法",
    description: "纯文本算术",
    zeroShot: `Q: What is 247 + 358?\nA:`,
    oneShot: `Q: What is 12 + 7?\nA: 19\nQ: What is 247 + 358?\nA:`,
    fewShot: `Q: What is 12 + 7?\nA: 19\nQ: What is 53 + 81?\nA: 134\nQ: What is 247 + 358?\nA:`,
    accuracy: { zero: 0.08, one: 0.41, few: 0.78 },
  },
  {
    task: "情感分类",
    description: "评论 → positive/negative",
    zeroShot: `Review: "The movie was a complete waste of time."\nSentiment:`,
    oneShot: `Review: "Loved every minute."\nSentiment: positive\nReview: "The movie was a complete waste of time."\nSentiment:`,
    fewShot: `Review: "Loved every minute."\nSentiment: positive\nReview: "Just okay, nothing special."\nSentiment: neutral\nReview: "The movie was a complete waste of time."\nSentiment:`,
    accuracy: { zero: 0.62, one: 0.78, few: 0.89 },
  },
];

// Emergent ability:某些任务在参数量过临界后突然跳到可用,小模型怎么训都做不到
export interface EmergentTask {
  name: string;
  /** 不同参数量下的准确率,key 是参数量(log10 N) */
  curve: Array<{ logN: number; acc: number }>;
}

// 数据点取自 Wei et al. 2022 "Emergent Abilities" Fig 2 — 数量级
export const EMERGENT_TASKS: EmergentTask[] = [
  {
    name: "三位数加法",
    curve: [
      { logN: 8, acc: 0.0 },
      { logN: 9, acc: 0.0 },
      { logN: 10, acc: 0.02 },
      { logN: 11, acc: 0.41 },
      { logN: 12, acc: 0.78 },
    ],
  },
  {
    name: "MMLU 多任务理解",
    curve: [
      { logN: 8, acc: 0.25 },
      { logN: 9, acc: 0.27 },
      { logN: 10, acc: 0.3 },
      { logN: 11, acc: 0.42 },
      { logN: 12, acc: 0.66 },
    ],
  },
  {
    name: "波斯语 QA",
    curve: [
      { logN: 8, acc: 0.1 },
      { logN: 9, acc: 0.11 },
      { logN: 10, acc: 0.12 },
      { logN: 11, acc: 0.18 },
      { logN: 12, acc: 0.52 },
    ],
  },
];

// Sparse attention patterns
export type AttentionPattern = "dense" | "strided" | "fixed";

/** 给定 pattern 和 n,返回 mask:matrix[i][j] = 1 表示 i 可 attend 到 j */
export function buildAttentionMask(
  pattern: AttentionPattern,
  n: number,
  stride = 4,
): number[][] {
  const mat = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (j > i) continue; // 都是 causal,先排除上三角
      if (pattern === "dense") {
        mat[i][j] = 1;
      } else if (pattern === "strided") {
        // 每隔 stride 看一个
        if (j === i || (i - j) % stride === 0 || (i - j) < stride) mat[i][j] = 1;
      } else if (pattern === "fixed") {
        // 看局部 stride 窗 + 固定的 summary 位置
        const localWindow = stride;
        const summaryEvery = stride;
        if (i - j < localWindow) mat[i][j] = 1;
        if (j % summaryEvery === summaryEvery - 1) mat[i][j] = 1;
      }
    }
  }
  return mat;
}

/** 数 mask 矩阵里非零项 → 估计有效计算量 */
export function maskComplexity(mask: number[][]): number {
  let count = 0;
  for (const row of mask) for (const v of row) if (v) count++;
  return count;
}
