// 论文 Table 4:五项改动的累积消融(SQuAD F1 / MNLI)
export interface AblationStep {
  label: string;
  squad: number;
  mnli: number;
}
export const ABLATION_STEPS: AblationStep[] = [
  { label: "BERT-large(原版 baseline)", squad: 90.9, mnli: 86.6 },
  { label: "+ 动态 masking", squad: 91.2, mnli: 86.7 },
  { label: "+ 去 NSP + 单序列输入", squad: 91.4, mnli: 87.0 },
  { label: "+ 大 batch(8K)", squad: 91.5, mnli: 87.1 },
  { label: "+ 更多数据(160GB)+ 更长训练", squad: 94.6, mnli: 89.4 },
];

// 论文 Table 2:输入格式对比(NSP 辩论)
export interface NspFormatRow {
  format: string;
  mnli: number;
  squad: number;
}
export const NSP_FORMAT_COMPARE: NspFormatRow[] = [
  { format: "句对 + NSP(BERT 原版)", mnli: 87.3, squad: 91.1 },
  { format: "句对,去 NSP loss", mnli: 87.4, squad: 91.4 },
  { format: "单序列,无 NSP", mnli: 87.9, squad: 92.0 },
];

// 数据 / 算力规模对比
export interface ScaleRow {
  model: string;
  dataGB: number;
  tokensB: number; // 十亿 token
  batch: number;
  lr: string;
}
export const SCALE_COMPARE: ScaleRow[] = [
  { model: "BERT", dataGB: 16, tokensB: 130, batch: 256, lr: "1e-4" },
  { model: "RoBERTa", dataGB: 160, tokensB: 2000, batch: 8000, lr: "4e-4" },
];

// 简单确定性伪随机(mulberry32),用于生成静态/动态 mask 位置
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const DEMO_TOKENS = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "and", "cat", "ran"];

// 静态 masking:同一 seed 用于所有 epoch(位置固定)
export function staticMaskPositions(epoch: number, ratio = 0.25): boolean[] {
  const rand = mulberry32(42); // 固定 seed,与 epoch 无关
  void epoch;
  return DEMO_TOKENS.map(() => rand() < ratio);
}

// 动态 masking:每个 epoch 用不同 seed(位置每次重新选)
export function dynamicMaskPositions(epoch: number, ratio = 0.25): boolean[] {
  const rand = mulberry32(42 + epoch * 1000);
  return DEMO_TOKENS.map(() => rand() < ratio);
}
