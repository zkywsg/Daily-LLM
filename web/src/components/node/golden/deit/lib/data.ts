// DeiT 数据:训练 recipe 对比、benchmark 表、token 序列结构。

/** 机制一:现代训练 recipe vs 原版 ViT 训练设置 */
export interface RecipeRow {
  setting: string;
  vit: string;
  deit: string;
  hasDeit: boolean;
}

export const RECIPE_ROWS: RecipeRow[] = [
  { setting: "优化器", vit: "Adam", deit: "AdamW", hasDeit: true },
  { setting: "学习率 schedule", vit: "linear warmup + linear decay", deit: "cosine + 长 warmup", hasDeit: true },
  { setting: "Weight decay", vit: "0.1", deit: "0.05", hasDeit: true },
  { setting: "Stochastic depth", vit: "无", deit: "0.1", hasDeit: true },
  { setting: "Mixup", vit: "无", deit: "0.8", hasDeit: true },
  { setting: "CutMix", vit: "无", deit: "1.0", hasDeit: true },
  { setting: "RandAugment", vit: "无", deit: "9 / 0.5", hasDeit: true },
  { setting: "Random erasing", vit: "无", deit: "0.25", hasDeit: true },
  { setting: "Label smoothing", vit: "无", deit: "0.1", hasDeit: true },
  { setting: "Repeated augmentation", vit: "无", deit: "3×", hasDeit: true },
  { setting: "EMA", vit: "无", deit: "有", hasDeit: true },
];

/** 机制二:Hard vs Soft distillation 目标对比(10 类玩具分布) */
export const TEACHER_LOGITS = [0.02, 0.03, 0.78, 0.04, 0.02, 0.03, 0.02, 0.02, 0.02, 0.02];

export function hardTarget(logits: number[]): number[] {
  const maxIdx = logits.indexOf(Math.max(...logits));
  return logits.map((_, i) => (i === maxIdx ? 1 : 0));
}

/** 机制三 + 性能:核心 benchmark 表(论文 Table 1) */
export interface BenchRow {
  model: string;
  params: string;
  top1: number;
  hardware: string;
  trainTime: string;
  highlight: boolean;
}

export const BENCH_ROWS: BenchRow[] = [
  { model: "ResNet-50", params: "25M", top1: 76.1, hardware: "8 V100", trainTime: "~30 h", highlight: false },
  { model: "EfficientNet-B0", params: "5.3M", top1: 77.1, hardware: "32 TPU", trainTime: "~3 天", highlight: false },
  { model: "ViT-B/16 (JFT 预训练)", params: "86M", top1: 77.9, hardware: "TPUv3-2500", trainTime: "~30 天", highlight: false },
  { model: "DeiT-S", params: "22M", top1: 79.8, hardware: "1 V100", trainTime: "3 天", highlight: true },
  { model: "DeiT-B", params: "86M", top1: 81.8, hardware: "1 V100", trainTime: "4 天", highlight: true },
  { model: "EfficientNet-B7", params: "66M", top1: 82.9, hardware: "32 TPU", trainTime: "~3 周", highlight: false },
  { model: "DeiT-B with distillation", params: "86M", top1: 83.4, hardware: "1 V100", trainTime: "4 天", highlight: true },
];

/** 数据效率对比:训练所需数据规模(log 刻度用) */
export interface DataEfficiencyPoint {
  label: string;
  numImages: number; // 训练图像数
  top1: number;
  isDeit: boolean;
}

export const DATA_EFFICIENCY: DataEfficiencyPoint[] = [
  { label: "ViT-B/16\n(JFT-300M 预训练)", numImages: 300e6, top1: 77.9, isDeit: false },
  { label: "ViT-B/16\n(仅 ImageNet-1K)", numImages: 1.3e6, top1: 76.5, isDeit: false },
  { label: "DeiT-B\n(仅 ImageNet-1K)", numImages: 1.3e6, top1: 81.8, isDeit: true },
  { label: "DeiT-B distill\n(仅 ImageNet-1K)", numImages: 1.3e6, top1: 83.4, isDeit: true },
];

/** token 序列结构:ViT vs DeiT */
export const VIT_TOKENS = ["[CLS]", ...Array.from({ length: 6 }, (_, i) => `patch${i + 1}`)];
export const DEIT_TOKENS = ["[CLS]", ...Array.from({ length: 6 }, (_, i) => `patch${i + 1}`), "[DIST]"];
