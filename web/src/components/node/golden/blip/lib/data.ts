// BLIP 三任务
export interface TaskInfo {
  name: string;
  full: string;
  desc: string;
}
export const THREE_TASKS: TaskInfo[] = [
  { name: "ITC", full: "Image-Text Contrastive", desc: "同 CLIP 的对称 InfoNCE,batch 内对比学习,让正样本图文相似度高、负样本低" },
  { name: "ITM", full: "Image-Text Matching", desc: "二分类判断图文是否匹配,用 hard negative mining 从 ITC 相似度矩阵里选最像但非正样本的负例" },
  { name: "LM", full: "Language Modeling", desc: "以图像为 condition,causal decoder 自回归生成 caption:-log p(token_t | tokens_<t, image)" },
];

// CapFilt 数据质量提升前后对比
export interface CapFiltRow {
  metric: string;
  before: number;
  after: number;
}
export const CAPFILT_IMPROVEMENT: CapFiltRow[] = [
  { metric: "COCO CIDEr", before: 117.5, after: 133.3 },
  { metric: "VQA", before: 75.3, after: 78.3 },
];

export const DATA_SCALE = { beforeM: 14, afterM: 130 };

// BLIP-2 参数分布
export interface ParamRow {
  component: string;
  paramsM: number;
  trainable: boolean;
}
export const QFORMER_PARAMS: ParamRow[] = [
  { component: "冻结 ViT-G/14", paramsM: 1000, trainable: false },
  { component: "Q-Former(可训练)", paramsM: 188, trainable: true },
  { component: "冻结 Flan-T5 XXL", paramsM: 11000, trainable: false },
];

// 训练成本对比:Flamingo vs BLIP-2
export interface CostRow {
  model: string;
  gpuDays: number;
  costUSD: number;
}
export const TRAINING_COST_COMPARE: CostRow[] = [
  { model: "Flamingo-80B", gpuDays: 15000, costUSD: 1_000_000 },
  { model: "BLIP-2", gpuDays: 144, costUSD: 10_000 },
];

// benchmark 对比(论文 Table)
export interface BenchRow {
  task: string;
  flamingo: number;
  blip2: number;
  delta: number;
}
export const BENCHMARK_TABLE: BenchRow[] = [
  { task: "VQA v2(zero-shot)", flamingo: 56.3, blip2: 65.2, delta: 8.9 },
  { task: "NoCaps CIDEr", flamingo: 99.0, blip2: 121.0, delta: 22 },
  { task: "COCO CIDEr", flamingo: 65.3, blip2: 80.3, delta: 15 },
  { task: "OK-VQA", flamingo: 50.6, blip2: 45.9, delta: -4.7 },
];
