// Segment-level recurrence:段序列 + 每段能看到的有效上下文
export interface SegmentStep {
  segment: number; // 第几段(从 1 开始)
  segLen: number; // 单段长度(token)
  effectiveContext: number; // 该段理论最大可及上下文 = segment * segLen(简化模型,O(N*L))
}
export function buildSegmentSteps(numSegments: number, segLen: number = 384): SegmentStep[] {
  return Array.from({ length: numSegments }, (_, i) => ({
    segment: i + 1,
    segLen,
    effectiveContext: (i + 1) * segLen,
  }));
}

// 有效上下文 vs 段数(层数固定为 18,近似 O(N x L) 增长,对照论文 512 -> 3800 的跃迁)
export interface ContextGrowthPoint {
  segments: number;
  fixedWindow: number; // 原版 Transformer:上下文恒为窗口大小,不随段数增长
  transformerXL: number; // Transformer-XL:随段数线性累积(受层数上限截断)
}
const SEG_LEN = 384;
const NUM_LAYERS = 18;
export const CONTEXT_GROWTH: ContextGrowthPoint[] = Array.from({ length: 11 }, (_, i) => {
  const segments = i; // 0..10
  const uncapped = segments * SEG_LEN;
  const cap = NUM_LAYERS * SEG_LEN * 1.5; // 论文报告的实际上限附近做视觉截断(约 3800 附近打平)
  return {
    segments,
    fixedWindow: SEG_LEN,
    transformerXL: Math.min(uncapped, cap),
  };
});

// perplexity / benchmark 表(来自论文 Table 数据,見 05-transformer/02-transformer-xl.md 性能数据一节)
export interface BenchmarkRow {
  benchmark: string;
  model: string;
  metric: number;
  unit: string;
  highlight?: boolean;
}
export const BENCHMARKS: BenchmarkRow[] = [
  { benchmark: "WikiText-103", model: "LSTM-based SOTA (2018)", metric: 40.8, unit: "PPL" },
  { benchmark: "WikiText-103", model: "Transformer 64 层 (Al-Rfou)", metric: 30.0, unit: "PPL" },
  { benchmark: "WikiText-103", model: "Transformer-XL Large", metric: 18.3, unit: "PPL", highlight: true },
  { benchmark: "enwik8", model: "Transformer 64 层", metric: 1.06, unit: "BPC" },
  { benchmark: "enwik8", model: "Transformer-XL Large (18层)", metric: 0.99, unit: "BPC", highlight: true },
  { benchmark: "One Billion Word", model: "LSTM", metric: 23.7, unit: "PPL" },
  { benchmark: "One Billion Word", model: "Transformer-XL", metric: 21.8, unit: "PPL", highlight: true },
];

// 训练细节表
export interface TrainConfigRow {
  dim: string;
  value: string;
}
export const TRAIN_CONFIG: TrainConfigRow[] = [
  { dim: "模型", value: "18 层, d_model=1024, h=16, d_ff=4096, ~257M 参数" },
  { dim: "Segment length", value: "训练 384 token,推理可设到 1600 token" },
  { dim: "Cache length", value: "训练 384(=segment length),推理 1600+" },
  { dim: "Vocabulary", value: "267K word-level(adaptive softmax 分桶)" },
  { dim: "优化器", value: "Adam,lr=2.5e-4,cosine schedule,warmup 0 步" },
  { dim: "Dropout", value: "0.2(attention / FFN / embedding)" },
  { dim: "训练时间", value: "4 × V100 GPU × 8 天" },
];

// 绝对 PE vs 相对 PE:跨段位置歧义示意用的段内位置序列
export interface PositionRow {
  segment: number;
  positions: number[]; // 段内的绝对位置索引(每段都从 0 开始 —— 这正是问题所在)
}
export const ABS_POSITION_SEGMENTS: PositionRow[] = [
  { segment: 1, positions: [0, 1, 2, 3, 4, 5] },
  { segment: 2, positions: [0, 1, 2, 3, 4, 5] },
];

// 推理速度对比(sliding window vs segment cache)
export interface SpeedRow {
  method: string;
  relativeSpeed: number; // 相对速度(sliding window = 1x)
}
export const INFERENCE_SPEEDUP: SpeedRow[] = [
  { method: "Stride-1 Sliding Window (Al-Rfou)", relativeSpeed: 1 },
  { method: "Transformer-XL (Segment Cache)", relativeSpeed: 1874 },
];
