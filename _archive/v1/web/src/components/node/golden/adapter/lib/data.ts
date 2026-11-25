// 存储对比:9 个 GLUE 任务,Full Fine-Tuning vs Adapter Tuning
export interface StorageRow {
  method: string;
  totalGB: number;
  detail: string;
  color: string;
  bg: string;
}
export const STORAGE_COMPARE: StorageRow[] = [
  { method: "Full Fine-Tuning", totalGB: 11.7, detail: "9 × 1.3GB(每任务独立存一份 BERT-large 340M)", color: "#9ca3af", bg: "#f3f4f6" },
  { method: "Adapter Tuning", totalGB: 1.4, detail: "共享 base 1.3GB + 9 × 8MB adapter 权重", color: "#10b981", bg: "#ecfdf5" },
];

// 参数效率:训练参数占比 vs 保留的全参微调性能百分比
export interface ParamEfficiencyRow {
  label: string;
  trainedParamsPct: number;
  performancePct: number;
}
export const PARAM_EFFICIENCY: ParamEfficiencyRow[] = [
  { label: "Full Fine-Tuning", trainedParamsPct: 100, performancePct: 100 },
  { label: "Adapter Tuning", trainedParamsPct: 3.6, performancePct: 96.1 },
];

// Bottleneck 维度:d(BERT hidden size) vs r(bottleneck 维度)
export const BOTTLENECK_DIM = {
  d: 768,
  r: 64,
  layers: 12,
  paramsPerLayer: 2 * 768 * 64, // down(768→64) + up(64→768) ≈ 98K
  paramsPerTask: 1.2e6, // 12 层 × 98K ≈ 1.2M / 任务
};

// GLUE 9 任务逐项对比(Houlsby 2019,BERT-large)
export interface GlueTaskRow {
  task: string;
  fullFt: number;
  adapter: number;
}
export const GLUE_TASKS: GlueTaskRow[] = [
  { task: "MNLI", fullFt: 86.7, adapter: 84.9 },
  { task: "QQP", fullFt: 89.6, adapter: 88.3 },
  { task: "QNLI", fullFt: 92.7, adapter: 91.4 },
  { task: "SST-2", fullFt: 94.9, adapter: 93.5 },
  { task: "CoLA", fullFt: 60.5, adapter: 56.9 },
  { task: "STS-B", fullFt: 86.5, adapter: 84.7 },
  { task: "MRPC", fullFt: 89.3, adapter: 86.9 },
  { task: "RTE", fullFt: 70.1, adapter: 71.8 },
];

// 零初始化行为:训练前(adapter=identity) vs 训练后(adapter 学到偏移)
export interface ZeroInitFrame {
  label: string;
  wUpNorm: number; // W_up 的（示意）范数大小,0 表示零初始化
  outputShift: number; // 输出相对 base 表征偏移量(示意)
  description: string;
}
export const ZERO_INIT_FRAMES: Record<"before" | "after", ZeroInitFrame> = {
  before: {
    label: "训练开始(零初始化)",
    wUpNorm: 0,
    outputShift: 0,
    description: "W_up ≈ 0,adapter 输出 h ≈ x,等价于恒等映射,不 disrupt 预训练表示",
  },
  after: {
    label: "训练收敛(学到任务偏移)",
    wUpNorm: 1,
    outputShift: 1,
    description: "W_up 学到非零权重,adapter 输出 h = x + Δ,Δ 承担任务特化的小幅偏移",
  },
};
