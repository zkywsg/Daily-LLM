// 显存账:fp16 LoRA vs QLoRA(LLaMA-65B)
export interface MemoryStackRow {
  method: string;
  base: number;
  lora: number;
  optimizer: number;
  activation: number;
}
export const MEMORY_STACK: MemoryStackRow[] = [
  { method: "fp16 LoRA", base: 130, lora: 0.06, optimizer: 15, activation: 5 },
  { method: "QLoRA(NF4 + DQ + Paged)", base: 33, lora: 0.06, optimizer: 3, activation: 5 },
];

// NF4 的 16 个量化值(标准正态分布等分位点)
export const NF4_LEVELS: number[] = [
  -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1849, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.723, 1.0,
];

// INT4 均匀分布的 16 个量化值
export const INT4_LEVELS: number[] = Array.from({ length: 16 }, (_, i) => -1 + (i * 2) / 15);

// 标准正态分布 PDF(用于画权重分布曲线)
export function normalPdf(x: number, sigma = 0.35): number {
  return Math.exp(-(x * x) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
}

// Double Quantization:scale factor 元数据开销
export interface DqRow {
  label: string;
  scaleBits: number;
  totalMetadataKB: number;
}
export const DQ_COMPARE: DqRow[] = [
  { label: "单重量化(fp32 scale)", scaleBits: 32, totalMetadataKB: 8 },
  { label: "双重量化(8-bit scale)", scaleBits: 8, totalMetadataKB: 2.1 },
];

// Paged Optimizer:训练过程中 GPU 显存占用模拟(峰值 spike 被分页削平)
export interface PagingStep {
  step: number;
  withoutPaging: number; // GB
  withPaging: number; // GB
}
export const PAGING_TIMELINE: PagingStep[] = [
  { step: 0, withoutPaging: 33, withPaging: 33 },
  { step: 1, withoutPaging: 38, withPaging: 34 },
  { step: 2, withoutPaging: 48, withPaging: 35 }, // 显存 spike(梯度检查点重算 / batch 边界)
  { step: 3, withoutPaging: 40, withPaging: 34 },
  { step: 4, withoutPaging: 46, withPaging: 35 }, // 又一次 spike
  { step: 5, withoutPaging: 39, withPaging: 34 },
];
export const GPU_CAPACITY_GB = 41; // A6000 系列可用显存(演示阈值)

// Guanaco 系列性能对比
export interface GuanacoRow {
  model: string;
  hardware: string;
  vicunaScore: number;
}
export const GUANACO_SCORES: GuanacoRow[] = [
  { model: "ChatGPT(2023.3)", hardware: "-", vicunaScore: 100 },
  { model: "GPT-4", hardware: "-", vicunaScore: 119 },
  { model: "Vicuna-13B(FP16 LoRA)", hardware: "8× A100-40GB", vicunaScore: 92 },
  { model: "Guanaco-7B(QLoRA)", hardware: "单 RTX 3090", vicunaScore: 87 },
  { model: "Guanaco-13B(QLoRA)", hardware: "单 A100-40GB", vicunaScore: 93 },
  { model: "Guanaco-33B(QLoRA)", hardware: "单 A6000-48GB", vicunaScore: 97 },
  { model: "Guanaco-65B(QLoRA)", hardware: "单 A6000-48GB", vicunaScore: 99.3 },
];
