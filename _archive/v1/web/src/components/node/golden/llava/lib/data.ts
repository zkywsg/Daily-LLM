// 架构组件:可训练性与参数量
export interface ComponentRow {
  name: string;
  trainable: string;
  params: string;
}
export const ARCHITECTURE_COMPONENTS: ComponentRow[] = [
  { name: "CLIP ViT-L/14", trainable: "冻结", params: "~300M" },
  { name: "Projection(Linear/MLP)", trainable: "唯一从零训练", params: "~4-30M" },
  { name: "LLaMA / Vicuna", trainable: "Stage1 冻结,Stage2 全微调", params: "7B / 13B" },
];

// 训练成本对比(GPU 小时)
export interface CostRow {
  model: string;
  gpuHours: number;
  costUSD: number;
}
export const TRAINING_COST_COMPARE: CostRow[] = [
  { model: "Flamingo-80B", gpuHours: 360000, costUSD: 1_000_000 },
  { model: "BLIP-2", gpuHours: 3456, costUSD: 10_000 },
  { model: "LLaVA-7B", gpuHours: 96, costUSD: 200 },
];

// 桥接模块参数量对比
export interface BridgeParamRow {
  method: string;
  paramsM: number;
}
export const BRIDGE_PARAM_COMPARE: BridgeParamRow[] = [
  { method: "LLaVA(单 Linear)", paramsM: 4 },
  { method: "LLaVA-1.5(2 层 MLP)", paramsM: 30 },
  { method: "BLIP-2 Q-Former", paramsM: 188 },
  { method: "Flamingo Perceiver Resampler", paramsM: 400 },
];

// 两阶段训练
export interface StageInfo {
  stage: string;
  dataSize: string;
  trainable: string;
  gpuTime: string;
}
export const TWO_STAGE_TRAINING: StageInfo[] = [
  { stage: "Stage 1: Feature Alignment", dataSize: "CC3M 595K image-caption 对", trainable: "只训 projection,冻结 CLIP + LLM", gpuTime: "8×A100×4 小时" },
  { stage: "Stage 2: Visual Instruction Tuning", dataSize: "158K GPT-4 生成的多模态指令", trainable: "projection + LLM 全微调", gpuTime: "8×A100×8 小时" },
];

// LLaVA-Bench 对比
export interface BenchRow {
  model: string;
  conversation: number;
  detail: number;
  reasoning: number;
  overall: number;
}
export const LLAVA_BENCH: BenchRow[] = [
  { model: "BLIP-2", conversation: 54.6, detail: 29.1, reasoning: 32.9, overall: 38.1 },
  { model: "MiniGPT-4", conversation: 65.0, detail: 67.3, reasoning: 76.6, overall: 69.7 },
  { model: "LLaVA", conversation: 83.1, detail: 75.3, reasoning: 96.5, overall: 85.1 },
  { model: "GPT-4(text-only)", conversation: 88.5, detail: 89.4, reasoning: 98.6, overall: 92.1 },
];

// ScienceQA 对比
export interface ScienceQaRow {
  model: string;
  accuracy: number;
}
export const SCIENCEQA_COMPARE: ScienceQaRow[] = [
  { model: "GPT-3.5 + CoT", accuracy: 75.2 },
  { model: "GPT-4 + CoT", accuracy: 84.9 },
  { model: "LLaMA-Adapter", accuracy: 78.3 },
  { model: "LLaVA + GPT-4 judge", accuracy: 92.5 },
];
