// R1-Zero 训练过程中思考长度 + AIME 准确率的联合增长(论文演示数据)
export interface TrainingPoint {
  step: number;
  thinkTokens: number;
  aime: number;
  hasAha: boolean;
}
export const TRAINING_STEP_GROWTH: TrainingPoint[] = [
  { step: 0, thinkTokens: 100, aime: 15.6, hasAha: false },
  { step: 2000, thinkTokens: 1200, aime: 32, hasAha: false },
  { step: 4000, thinkTokens: 3500, aime: 48, hasAha: true },
  { step: 6000, thinkTokens: 6500, aime: 60, hasAha: true },
  { step: 8000, thinkTokens: 10000, aime: 71.0, hasAha: true },
];

// GRPO 组内采样演示:同一 prompt 采 G 个 response,组内归一化得 advantage
export interface GrpoSample {
  idx: number;
  reward: number;
  advantage: number;
}
export function simulateGrpoGroup(rewards: number[]): GrpoSample[] {
  const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
  const variance = rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / rewards.length;
  const std = Math.sqrt(variance) || 1;
  return rewards.map((r, i) => ({ idx: i, reward: r, advantage: (r - mean) / std }));
}
export const DEMO_GROUP_REWARDS: number[] = [1, 0, 1, 0, 1, 1, 0, 0];

// R1 多阶段训练 pipeline
export interface StageInfo {
  name: string;
  short: string;
  desc: string;
}
export const MULTI_STAGE_PIPELINE: StageInfo[] = [
  { name: "Stage 1: Cold-start SFT", short: "冷启动 SFT", desc: "几千条高质量长链推理数据(部分来自 R1-Zero 输出 + 人工清洗),让模型先学会可读的推理格式" },
  { name: "Stage 2: Reasoning RL", short: "推理 RL", desc: "大规模 GRPO on 数学/代码/逻辑题,reward = 正确性 + 语言一致性 — 这一步是 reasoning 内核的真正来源" },
  { name: "Stage 3: Rejection Sampling SFT", short: "拒绝采样 SFT", desc: "从 Stage 2 模型采样大量回答,过滤高质量样本,加入通用任务数据(写作/QA/role-play)" },
  { name: "Stage 4: RLHF", short: "RLHF 对齐", desc: "类似 InstructGPT 的 RLHF,让模型既能推理也能对齐人类偏好" },
];

// benchmark 对比(R1 论文数据)
export interface BenchRow {
  benchmark: string;
  deepseekV3: number;
  r1zero: number | null;
  r1: number;
  o1mini: number;
  o1: number;
}
export const BENCHMARK_COMPARE: BenchRow[] = [
  { benchmark: "AIME 2024", deepseekV3: 39.2, r1zero: 71.0, r1: 79.8, o1mini: 63.6, o1: 79.2 },
  { benchmark: "MATH-500", deepseekV3: 90.2, r1zero: null, r1: 97.3, o1mini: 90.0, o1: 96.4 },
  { benchmark: "Codeforces %ile", deepseekV3: 58.7, r1zero: null, r1: 96.3, o1mini: 93.4, o1: 96.6 },
  { benchmark: "GPQA Diamond", deepseekV3: 59.1, r1zero: null, r1: 71.5, o1mini: 60.0, o1: 75.7 },
];

// 蒸馏模型表现
export interface DistillRow {
  model: string;
  aime: number;
  math500: number;
}
export const DISTILL_TABLE: DistillRow[] = [
  { model: "R1-Distill-Qwen-1.5B", aime: 28.9, math500: 83.9 },
  { model: "R1-Distill-Qwen-7B", aime: 55.5, math500: 92.8 },
  { model: "R1-Distill-Llama-8B", aime: 50.4, math500: 89.1 },
  { model: "R1-Distill-Qwen-32B", aime: 72.6, math500: 94.3 },
  { model: "R1-Distill-Llama-70B", aime: 70.0, math500: 94.5 },
];
