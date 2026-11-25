// 三代 reasoning 范式对比(GSM8K 演示数据)
export interface ParadigmRow {
  stage: string;
  thinkTokens: number;
  accuracy: number;
  desc: string;
}
export const PARADIGM_COMPARE: ParadigmRow[] = [
  { stage: "CoT", thinkTokens: 50, accuracy: 57, desc: "单次 forward + 短 thinking,prompt 触发" },
  { stage: "Self-Consistency", thinkTokens: 250, accuracy: 75, desc: "同 prompt 采 5 次 + majority vote" },
  { stage: "o1", thinkTokens: 8000, accuracy: 95, desc: "模型内部长 thinking(反思/回溯/自验证),训练进模型权重" },
];

// RL 训练阶段:模型逐渐学会长 thinking
export interface RlStage {
  step: string;
  text: string;
  reward: number;
  hasReflection: boolean;
}
export const RL_TRAINING_STAGES: RlStage[] = [
  { step: "初始 base 模型", text: "答案是 42。", reward: 0.05, hasReflection: false },
  { step: "RL 训了几百步", text: "让我先理解...设 x = ...计算...答案是 17。", reward: 0.3, hasReflection: false },
  {
    step: "RL 训了几千步",
    text: "让我先理解题目。\n尝试方法 1:坐标几何...算到一半发现太复杂。\nWait,换个角度,用对称性...\n等等,我刚才那步公式用错了,重新算...\n验证:把答案代回...对的。\n答案是 23。",
    reward: 0.9,
    hasReflection: true,
  },
];

// Test-time compute scaling:thinking tokens (log) vs 准确率,三个任务
export interface ScalingPoint {
  thinkTokens: number;
  accuracy: number;
}
export const TEST_TIME_SCALING: Record<string, ScalingPoint[]> = {
  AIME: [
    { thinkTokens: 500, accuracy: 20 },
    { thinkTokens: 2000, accuracy: 45 },
    { thinkTokens: 8000, accuracy: 65 },
    { thinkTokens: 32000, accuracy: 78 },
    { thinkTokens: 64000, accuracy: 83 },
  ],
  Codeforces: [
    { thinkTokens: 500, accuracy: 15 },
    { thinkTokens: 2000, accuracy: 35 },
    { thinkTokens: 8000, accuracy: 55 },
    { thinkTokens: 32000, accuracy: 72 },
    { thinkTokens: 64000, accuracy: 80 },
  ],
  GPQA: [
    { thinkTokens: 500, accuracy: 40 },
    { thinkTokens: 2000, accuracy: 55 },
    { thinkTokens: 8000, accuracy: 68 },
    { thinkTokens: 32000, accuracy: 75 },
    { thinkTokens: 64000, accuracy: 78 },
  ],
};

// 训练算力 vs 推理算力 的性价比对比(演示用简化数字)
export interface ComputeTradeoff {
  axis: string;
  multiplier: string;
  accuracyGain: string;
}
export const COMPUTE_TRADEOFF: ComputeTradeoff[] = [
  { axis: "传统 scaling(训练算力)", multiplier: "×10", accuracyGain: "+10%" },
  { axis: "o1 scaling(推理算力)", multiplier: "×100", accuracyGain: "+70%" },
];

// benchmark 对比表(o1 论文数据)
export interface BenchRow {
  benchmark: string;
  gpt4o: number;
  o1preview: number;
  o1: number;
  human: number;
}
export const BENCHMARK_COMPARE: BenchRow[] = [
  { benchmark: "AIME 2024", gpt4o: 13.4, o1preview: 56.7, o1: 83.3, human: 95 },
  { benchmark: "MATH-500", gpt4o: 60.3, o1preview: 85.5, o1: 94.8, human: 95 },
  { benchmark: "Codeforces %ile", gpt4o: 11, o1preview: 62, o1: 89, human: 50 },
  { benchmark: "GPQA Diamond", gpt4o: 50.6, o1preview: 73.3, o1: 77.3, human: 69.7 },
];
