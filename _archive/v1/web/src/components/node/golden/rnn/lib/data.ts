export function tanh(x: number): number {
  return Math.tanh(x);
}
export function tanhGrad(x: number): number {
  const t = Math.tanh(x);
  return 1 - t * t;
}

// 模拟隐状态递推序列
export interface HiddenTrace {
  step: number;
  h: number;
  gradToH1: number; // 该步梯度回传到 h_1 的幅度(模拟)
}

export function simulateBPTT(spectralRadius: number, steps: number = 20): HiddenTrace[] {
  const out: HiddenTrace[] = [];
  let h = 0.3;
  let gradAccum = 1.0;
  for (let t = 0; t < steps; t++) {
    const preAct = spectralRadius * h + 0.4 * Math.sin(t * 0.6);
    h = tanh(preAct);
    // 梯度按 tanhGrad * spectralRadius 连乘(从末端往前累积到该步）
    const g = tanhGrad(preAct) * spectralRadius;
    gradAccum *= g;
    out.push({ step: t, h, gradToH1: gradAccum });
  }
  return out;
}

// 权重共享参数量对比:共享 vs 不共享
export function sharedParams(hiddenDim: number, inputDim: number): number {
  return hiddenDim * hiddenDim + hiddenDim * inputDim + hiddenDim; // W_h + W_x + bias
}
export function unsharedParams(hiddenDim: number, inputDim: number, steps: number): number {
  return sharedParams(hiddenDim, inputDim) * steps;
}

// 简单序列记忆任务演示:记住第一个词，若干步后复述
export const MEMORY_TASK_SEQ = ["cat", "the", "sat", "on", "mat", "?"];
export const MEMORY_TASK_ANSWER_IDX = 0; // "cat" 是需要记住的词,在第 5 步(?)需要回忆
