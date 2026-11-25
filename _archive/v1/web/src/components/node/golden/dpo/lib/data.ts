// DPO 训练动力学模拟
export interface RewardCurve {
  step: number;
  chosenLogp: number;     // actor on chosen
  rejectedLogp: number;   // actor on rejected
  refChosen: number;      // ref on chosen (常数,作 baseline)
  refRejected: number;    // ref on rejected
  chosenReward: number;   // β · (chosenLogp - refChosen)
  rejectedReward: number; // β · (rejectedLogp - refRejected)
}

// 根据 β 和 lr 模拟 100 步训练里 chosen / rejected log-prob 走势
export function simulateTraining(beta: number, lr: number, steps: number = 100): RewardCurve[] {
  const refChosen = -2.5;
  const refRejected = -2.7;
  const out: RewardCurve[] = [];

  let chosenLogp = refChosen;
  let rejectedLogp = refRejected;

  // β 小 + lr 大 → 激进偏向 chosen,但偏离 ref 越严重
  // β 大 → 被 ref 拉住,变化慢
  for (let i = 0; i <= steps; i++) {
    out.push({
      step: i,
      chosenLogp,
      rejectedLogp,
      refChosen,
      refRejected,
      chosenReward: beta * (chosenLogp - refChosen),
      rejectedReward: beta * (rejectedLogp - refRejected),
    });
    // 梯度:推高 chosen, 压低 rejected,幅度被 β 抑制
    const margin = beta * ((chosenLogp - refChosen) - (rejectedLogp - refRejected));
    const sigmoidMargin = 1 / (1 + Math.exp(margin));
    const grad = sigmoidMargin * lr;
    chosenLogp += grad / beta;
    rejectedLogp -= grad / beta;
  }
  return out;
}

// PPO vs DPO 模型/资源对比
export interface MethodSpec {
  name: string;
  models: number;
  hparams: number;
  codeLines: number;
  trainTime: string;
  cost: number;     // PPO 相对 1.0
  isDpo: boolean;
}

export const METHOD_COMPARE: MethodSpec[] = [
  { name: "PPO (InstructGPT)", models: 4, hparams: 14, codeLines: 5000, trainTime: "几天 · 多节点", cost: 1.0,  isDpo: false },
  { name: "DPO",               models: 2, hparams: 2,  codeLines: 300,  trainTime: "几小时 · 单节点",  cost: 0.05, isDpo: true  },
];

// DPO family timeline
export interface DpoVariant {
  name: string;
  fullName: string;
  year: number;
  month: number;
  authors: string;
  oneLiner: string;
  color: string;
}

export const DPO_FAMILY: DpoVariant[] = [
  { name: "DPO",  fullName: "Direct Preference Optimization",     year: 2023, month: 5,  authors: "Rafailov et al.",    oneLiner: "原始方法 · log-ratio + Bradley-Terry",          color: "#ec4899" },
  { name: "IPO",  fullName: "Identity Preference Optimization",   year: 2023, month: 10, authors: "Azar et al.",         oneLiner: "把 sigmoid 换 MSE · 对噪声偏好更鲁棒",          color: "#f59e0b" },
  { name: "KTO",  fullName: "Kahneman-Tversky Optimization",      year: 2024, month: 2,  authors: "Ethayarajh et al.",   oneLiner: "二元 like/dislike · 不需要成对比较",            color: "#3b82f6" },
  { name: "ORPO", fullName: "Odds Ratio Preference Optimization", year: 2024, month: 3,  authors: "Hong et al.",         oneLiner: "SFT + DPO 合并 · 一步完成",                     color: "#10b981" },
  { name: "SimPO", fullName: "Simple Preference Optimization",    year: 2024, month: 5,  authors: "Meng et al.",         oneLiner: "去掉 reference model · length-normalized prob",  color: "#a855f7" },
];

// 论文 benchmark
export interface BenchScore {
  method: string;
  imdb: number;
  hhWinRate: number;
}

export const BENCHMARKS: BenchScore[] = [
  { method: "PPO",          imdb: 0.71, hhWinRate: 56 },
  { method: "DPO",          imdb: 0.72, hhWinRate: 64 },
  { method: "Best-of-N",    imdb: 0.70, hhWinRate: 53 },
];
