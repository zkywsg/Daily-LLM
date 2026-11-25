// Kaplan power law constants
export const ALPHA_N = 0.076;
export const ALPHA_D = 0.095;
export const ALPHA_C = 0.05;
export const NC = 8.8e13;
export const DC = 5.4e13;
export const CC = 3.1e8;        // 任意,曲线相对值才重要

export function lossN(N: number): number {
  return Math.pow(NC / N, ALPHA_N);
}
export function lossD(D: number): number {
  return Math.pow(DC / D, ALPHA_D);
}
export function lossC(C: number): number {
  return Math.pow(CC / C, ALPHA_C);
}

// Chinchilla 计算器
export function chinchillaOptimal(flops: number): { N: number; D: number } {
  const N = Math.sqrt(flops / 120);
  const D = 20 * N;
  return { N, D };
}

// 实际模型点位
export interface ModelPoint {
  name: string;
  params: number;   // 实际参数(单位:个)
  data: number;     // 实际训练 token(单位:个)
  year: number;
  ratio: number;    // D / N
  category: "kaplan-era" | "chinchilla" | "llama-overtrain";
}

export const MODEL_POINTS: ModelPoint[] = [
  { name: "GPT-3 175B",     params: 175e9, data: 300e9, year: 2020, ratio:   1.7, category: "kaplan-era"     },
  { name: "Gopher 280B",    params: 280e9, data: 300e9, year: 2021, ratio:   1.1, category: "kaplan-era"     },
  { name: "MT-NLG 530B",    params: 530e9, data: 270e9, year: 2022, ratio:   0.5, category: "kaplan-era"     },
  { name: "Chinchilla 70B", params:  70e9, data: 1.4e12, year: 2022, ratio:  20.0, category: "chinchilla"    },
  { name: "LLaMA-1 7B",     params:   7e9, data: 1.0e12, year: 2023, ratio: 143.0, category: "llama-overtrain" },
  { name: "LLaMA-2 7B",     params:   7e9, data: 2.0e12, year: 2023, ratio: 286.0, category: "llama-overtrain" },
  { name: "LLaMA-3 8B",     params:   8e9, data: 15.0e12, year: 2024, ratio: 1875.0, category: "llama-overtrain" },
  { name: "Mistral 7B",     params:   7e9, data: 8.0e12, year: 2023, ratio: 1143.0, category: "llama-overtrain" },
];

// Wei et al. 2022 涌现示例:同一任务在 0/1 acc vs token-level acc 下的不同曲线
// 横轴 log10(FLOPs) 22..25,纵轴 0..100%
export function emergentDiscreteAcc(logF: number): number {
  // 类似阶跃 — sigmoid 跳变于 24.0
  const v = 1 / (1 + Math.exp(-(logF - 24.0) * 4));
  return Math.min(100, Math.max(0, v * 100));
}

export function emergentContinuousScore(logF: number): number {
  // 平滑 — 线性回升
  const norm = (logF - 22) / (25 - 22);
  return Math.min(100, Math.max(0, norm * 100));
}
