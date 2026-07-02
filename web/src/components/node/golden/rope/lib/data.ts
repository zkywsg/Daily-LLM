// 2D 旋转:给定位置 m 和 base theta,计算旋转后的向量
export function rotate2D(x: number, y: number, angle: number): [number, number] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return [x * cos - y * sin, x * sin + y * cos];
}

// 多频率:d/2 个 2D 平面的 theta_i
export function thetaFor(i: number, d: number, base: number = 10000): number {
  return Math.pow(base, (-2 * i) / d);
}

// 相对位置内积不变性演示数据
export interface DotProductDemo {
  m: number;
  n: number;
  dotProduct: number;
}

// 长程衰减曲线:不同相对距离下的平均内积衰减(模拟)
export function decayAtDistance(dist: number, d: number = 64): number {
  // 模拟:多个频率求和后随距离增大而振荡衰减(简化模型)
  let sum = 0;
  const numFreqs = d / 2;
  for (let i = 0; i < numFreqs; i++) {
    const theta = thetaFor(i, d);
    sum += Math.cos(dist * theta);
  }
  return sum / numFreqs;
}

// 频率维度对比:高频 vs 低频
export interface FreqDim {
  name: string;
  index: number;
  theta: number;
  period: number; // 2π/theta,旋转一圈需要的距离
  role: string;
}
export function buildFreqDims(d: number = 64): FreqDim[] {
  const indices = [0, 8, 16, 24, 31];
  return indices.map((i) => {
    const theta = thetaFor(i, d);
    return {
      name: `dim ${i * 2}`,
      index: i,
      theta,
      period: (2 * Math.PI) / theta,
      role: theta > 0.1 ? "高频(短距)" : theta > 0.001 ? "中频" : "低频(长距)",
    };
  });
}

// PE 方案对比时间线
export interface PeModel {
  model: string;
  year: number;
  peType: "learned" | "sinusoidal" | "relative-bias" | "rope" | "alibi";
}
export const PE_TIMELINE: PeModel[] = [
  { model: "Transformer", year: 2017, peType: "sinusoidal" },
  { model: "BERT",        year: 2018, peType: "learned" },
  { model: "GPT-3",       year: 2020, peType: "learned" },
  { model: "T5",          year: 2019, peType: "relative-bias" },
  { model: "GPT-NeoX",    year: 2022, peType: "rope" },
  { model: "PaLM",        year: 2022, peType: "rope" },
  { model: "LLaMA",       year: 2023, peType: "rope" },
  { model: "Mistral",     year: 2023, peType: "rope" },
  { model: "BLOOM",       year: 2022, peType: "alibi" },
];

// 长上下文扩展方法
export interface ExtMethod {
  name: string;
  desc: string;
  from: string;
  to: string;
}
export const EXTENSION_METHODS: ExtMethod[] = [
  { name: "Position Interpolation", desc: "把频率缩小 L_new/L_train 倍", from: "2K", to: "16K" },
  { name: "NTK-aware Scaling",       desc: "高频保持,低频按比例缩放",   from: "2K", to: "32K" },
  { name: "YaRN",                    desc: "组合 PI + NTK + 温度调整",  from: "2K", to: "128K" },
];
